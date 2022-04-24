import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender


def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    #print(N_rays_all)
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        #print("running chunk!",chunk_idx)
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    #print(len(rgbs))
    #print(rgbs[0].size())
    trgbs = torch.cat(rgbs)
    #print(trgbs.size())
    return trgbs, None, torch.cat(depth_maps), None, None

@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass
        
    W, H = test_dataset.img_wh
    near_far = test_dataset.near_far
    totalimgnum = test_dataset.all_rays.shape[0] // (W*H) #total number of images in the testdataset.
    #img_eval_interval = 1 if N_vis < 0 else test_dataset.all_rays.shape[0] // N_vis
    img_eval_interval = max(1,totalimgnum // N_vis) #interval between two images (1= every images is rendered, 2=every other image, 3=every 3rd etc)
    iidxs = list(range(0, totalimgnum, img_eval_interval))
    #print("evalconditions",idxs,img_eval_interval,near_far)
    #mr evalconditions [0, 589824, 1179648, 1769472, 2359296, 2949120, 3538944, 4128768, 4718592, 5308416, 5898240, 6488064, 7077888, 7667712, 8257536, 8847360] 589824 [0.4, 6.0]
    #exit()
    """
    rays should be: rays tensor([[ 0.9898,  0.4947,  0.5719, -0.4993, -0.3520,  0.7917],
        [ 0.9898,  0.4947,  0.5719, -0.6454, -0.3635,  0.6719],
        [ 0.8594,  0.3737,  0.5081, -0.3992, -0.3140,  0.8614],
        ...,
        [ 0.9898,  0.4947,  0.5719, -0.8712, -0.2401,  0.4283],
        [ 0.6870,  0.3917,  0.4425, -0.2424,  0.0358,  0.9695],
        [ 0.9898,  0.4947,  0.5719, -0.2232, -0.1485,  0.9634]],
       device='cuda:0')
    """
    #instead it is: raysize tensor([[ 0.6870,  0.3917,  0.4425, -0.7387, -0.3906,  0.5493]])
    #print()
    #print("allrays",test_dataset.all_rays)
    for iidx in tqdm(iidxs, file=sys.stdout):
        W, H = test_dataset.img_wh
        startidx = iidx*W*H #start ray of the image in question.
        endidx = startidx+W*H #end ray of the image in question.
        samples  = test_dataset.all_rays[startidx:endidx]
        rays = samples
        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        #print("mapsize:",rgb_map.size())
        rgb_map = rgb_map.clamp(0.0, 1.0)
        
        #print(rgb_map.size())
        rgb_map = rgb_map.reshape(H, W, 3).cpu()
        depth_map = depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[startidx:endidx].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        #rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        #rgb_maps.append(rgb_map)
        #depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{iidx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{iidx:03d}.png', rgb_map)

    #imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    #imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

