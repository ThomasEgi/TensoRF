import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np
from .ray_utils import *
from math import sqrt

from panda3d.core import NodePath, LMatrix4f , LVecBase3#because fuckall, there seems to be no accelerated batch-capable way for doing transforms or even proper vectorxmatrix operations.
from panda3d.core import GeomVertexFormat,GeomVertexData,Geom,GeomNode,GeomVertexWriter,GeomPoints,GeomEnums

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


class MeshRoom(Dataset):
    """Meshroom Dataset."""
    def __init__(self, datadir, split='train', downsample=1.0, wh=[1320,979], is_stack=False):
        
        self.root_dir = datadir
        self.mgdata = None
        self.__loadMainDataFile()
        self.__loadBbFromMgdata()
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        print (self.center, self.radius)
        #exit()
        self.split = split
        self.is_stack = is_stack
        self.downsample = downsample
        self.img_wh = (int(wh[0]/downsample),int(wh[1]/downsample))
        self.define_transforms()

        self.white_bg = True
        self.near_far = [0.1,5.0] ##we need to smartly calculate this on per-camera-perspective basis.
        #self.scene_bbox = torch.from_numpy(np.loadtxt(f'{self.root_dir}/bbox.txt')).float()[:6].view(2,3)
        #self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #self.read_meta()
        #self.define_proj_mat()
        self.__read_meta()
        
        #exit()
    def __loadBbFromMgdata(self):
        #Meshing_
        for key in self.mgdata['graph'].keys():
            if key.startswith('Meshing_'):
                dmeshing = self.mgdata['graph'][key]
                if dmeshing['inputs']['useBoundingBox'] == False:
                    print("F*Ck0FF. dataset contains no Bounding box. I was so not made for this!\n come back when you added one for the mesh in meshroom!")
                    exit()
                bb = dmeshing['inputs']['boundingBox']
                x,y,z = bb['bboxTranslation'].values()
                #we are boldly ignoring rotation for now
                h,p,r = bb['bboxRotation'].values()
                #we are also ignoring individual bb-dimensions and use the radius instead. or the max dimensions, we'll see which one is more useful.
                sx,sy,sz = bb['bboxScale'].values()
                from math import sqrt
                radius = sqrt(sx**2+sy**2+sz**2)/2.
                maxdim = max(sx,sy,sz)
                self.scene_bsphere = [x,y,z,maxdim] #maybe maxdim .5 ?
                self.scene_bbox = torch.from_numpy(np.array([[x-maxdim,y-maxdim,z-maxdim],[x+maxdim,y+maxdim,z+maxdim]],dtype=np.float32))
                
    def __loadMainDataFile(self):
        #self.root_dir
        import json 
        #self.root_dir
        for ifile in os.listdir(self.root_dir):
            print(ifile)
            if ifile.endswith(".mg"):
                dfile = os.path.join(self.root_dir, ifile)
                print(dfile)
                with open(dfile,"r") as f:
                    data = f.read()
                    self.mgdata = json.loads(data)
                    return
        print("mgdata not found!")
    
    def __read_meta(self):
        
        ###
        ##need poses, images, and camera intrinsics
        ##maybe we should just randomly train and evaluate images on the fly instead of fixed sets.maybe not. who knows.
        sfm1 = self.mgdata['graph']['StructureFromMotion_1'] #we should check if it is _1. but hey.
        sfmuid  = sfm1['uids']['0'] #let's just hope it's there.
        sfmnodetype = sfm1['nodeType'] #nodetype-string, we need this for the correct path.
        #inpts = cint1['inputs']
        cdictname = "MeshroomCache"
        
        import json  #again yeah.
        #self.root_dir
        sfmfpath = os.path.join(self.root_dir,cdictname,sfmnodetype,sfmuid,"cameras.sfm") #bit more hardcoding. which is ok unless you fkd up your meshroom stuff for good.
        self.camerasfmdata = None
        with open(sfmfpath,"r") as f:
            data = f.read()
            self.camerasfmdata = json.loads(data)
            print("loaded sfm camera data")
                    #return
        #print("mgdata not found!")
        if not self.camerasfmdata:
            print("wtf, camera data not loaded?")
            exit()
        
        camintrinsics = self.camerasfmdata['intrinsics']
        self.camintrinsics = {}
        for intr in camintrinsics:
            intrinsicId= intr['intrinsicId']
            print(intrinsicId)
            self.camintrinsics[intrinsicId] = intr
        
        
        camposes = self.camerasfmdata['poses']
        self.camposes = {}
        for pose in camposes:
            poseID= pose['poseId']
            print(poseID)
            self.camposes[poseID] = pose['pose']
        
        print(self.camposes)
        
        #with open(os.path.join(self.root_dir, "intrinsics.txt")) as f:
        #    focal = float(f.readline().split()[0])
        ##did you actually hardcode ... everything? bro...i wondered why the heck none of my datasets worked.
        #focal = 300
        #self.intrinsics = np.array([[focal,0,400.0],[0,focal,400.0],[0,0,1]])
        #print(self.intrinsics)
        #self.intrinsics[:2] *= (np.array(self.img_wh)/np.array([800,800])).reshape(2,1)
        #print(self.intrinsics )
        ##srsly...
        #return
        #we don't even need the intrinsics outside this class. probably not even outside this method.
        
        #let's skipp that crap. jump straight to views
        
        #pose_files = sorted(os.listdir(os.path.join(self.root_dir, 'pose')))
        #img_files  = sorted(os.listdir(os.path.join(self.root_dir, 'rgb')))

        #if self.split == 'train':
        #    pose_files = [x for x in pose_files if x.startswith('0_')]
        #    img_files = [x for x in img_files if x.startswith('0_')]
        #elif self.split == 'val':
        #    pose_files = [x for x in pose_files if x.startswith('1_')]
        #    img_files = [x for x in img_files if x.startswith('1_')]
        #elif self.split == 'test':
        #    test_pose_files = [x for x in pose_files if x.startswith('2_')]
        #    test_img_files = [x for x in img_files if x.startswith('2_')]
        #    if len(test_pose_files) == 0:
        #        test_pose_files = [x for x in pose_files if x.startswith('1_')]
        #        test_img_files = [x for x in img_files if x.startswith('1_')]
        #    pose_files = test_pose_files
        #    img_files = test_img_files

        # ray directions for all pixels, same for all images (same H, W, focal)
        #time to go for views.
        
        self.render_path = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        print(self.render_path)
        
        views = self.camerasfmdata['views']
        
        dummyRootNP = NodePath("dummyRoot")
        
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        temprays = []
        numviews=0
        totalidx = 0
        for view in views:
            numviews +=1
            if 'train' == self.split:
                if numviews > len(views)-3: #we can use the last 3 for training too. but then it's not independent.
                    pass
                    continue
            else:
                if numviews <= len(views)-3:
                    continue
            pose = self.camposes[view['poseId']]
            intr = self.camintrinsics[view['intrinsicId']]
            imgpath = view['path']
            w,h = int(view['width']), int(view['height'])
            f = float(intr["pxFocalLength"])
            cx, cy = intr['principalPoint']
            cx, cy = float(cx),float(cy)
            distortionparams = intr['distortionParams'] #  meshroom spits out k1,k2,k3  but opencv and our undistort 5 wants k1 k2 p1 p2 k3
            distlist = []
            for dparam in distortionparams:
                distlist.append(float(dparam))
            #we use undistort with 0 params. just fill up to 5 all the time. 
            while len(distlist) <5:
                distlist.append(0)
            #distlist2 = []
            #distlist2.append(distlist[0],distlist[1],distlist[3],distlist[4],distlist[2])
            #distlist = distlist2
            rot = pose['transform']['rotation']
            pos = pose['transform']['center']
            
            dummyNP = dummyRootNP.attachNewNode("dummynp")
            
            #self.center 
            #self.near_far = [0.1,5.0]#TODO: calculate near and far based on cameras , boundingbox and radius
            
            #panda needs the matrices transposed so yeah, indexing might appear a bit weirdo and is probably different if you use other libraries.
            #in that case you _may_ have more luck with the line provided below.
            #non-panda3d matvals = [ rot[0],rot[1],rot[2],pos[0], rot[3],rot[4],rot[5],pos[1], rot[6],rot[7],rot[8],pos[2], 0,0,0,1]
            matvals = [ rot[0],rot[3],rot[6],0, rot[1],rot[4],rot[7],0, rot[2],rot[5],rot[8],0, pos[0],pos[1],pos[2],1]
            matvalsf = []
            for xmat in matvals:
                matvalsf.append(float(xmat))
            
            #tensor([[ 8.8112e-01,  4.7053e-01, -4.7230e-02,  7.4131e-02],
            #[ 4.7290e-01, -8.7671e-01,  8.8002e-02, -1.3842e-01],
            #[ 3.2000e-08, -9.9875e-02, -9.9500e-01,  2.0632e+00],
            #[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
            #print(matvalsf)
            c2w = torch.FloatTensor([matvalsf[0:4],matvalsf[4:8],matvalsf[8:12],matvalsf[12:16]])
            c2w = torch.transpose(c2w,0,1)
            #print (c2w)
            #exit()
            self.poses.append(c2w)  # C2W
            
            dummyNP.setMat(LMatrix4f(*matvalsf))
            #let's stick with the original image loading code. should be reasonably fine.
            img = Image.open(imgpath)
            img = self.transform(img)  # (4, h, w)
            img = img.view(img.shape[0], -1).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]
            
            ##this section needs work...
            #c2w = np.loadtxt(os.path.join(self.root_dir, 'pose', pose_fname)) #@ self.blender2opencv
            #c2w = torch.FloatTensor(c2w)
            ##self.poses.append(c2w)  # C2W #let's try to get away without
            
            #  get_rays outputs:
            #rays_o: (H*W, 3), the origin of the rays in world coordinate
            #rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
            ray_o1 = dummyNP.getPos() #waste of memory to copy the ray origin gazillion of times. but yeah. can't fix rome in a night.
            #code below ain't a prime example of efficiency, it's more a proof of concept too.
            ray_o1 = [ray_o1[0],ray_o1[1],ray_o1[2]]
            #rays_o = []
            
            #bodydata=GeomVertexData("body vertices", GeomVertexFormat.getV3c4() , Geom.UHStatic)
            #bodydata.setNumRows(h*w)
            #vertex = GeomVertexWriter(bodydata, 'vertex')
            #color = GeomVertexWriter(bodydata, 'color')
            #lastpercent = 0
            #pixidx = 0
            
            ###faster way
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32) + 0.5, #not sure about the +0.5 technically should get you the center of the pixels i guess? but for real cameras?
                torch.arange(w, dtype=torch.float32) + 0.5,
            )
            xx = (xx - cx) / f #could be fx and fy, if we had both separate.
            yy = (yy - cy) / f
            zz = torch.ones_like(xx)
            
            
            #using radial3 model.
            r2 = xx*xx + yy*yy
            r4 = r2*r2
            r6 = r4*r2
            r =  torch.ones_like(xx) + distlist[0]*r2 + distlist[1]*r4 + distlist[2]*r6
                
            #let's skip p1 and p2 parameters
            #dx = 2*p1*x0*y0 + p2*(r2 + 2*x0*x0) 
            #dy = p1*(r2 + 2*y0*y0) + 2*p2*x0*y0
            xx = xx*r#+dx
            yy = yy*r#+dy
            #return np.array((x * fx + cx, y * fy + cy))
            #print(xx,yy)
            
            dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
            dirs = dirs.reshape(1, -1, 3, 1)
            dirs = (c2w[None,:3,:3]@dirs)[...,0][0]
            del xx, yy, zz
            #print(dirs)
            origins = torch.tensor(ray_o1)
            origins = origins.repeat(h*w,1)
            trays = torch.cat((origins,dirs),1)
            #print(trays)
            self.all_rays.append(trays)
            #exit()
            #print((c2w[None,:3,:3]@dirs)[...,0][0])
            #exit()
            #dirs = (c2w[:, None, :3, :3] @ dirs)[..., 0]
            
                
            """
            for py in range(h):
                if int(10*py/h) != lastpercent:
                    lastpercent = int(10*py/h)
                    print(lastpercent,numviews)
                for px in range(w):
                    #print(px)
                    #let's hope we do row-collum-origin the right way.
                    #meshroom spits out k1,k2,k3  but opencv and our undistort 5 wants k1 k2 p1 p2 k3
                    distx, disty = self.distort5(px, py, cx, cy, f,f, distlist[0], distlist[1], distlist[3], distlist[4],distlist[2])
                    #print(distx,px,disty,py)
                    #worldCoordsViewVec = dummyRootNP.getRelativeVector(dummyNP,LVecBase3(px-cx, py-cy, f)).normalized()  #because f is the same for x and y we don't have to divide x and y but we can use z instead
                    worldCoordsViewVec = dummyRootNP.getRelativeVector(dummyNP,LVecBase3(distx-cx, disty-cy, f)).normalized()  #because f is the same for x and y we don't have to divide x and y but we can use z instead
                    temprays.append([ray_o1[0],ray_o1[1],ray_o1[2],worldCoordsViewVec[0],worldCoordsViewVec[1],worldCoordsViewVec[2]])
                    
                    vertex.addData3(ray_o1[0]+worldCoordsViewVec[0]*.5,ray_o1[1]+worldCoordsViewVec[1]*.5,ray_o1[2]+worldCoordsViewVec[2]*.5)
                    c=self.all_rgbs[-1][pixidx]
                    color.addData4(c[0],c[1],c[2],.6)
                    #totalidx+=1
                    pixidx+=1
            """
            
            #primitive = GeomPoints(GeomEnums.UH_static)
            #primitive.add_next_vertices(h*w)
            #geom = Geom(bodydata)
            #geom.add_primitive(primitive)
            #gnode = GeomNode('points')
            #gnode.add_geom(geom)
            #dummyRootNP.attachNewNode(gnode)
            
        self.all_rays = torch.cat(self.all_rays,0)  # (h*w, 6) #merge all the tensors of the individual images into one giant pile.
        print(self.all_rays)
        #exit()            
        dummyRootNP.writeBamFile('./testfile_'+self.split+'.bam')
        print(self.all_rgbs)
        print(len(self.all_rgbs))
        #exit()
        self.all_rgbs = torch.cat(self.all_rgbs, 0)
        #yeah while this is useful stuff. we can't use it as we have to work with a properly distorted real world lens.
        
        
        #self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.intrinsics[0,0],self.intrinsics[1,1]], center=self.intrinsics[:2,2])  # (h, w, 3)
        #self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        ##mind if i ask wtf? is this for outputting actual rendered images? why here?
        
        #self.center
        #self.radius
        #self.poses = []
        #self.all_rays = []
        #self.all_rgbs = []
        """
        assert len(img_files) == len(pose_files)
        for img_fname, pose_fname in tqdm(zip(img_files, pose_files), desc=f'Loading data {self.split} ({len(img_files)})'):
            image_path = os.path.join(self.root_dir, 'rgb', img_fname)
            img = Image.open(image_path)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(img.shape[0], -1).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            c2w = np.loadtxt(os.path.join(self.root_dir, 'pose', pose_fname)) #@ self.blender2opencv
            c2w = torch.FloatTensor(c2w)
            self.poses.append(c2w)  # C2W
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 8)
        """
#             w2c = torch.inverse(c2w)
#
        self.img_wh = (w,h)
        self.poses = torch.stack(self.poses)
        """
        if 'train' == self.split:
            if self.is_stack:
                self.all_rays = torch.stack(self.all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames])*h*w, 3)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames])*h*w, 3) 
            else:
                self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        """
        print()
        print(self.all_rays.size())
        print(self.all_rgbs.size())
        print(self.all_rgbs)
        #exit()
        """
        torch.Size([50331648, 6])
        torch.Size([50331648, 3])
        tensor([[0.3412, 0.3059, 0.2392],
        [0.4000, 0.3647, 0.2980],
        [0.3686, 0.3333, 0.2745],
        ...,
        [0.7098, 0.5961, 0.4314],
        [0.7176, 0.6039, 0.4392],
        [0.7490, 0.6353, 0.4706]])
        """
        #exactly the number of rays in the expected 16 images @ 2048*1536 pixel resolution. also matching rgb.
    
    def bbox2corners(self):
        corners = self.scene_bbox.unsqueeze(0).repeat(4,1,1)
        for i in range(3):
            corners[i,[0,1],i] = corners[i,[1,0],i] 
        return corners.view(-1,3)
        
    def distort5(self,x, y, cx, cy, fx, fy, k1, k2, p1, p2, k3):
        #this should distort the ray to correspond to the true direction of the pixel indicatead in the image
        #way easier than undistorting the image. huehue.
        x0 = (x-cx)/fx #undistorted coords
        y0 = (y-cy)/fy
        r2 = x0*x0 + y0*y0
        r4 = r2*r2
        r6 = r4*r2
        k =  1 + k1*r2 + k2*r4 + k3*r6
        
        dx = 2*p1*x0*y0 + p2*(r2 + 2*x0*x0)
        dy = p1*(r2 + 2*y0*y0) + 2*p2*x0*y0
        x = x0*k+dx
        y = y0*k+dy
        return np.array((x * fx + cx, y * fy + cy))
        
    def read_meta(self):
        with open(os.path.join(self.root_dir, "intrinsics.txt")) as f:
            focal = float(f.readline().split()[0])
        self.intrinsics = np.array([[focal,0,400.0],[0,focal,400.0],[0,0,1]])
        self.intrinsics[:2] *= (np.array(self.img_wh)/np.array([800,800])).reshape(2,1)

        pose_files = sorted(os.listdir(os.path.join(self.root_dir, 'pose')))
        img_files  = sorted(os.listdir(os.path.join(self.root_dir, 'rgb')))

        if self.split == 'train':
            pose_files = [x for x in pose_files if x.startswith('0_')]
            img_files = [x for x in img_files if x.startswith('0_')]
        elif self.split == 'val':
            pose_files = [x for x in pose_files if x.startswith('1_')]
            img_files = [x for x in img_files if x.startswith('1_')]
        elif self.split == 'test':
            test_pose_files = [x for x in pose_files if x.startswith('2_')]
            test_img_files = [x for x in img_files if x.startswith('2_')]
            if len(test_pose_files) == 0:
                test_pose_files = [x for x in pose_files if x.startswith('1_')]
                test_img_files = [x for x in img_files if x.startswith('1_')]
            pose_files = test_pose_files
            img_files = test_img_files

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.intrinsics[0,0],self.intrinsics[1,1]], center=self.intrinsics[:2,2])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)


        self.render_path = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []

        assert len(img_files) == len(pose_files)
        for img_fname, pose_fname in tqdm(zip(img_files, pose_files), desc=f'Loading data {self.split} ({len(img_files)})'):
            image_path = os.path.join(self.root_dir, 'rgb', img_fname)
            img = Image.open(image_path)
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(img.shape[0], -1).permute(1, 0)  # (h*w, 4) RGBA
            if img.shape[-1]==4:
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            c2w = np.loadtxt(os.path.join(self.root_dir, 'pose', pose_fname)) #@ self.blender2opencv
            c2w = torch.FloatTensor(c2w)
            self.poses.append(c2w)  # C2W
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 8)
            
#             w2c = torch.inverse(c2w)
#

        self.poses = torch.stack(self.poses)
        #if 'train' == self.split:
        #    if self.is_stack:
        #        self.all_rays = torch.stack(self.all_rays, 0).reshape(-1,*self.img_wh[::-1], 6)  # (len(self.meta['frames])*h*w, 3)
        #        self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames])*h*w, 3) 
        #    else:
        #        self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
        #        self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        #else:
        #    self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
        #    self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

 
    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = torch.from_numpy(self.intrinsics[:3,:3]).unsqueeze(0).float() @ torch.inverse(self.poses)[:,:3] #anyone even using this?

    def world2ndc(self, points):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]

            sample = {'rays': rays,
                      'rgbs': img}
        return sample
