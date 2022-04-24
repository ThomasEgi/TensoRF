from direct.showbase.ShowBase import ShowBase



class MyApp(ShowBase):


    def __init__(self):

        ShowBase.__init__(self)

        # Load the environment model.
        self.cindex = 0 
        self.scene = self.loader.loadModel("testfile_train.bam")
        # Reparent the model to render.

        self.scene.reparentTo(self.render)
        self.scene.setTransparency(True)
        # Apply scale and position transforms on the model.
        print(self.scene.getChildren())
        #self.scene.setScale(0.25, 0.25, 0.25)

        #self.scene.setPos(-8, 42, 0)
        self.npchildren = self.scene.getChildren().getPaths()
        for x in self.npchildren:
            axis = loader.loadModel("zup-axis")
            #axis.reparentTo(x)
            axis.setScale(0.05)
            axis.setColorScale(1,0,1,.1)
        #axis = loader.loadModel("zup-axis")
        #axis.reparentTo(self.scene)
        
        
        axis = loader.loadModel("./data/flower1/MeshroomCache/Texturing/c02a5a12a18876e8f3014aed88f07aa19a8584ac/texturedMesh.obj")
        axis.reparentTo(self.scene)
        base.accept("i",self.nextview)
        base.camLens.setNear(.1)
        self.disableMouse()
        
        
    def nextview(self):
        print("next")
        try:
            
            nextchild = self.npchildren[self.cindex]
            print("loaded next child",self.cindex)
            self.cindex +=2
            
        except:
            nextchild = self.npchildren[0]
            self.cindex =0
        base.camera.reparentTo(nextchild)
        base.camera.setPos(0,0,0)
        base.camera.setHpr(0,90,0)
        
    
    
app = MyApp()

app.run()
