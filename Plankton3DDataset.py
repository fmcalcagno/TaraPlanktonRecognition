import torch
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile

class plankton3DDataset(Dataset):
    """Dataset Class for loading Gif into a 3D Tensor"""
    def __init__(self,gifList,rootDir, channels,  timeDepth, xSize, ySize, 
                 startFrame,endFrame,numFilters,filters,keys,classes_transf,transfpos,transform=None):
        """
        Args:
        clipsList (string): Path to the clipsList file with labels.
        rootDir (string): Directory with all the 3D Images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        channels: Number of channels of frames
        timeDepth: Number of frames to be loaded in a sample to construct the 3D Matrix
        xSize, ySize: Dimensions of the frames
        startFrame,endFrame: first and last frame from the original gif
        filters: numbers of filters that you would input
        classes_trans: Transformation from folder name to class name
        transfpos: hierarchy to apply as output (0 is 'living' and 'not living')
        transform: transformation to apply to the frames
        """
        self.gifList = gifList
        self.rootDir = rootDir
        self.channels = channels
        self.timeDepth = timeDepth
        self.xSize = xSize
        self.ySize = ySize
        #self.mean = mean
        self.transform = transform
        self.startFrame=startFrame
        self.endFrame=endFrame
        self.numFilters=numFilters
        self.keys=keys
        #self.keys= pd.read_csv("../plancton_species.csv",delimiter=";",header=0,names =("0"))
        self.classes =[d for d in self.keys]
        self.classes.sort()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.filters=filters
        self.classes_transf=classes_transf
        self.transfpos=transfpos
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return len(self.gifList)
    
    def crop6(self,im):
        number_of_cols=3
        W=im.width
        H=im.height
        w=(W-16)/3
        h=(H-24)/2
        images=[]
        w1=0
        w2=w
        for i in range(number_of_cols):
            im1=im.crop((w1, 0, w2, h))
            images.append(im1)
            im1=im.crop((w1, h+8, w2, 2*h+8))
            images.append(im1)
            w1=w2+8
            w2=w2+w+8
        return images
    
    def readGif(self, gifFile):
        # Open the gif file, crop it and return the frames in a tensor
        image_gif=Image.open(gifFile, mode='r')
        width, height = image_gif.size
        w=(width-16)/3
        h=(height-24)/2
        #frames = torch.FloatTensor(self.numFilters,self.timeDepth,  
        #                            self.xSize, self.ySize)
        
        frames=np.zeros((self.numFilters,self.timeDepth,int(h), int(w)))
        nframes = 0
        nframesin=0

        while image_gif:
            if self.startFrame<=nframes<=self.endFrame:
                six_images=self.crop6(image_gif)
                for ifilter in range(self.numFilters):
                    realfilter=self.filters[ifilter]
                    if self.channels == 3: pil_image = six_images[realfilter].convert("RGB")
                    if self.channels == 1: pil_image = six_images[realfilter].convert("L") 
                    #imResize=pil_image.resize((self.xSize, self.ySize),Image.NEAREST)
                    #frame = torch.from_numpy(np.asarray(imResize).astype(np.float32))    
                    frame = np.asarray(pil_image).astype(np.float32)   
                    #I have to substract the mean and divide by the stddev for each filter
                    #if self.channels == 1: frame=torch.unsqueeze(frame,2)
                    #frames[ifilter,nframes, :, :] = (frame - self.colmean[ifilter])/self.colstddev[ifilter]
                    frames[ifilter,nframes, :, :] = frame
                nframesin+=1
            nframes += 1
            try:
                image_gif.seek( nframes )
            except EOFError:
                break;
            
        image_gif.close()
        return frames
    

    def __getitem__(self, idx):
        file,folder,typeplan=self.gifList.iloc[idx]
        typeplan=self.classes_transf[folder][self.transfpos]
        gifFile= os.path.join(self.rootDir, os.path.join(folder,file))
        clip = self.readGif(gifFile)
        if self.transform:
            clip = self.transform(clip)
        return clip,self.class_to_idx[typeplan]
