import torch
from skimage import transform
import random
import numpy as np

class Rescale(object):
    """Rescale the 3 3D images in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size of widht ad height (Lenght stays the same) 
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
            
    def __call__(self, sample):
        image=sample
    
        c,f,h, w = image.shape
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
            
        
        new_h, new_w = int(new_h), int(new_w)
        #Generate a resize
        frames=np.zeros((c,f,new_h, new_w))
        for c_i in range(c):
            for f_i in range(f):
                new_image=transform.resize(image[c_i,f_i,:,:],(new_h, new_w))
                frames [c_i,f_i,:,:]=new_image
        
        return frames

class RandomCrop(object):
    """Crop randomly the 3D image in a sample, using the same crop for all filters and frames
    
    Args:
        output_size (tuple or int): Desired output size of each frame of each filter. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    
    def __call__(self, sample):
        image= sample
        
        c,f,h, w = image.shape
        new_h, new_w = self.output_size      
        top = np.random.randint(0, abs(h - new_h))
        left = np.random.randint(0, abs(w - new_w))
        frames = image[:,:,top: top + new_h,left: left + new_w]
        return frames
    
class Normalize(object):
    """Normalize the 3 filters 
    
    Args:
        output_size (tuple or int): Desired output size of each frame of each filter. If int, square crop
            is made.
    """
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev= stddev
    
    def __call__(self, sample):
        image= sample
        
        c,f,h, w = image.shape
        for c_i in range(c):
             image[c_i,:,:,:]=(image[c_i,:,:,:] - self.mean[c_i])/self.stddev[c_i]
        return image
    
class RandomRotate(object):
    """ Rotate 90/180/270 degrees all filters"""
    def __init__(self):
        
        self.degrees=list([0,90,180,270])
        
    
    def __call__ (self,sample):
        c,f,h, w = sample.shape
        choice=random.choice(self.degrees)
        if choice==90:
            frames=np.zeros((c,f,w, h))
            for c_i in range(c):
                for f_i in range(f):
                    frames[c_i,f_i,:,:]= np.rot90(sample[c_i,f_i,:,:])
            return frames       
            #rotate 90 degrees
        elif choice==180:
            #rotate 180 degrees
            frames=np.zeros((c,f,h, w))
            for c_i in range(c):
                for f_i in range(f):
                    frames[c_i,f_i,:,:]= np.rot90(sample[c_i,f_i,:,:],2)
            return frames
        elif choice==270:
            frames=np.zeros((c,f,w, h))
            for c_i in range(c):
                for f_i in range(f):
                    frames[c_i,f_i,:,:]= np.rot90(sample[c_i,f_i,:,:],-1)
            return frames
        else:
            return sample
            #rotate 270 degrees
            
class RandomFlip(object):        
    def __init__(self):
        self.direction=list(["Nothing","LRDirection","UDDirection"])
        
    def __call__ (self,sample):
        c,f,h, w = sample.shape
        choice=random.choice(self.direction)
        if choice == "Nothing":
            #Do nothing
            return sample
        elif choice=="LRDirection":
            frames=np.zeros((c,f,h, w))
            for c_i in range(c):
                for f_i in range(f):
                    frames[c_i,f_i,:,:]= np.fliplr(sample[c_i,f_i,:,:])
            return frames
        elif choice=="UDDirection":
            frames=np.zeros((c,f,h,w))
            for c_i in range(c):
                for f_i in range(f):
                    frames[c_i,f_i,:,:]= np.flipud(sample[c_i,f_i,:,:])
            return frames
            
        
                
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):

        return torch.from_numpy(np.asarray(sample).astype(np.float32))  
    
