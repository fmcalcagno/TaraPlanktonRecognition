import torch
import torch.utils.data as data
from Transformations3D import Rescale,RandomCrop,Normalize,RandomRotate,RandomFlip,RandomBrightner,ToTensor
from Plankton3DDataset import plankton3DDataset
import UtilsPL
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from  torch.utils.data.sampler import  WeightedRandomSampler


def get_data_loaders(trainlist,testlist,hierclasses,classes_transf,transfpos,pwargs,kwargs,train_batch_size, val_batch_size):
    #trainlist=trainlist.sample(1000, replace=False)
    #testlist=testlist.sample(100, replace=False)
    
    colmean=[250.8401,251.1146,252.9242,251.4343,156.3816]
    colstddev=[16.1009,13.6824, 7.7102,11.3070,66.3204]
    
   
    colmeanval=[250.9021,251.1151,252.9046,251.4320,156.2538]
    colstddevval=[15.8342,13.6886,7.9116,11.3310,66.1270]
    

    compose=transforms.Compose([Rescale(200),
                                #RandomCrop(112),
                                RandomRotate(),
                                RandomFlip(),
				RandomBrightner(1),
                                Normalize(colmean,colstddev),
                                ToTensor()])

    composeval=transforms.Compose([Rescale(200),
                                Normalize(colmeanval,colstddevval),
                                ToTensor()])


    
    
    train_loader = data.DataLoader(plankton3DDataset(gifList=trainlist,
                        keys=hierclasses,classes_transf=classes_transf,transform=compose,transfpos=transfpos,**pwargs), 
                batch_size=train_batch_size,  
		shuffle=True,
                collate_fn=UtilsPL.ucf_collate, **kwargs)


    val_loader = data.DataLoader(plankton3DDataset(gifList=testlist,
                        keys=hierclasses,classes_transf=classes_transf,transform=composeval,transfpos=transfpos,**pwargs),
                batch_size=val_batch_size,  collate_fn=UtilsPL.ucf_collate, **kwargs)
    
    return train_loader, val_loader


