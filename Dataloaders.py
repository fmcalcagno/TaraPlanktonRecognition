import torch
import torch.utils.data as data
from Transformations3D import Rescale,RandomCrop,Normalize,RandomRotate,RandomFlip,RandomBrightner,ToTensor
from Plankton3DDataset import plankton3DDataset
import UtilsPL
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from  torch.utils.data.sampler import  WeightedRandomSampler

def getweights(trainlist, classes_transf,transfpos):
    vec=[]
    for row in trainlist.values:
        filex,folder,typeplan=row
        hier1,hier2,hier3=classes_transf[folder]
        vec.append((filex,hier1,hier2,hier3))
    total=len(vec)
    hierarchies=pd.DataFrame.from_records(vec,columns=("filex","hier1","hier2","hier3"))

    if transfpos==0:
        return (hierarchies.sort_values('hier1').groupby('hier1').agg(['count']).iloc[:,1]/total).as_matrix()
    elif transfpos==1:
        return (hierarchies.sort_values('hier2').groupby('hier2').agg(['count']).iloc[:,1]/total).as_matrix()
    else:
        return (hierarchies.sort_values('hier3').groupby('hier3').agg(['count']).iloc[:,1]/total).as_matrix()


def get_data_loaders(trainlist,testlist,hierclasses,classes_transf,transfpos,pwargs,kwargs,train_batch_size, val_batch_size):
    #trainlist=trainlist.sample(1000, replace=False)
    #testlist=testlist.sample(100, replace=False)
    
    colmean=[250.8401,251.1146,252.9242,251.4343,156.3816]
    colstddev=[7.7102,11.3070,66.3204]
    
   
    colmeanval=[250.9021,251.1151,252.9046,251.4320,156.2538]
    colstddevval=[15.8342,13.6886,16.1009,13.6824,7.9116,11.3310,66.1270]
    

    compose=transforms.Compose([Rescale(150),
                                RandomCrop(112),
                                RandomRotate(),
                                RandomFlip(),
				RandomBrightner(1),
                                Normalize(colmean,colstddev),
                                ToTensor()])

    composeval=transforms.Compose([Rescale(112),
                                Normalize(colmeanval,colstddevval),
                                ToTensor()])


    train_loader = data.DataLoader(plankton3DDataset(gifList=trainlist,
                        keys=hierclasses,classes_transf=classes_transf,transform=compose,transfpos=transfpos,**pwargs), 
                batch_size=train_batch_size,  
		shuffle=True,
                collate_fn=UtilsPL.ucf_collate, **kwargs)


    val_loader = data.DataLoader(plankton3DDataset(gifList=testlist,
                        keys=hierclasses,classes_transf=classes_transf,transform=composeval,transfpos=transfpos,**pwargs),
                batch_size=val_batch_size, shuffle=True, collate_fn=UtilsPL.ucf_collate, **kwargs)
    
    return train_loader, val_loader


