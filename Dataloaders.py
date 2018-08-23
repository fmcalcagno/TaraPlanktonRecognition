import torch
import torch.utils.data as data
from Transformations3D import Rescale,RandomCrop,Normalize,RandomRotate,RandomFlip,ToTensor
from Plankton3DDataset import plankton3DDataset
import UtilsPL
import torchvision.transforms as transforms

def get_data_loaders(trainlist,testlist,hierclasses,classes_transf,pwargs,kwargs,train_batch_size, val_batch_size):

    colmean=[252.9242,251.4343,156.3816]
    colstddev=[7.7102,11.3070,66.3204]
    
    colmeanval=[252.9046,251.4320,156.2538]
    colstddevval=[7.9116,11.3310,66.1270]
    

    compose=transforms.Compose([Rescale(150),
                                RandomCrop(112),
                                RandomRotate(),
                                RandomFlip(),
                                Normalize(colmean,colstddev),
                                ToTensor()])


    composetest=transforms.Compose([Rescale(112),
                                Normalize(colmeanval,colstddevval),
                                ToTensor()])

    train_loader = data.DataLoader(plankton3DDataset(gifList=trainlist,
                        keys=hierclasses,classes_transf=classes_transf,transform=compose,**pwargs), 
                batch_size=train_batch_size, shuffle=True, collate_fn=UtilsPL.ucf_collate, **kwargs)

    val_loader = data.DataLoader(plankton3DDataset(gifList=testlist,
                        keys=hierclasses,classes_transf=classes_transf,transform=compose,**pwargs),
                batch_size=val_batch_size, shuffle=True, collate_fn=UtilsPL.ucf_collate, **kwargs)
    
    return train_loader, val_loader


