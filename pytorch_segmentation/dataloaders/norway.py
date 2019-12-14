from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def rotate_sampels(data,labels,ang,minsize=255):
    assert (ang > 0) and (ang < 90),'rotate_sampels angle should be in range 0-90,given:'+str(ang)
    
    rad = ang*np.pi/180.
    sampls = []
    samplsl= []
    c = data.shape[:-1][::-1]
    size = int(c[1]*np.sin(rad)+c[0]*np.cos(rad)),\
           int(c[0]*np.sin(rad)+c[1]*np.cos(rad))
    
    rot = cv2.getRotationMatrix2D((c[0]/2,c[1]/2),ang,1)
    rot[0,2]+=size[0]/2-c[0]/2
    rot[1,2]+=size[1]/2-c[1]/2
    
    res = np.concatenate([cv2.warpAffine(data[:,:,i],rot,size)[:,:,None] for i in range(data.shape[-1])],axis=-1)
    resl =np.concatenate([cv2.warpAffine(labels[:,:,i],rot,size,flags=cv2.INTER_NEAREST)[:,:,None] for i in range(labels.shape[-1])],axis=-1)
    
    i = np.arange(res.shape[0],dtype=np.int32)
    cl = c[0]*np.sin(rad)
    lo = np.maximum((cl-i)/np.tan(rad),((i-cl)*np.tan(rad))).astype(np.int32)
    lo = np.clip(lo,0,res.shape[1]-1)
    ch = c[1]*np.cos(rad)
    hi = (size[0]-np.maximum((ch-i)*np.tan(rad),((i-ch)/np.tan(rad)))).astype(np.int32)
    hi = np.clip(hi,0,res.shape[1]-1)
    ranges = np.concatenate([i[:,None],lo[:,None],hi[:,None]],-1)
    ranges = ranges[hi-lo > minsize]    
    for ii,l,h in ranges:
        sampls.append(res[ii,l:h,:])
        samplsl.append(resl[ii,l:h,:])
    return sampls,samplsl

def rotate90(data,labels):
    res = np.concatenate([cv2.rotate(data[:,:,i],cv2.ROTATE_90_CLOCKWISE)[:,:,None] for i in range(data.shape[-1])],axis=-1)
    resl=np.concatenate([cv2.rotate(labels[:,:,i],cv2.ROTATE_90_CLOCKWISE)[:,:,None] for i in range(labels.shape[-1])],axis=-1)
    return res,resl
    
def augment_rotation(data,labels,angls=[0]):
    res,resl =[],[]
    for ang in angls:
        if (ang == 0):
            r,rl = list(data),list(labels)
        else:
            r,rl = rotate_sampels(data,labels,ang)
        res+=r
        resl+=rl
    
    data,labels = rotate90(data,labels)
    for ang in angls:
        if (ang == 0):
            r,rl = list(data),list(labels)
        else:
            r,rl = rotate_sampels(data,labels,ang)
        res+=r
        resl+=rl
    return res,resl        
            
    res,resl = rotate_sampels(data,labels,10)
    
class NorwayDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 6
        self.palette = [0,0,255,255,0,0,0,255,0,255,255,255,255,0,255,0,255,255]
        super(NorwayDataset, self).__init__(**kwargs)

    def _set_files(self):
        if 'test' in self.split:
            self.root = os.path.join(self.root,'test_once','test1')
        elif 'val' in self.split:
            self.root = os.path.join(self.root,'test_once','test2')
        elif 'train' in self.split:
            self.root = os.path.join(self.root,'train','train')
        else:
            assert False, 'unknown split type,'+self.split
        self.files = np.load(self.root+'_seismic.npy')
        self.labels = np.load(self.root+'_labels.npy')
    
    def _load_data(self, index):
        label = self.labels[index]
        data = self.files[index]
        data = data[:,:,None].astype(np.float32)
        return data, label, str(index)

class NorwayAugDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 6
        self.palette = [0,0,255,255,0,0,0,255,0,255,255,255,255,0,255,0,255,255]
        super(NorwayAugDataset, self).__init__(**kwargs)

    def _set_files(self):
        if 'test' in self.split:
            self.root = os.path.join(self.root,'test_once','test1')
        elif 'val' in self.split:
            self.root = os.path.join(self.root,'test_once','test2')            
        elif 'train' in self.split:
            self.root = os.path.join(self.root,'train','train')
        else:
            assert False, 'unknown split type,'+self.split
        self.files = np.load(self.root+'_seismic.npy')
        self.labels = np.load(self.root+'_labels.npy')
        self.files, self.labels = augment_rotation(self.files,self.labels)
        self.files, self.labels = self.files, self.labels
    
    def _load_data(self, index):
        label = self.labels[index]
        data = self.files[index]
        data = data[:,:,None].astype(np.float32)
        return data, label, str(index)


class Norway(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False,elastic=False):
        
        self.MEAN = [0]
        self.STD = [1.]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val,
            'elastic': elastic
        }
    
        if split in ["train_aug", "trainval_aug", "val_aug", "test_aug"]:
            self.dataset = NorwayAugDataset(**kwargs)
        elif split in ["train", "trainval", "val", "test"]:
            self.dataset = NorwayDataset(**kwargs)
        else: 
            raise ValueError(f"Invalid split name {split}")
        super(Norway, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
