import os
import sys
import random

import nibabel as nib 
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset 

def LoadOnePerson(data_root):
    folders = os.listdir(data_root)
    data = {}
    for folder in folders:
        data_type = folder.split('.')[4]
        nii_file = os.path.join(data_root, folder, folder+'.nii')
        nii_data = nib.load(nii_file)
        data[data_type] = nii_data.get_data().squeeze()
    return data

def StackData(person_data):
    channels = []
    label = None
    #print(sorted(person_data.keys()))
    for img_type, img_data in sorted(person_data.items()):
        if img_type == 'OT':
            label = img_data
        else:
            if len(img_data.shape) == 3:
                h,w,d = img_data.shape
                img_data.shape = [h,w,d,1]
            else:
                img_data = np.mean(img_data.astype(np.float32), axis=3)
                h,w,d = img_data.shape
                img_data.shape = [h,w,d,1]
            #print(img_data.shape)
            channels.append(img_data.astype(np.float32))
    channels = np.concatenate(channels, axis=3)
    return channels, label

def Normalize(data_list, means, norm):
    ndata_list = []
    if means is not None and norm is not None:
        for data in data_list:
            ndata_list.append((data-means)/norm)
    else:
        h,w,d,c = data_list[0].shape
        means = np.zeros((1,1,1,c), np.float32)
        norm = np.zeros((1,1,1,c), np.float32)
        count = 0
        for data in data_list:
            means += np.sum(data, axis=(0,1,2), keepdims=True)
            count += data.shape[0]*data.shape[1]*data.shape[2]
        means /= count
        for data in data_list:
            ndata_list.append(data-means)
        for data in ndata_list:
            norm += np.sum(np.sqrt(data*data), axis=(0,1,2), keepdims=True)
        norm /= count
        for data in ndata_list:
            data /= norm
    return ndata_list, means, norm 

def SampleVolume(data, label, dst_shape=[96, 96, 5]):
    src_h,src_w,src_d,_ = data.shape
    dst_h,dst_w,dst_d = dst_shape
    h = random.randint(0, src_h-dst_h)
    w = random.randint(0, src_w-dst_w)
    d = random.randint(0, src_d-dst_d)
    sub_volume = data[h:h+dst_h,w:w+dst_w,d:d+dst_d,:]
    sub_label = label[h:h+dst_h,w:w+dst_w,d:d+dst_d]
    if random.random() > 0.5:
        sub_volume = sub_volume[:, ::-1, :, :]
        sub_label = sub_label[:, ::-1, :]
    return sub_volume,sub_label

def MakeGrid(imgs, width=8):
    h, w, c = imgs[0].shape
    height = int(len(imgs)/width) + (1 if len(imgs)%width > 0 else 0)
    ind = 0
    concat_img = np.zeros((h*height, w*width, c), np.uint8)
    for h_idx in range(height):
        for w_idx in range(width):
            if ind >= len(imgs):
                continue
            concat_img[h_idx*h:(h_idx+1)*h, w_idx*w:(w_idx+1)*w, :] =  imgs[ind]
            ind += 1
    return concat_img

def Visualize(person_data):
    normlize_data = {}
    for img_type, img_data in person_data.items():
        data = img_data.astype(np.float32)
        data *= 255.0/img_data.max()
        normlize_data[img_type] = data.astype(np.uint8)
    for i in range(19):
        groundtruth = 'OT' in normlize_data.keys()
        if groundtruth:
            gt = (normlize_data['OT'][:,:, i]>128).astype(np.uint8)
            contours,_ = cv2.findContours(gt.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgs = {}
        for img_type, img_data in normlize_data.items():
            if len(img_data.shape) == 3:
                imgs[img_type] = img_data[:, :, i]
            else:
                for j in range(img_data.shape[3]):
                    imgs[img_type+('_%d'%j)] = img_data[:, :, i, j]
        for img_type, img_data in imgs.items():
            img_data = cv2.merge([img_data, img_data, img_data])
            if groundtruth and len(contours)>0:
                img_data = cv2.drawContours(img_data, contours, -1, (255,0,0), 1)
            cv2.putText(img_data, img_type, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            imgs[img_type] = img_data
        print(len(imgs))
        concat_img = MakeGrid(imgs.values(), 8)
        cv2.imshow('img', concat_img)
        cv2.waitKey()

class ISLESDataset(Dataset):
    def __init__(self, folders, sample_shape=(96,96,5), means=None, norm=None, is_train=False):
        data_list = [LoadOnePerson(folder) for folder in folders]
        data_list = [StackData(data) for data in data_list]
        label_list = [data[1] for data in data_list]
        data_list = [data[0] for data in data_list]
        data_list,means,norm = Normalize(data_list, means, norm)
        self.folders = folders
        self.label_list = label_list
        self.data_list = data_list
        self.sample_shape = sample_shape
        self.means = means
        self.norm = norm
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        volume = self.data_list[index]
        label = self.label_list[index]
        if self.is_train:
            assert(label is not None)
            volume,label = SampleVolume(volume, label, self.sample_shape)
        volume = torch.Tensor(volume.copy()).permute(3,2,0,1) # H,W,D,C -> C,D,H,W
        if label is not None:
            label = torch.Tensor(label.copy()).permute(2,0,1) # H,W,D -> D,H,W
        return volume, label

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

def test_visulize():
    root = sys.argv[1]
    data = LoadOnePerson(root)
    for img_type, img_data in data.items():
        print(img_type, img_data.shape)
    Visualize(data)

def test_dataset():
    data_root = sys.argv[1]
    folders = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root))] 
    dataset = ISLESDataset(folders)
    print('number of items: %d' % len(dataset))
    for i in range(len(dataset)):
        volume,label = dataset[i]
        print(volume.shape, label.shape)

if __name__ == '__main__':
    test_visulize()
    #test_dataset()

