import os
import sys
import random

import nibabel as nib
import SimpleITK as sitk
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset 

from preprocess import ReColor,RandomRotate,SampleVolume,CurriculumWrapper

def LoadOnePersonNii(data_root):
    folders = os.listdir(data_root)
    data = {}
    for folder in folders:
        data_type = folder.split('.')[4]
        nii_file = os.path.join(data_root, folder, folder+'.nii')
        nii_data = nib.load(nii_file)
        data[data_type] = nii_data.get_data().squeeze()
    return data

def LoadOnePersonMha(data_root):
    folders = os.listdir(data_root)
    data = {}
    for folder in folders:
        data_type = folder.split('.')[4]
        mha_file = os.path.join(data_root, folder, folder+'.mha')
        mha_data = sitk.ReadImage(mha_file)
        data[data_type] = sitk.GetArrayFromImage(mha_data).transpose([1,2,0])
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
                channels.append(img_data.astype(np.float32))
            else:
                h,w,d,_ = img_data.shape
                avg_img_data = np.mean(img_data.astype(np.float32), axis=3)
                avg_img_data.shape = [h,w,d,1]
                channels.append(avg_img_data.astype(np.float32))
                max_img_data = np.max(img_data.astype(np.float32), axis=3)
                max_img_data.shape = [h,w,d,1]
                channels.append(max_img_data.astype(np.float32))
                min_img_data = np.min(img_data.astype(np.float32), axis=3)
                min_img_data.shape = [h,w,d,1]
                channels.append(min_img_data.astype(np.float32))
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
        print(means, norm)
    return ndata_list, means, norm 

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
    color_bar = [
            (0, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 0)
            ]
    # normalize each channel in range [0, 255]
    normalize_data = {}
    for img_type, img_data in person_data.items():
        if img_type == 'OT':
            print(np.max(img_data), np.min(img_data))
            max_label = np.max(img_data)
            data = img_data
        else:
            data = img_data.astype(np.float32)
            data *= 255.0/img_data.max()
        normalize_data[img_type] = data.astype(np.uint8)
    # for each time slice
    for i in range(normalize_data.values()[0].shape[2]):
        groundtruth = 'OT' in normalize_data.keys()
        # parse groundtruth
        if groundtruth:
            ot_data = normalize_data['OT'][:,:, i] 
            gt = (ot_data>0).astype(np.uint8) # TODO assume 0 is background class
            contours,_ = cv2.findContours(gt.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            R = np.zeros(ot_data.shape, np.uint8)
            G = np.zeros(ot_data.shape, np.uint8)
            B = np.zeros(ot_data.shape, np.uint8)
            print(np.unique(ot_data))
            for label in range(1, max_label+1):
                R[ot_data==label] = color_bar[label][0]
                G[ot_data==label] = color_bar[label][1]
                B[ot_data==label] = color_bar[label][2]
            ot_data = cv2.merge([B, G, R])
            cv2.putText(ot_data, 'OT', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        imgs = {}
        for img_type, img_data in normalize_data.items():
            if len(img_data.shape) == 3:
                imgs[img_type] = img_data[:, :, i]
            else:
                continue #TODO
                for j in range(img_data.shape[3]):
                    imgs[img_type+('_%d'%j)] = img_data[:, :, i, j]
        for img_type, img_data in imgs.items():
            img_data = cv2.merge([img_data, img_data, img_data])
            if groundtruth and len(contours)>0:
                cv2.drawContours(img_data, contours, -1, (255,0,0), 1)
            cv2.putText(img_data, img_type, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            imgs[img_type] = img_data
        if groundtruth and len(contours)>0:
            imgs['OT'] = ot_data
        print(len(imgs))
        concat_img = MakeGrid(imgs.values(), 8)
        cv2.imshow('img', concat_img)
        cv2.waitKey()

class ScanDataset(Dataset):
    def __init__(self, folders, sample_shape=(96,96,5), means=None, norm=None, is_train=False):
        data_list, label_list = self.load_data(folders)
        data_list,means,norm = Normalize(data_list, means, norm)
        self.folders = folders
        self.label_list = label_list
        self.data_list = data_list
        self.sample_shape = sample_shape
        self.means = means
        self.norm = norm
        self.is_train = is_train
        self.set_trans_prob(1.0)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        volume = self.data_list[index]
        label = self.label_list[index]
        if self.is_train:
            assert(label is not None)
            for trans in self.trans_all:
                volume, label = trans(volume, label)
            for trans in self.trans_data:
                volume = trans(volume)
        volume = torch.Tensor(volume.copy()).permute(3,2,0,1) # H,W,D,C -> C,D,H,W
        if label is not None:
            label = torch.Tensor(label.copy()).permute(2,0,1) # H,W,D -> D,H,W
        return volume, label

    def set_trans_prob(self, prob):
        self.trans_prob = prob
        self.trans_data = [ CurriculumWrapper(ReColor(alpha=0.05), prob) ]
        self.trans_all = [ SampleVolume(dst_shape=self.sample_shape),
                CurriculumWrapper(RandomRotate(random_flip=True), prob)]

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

class ISLESDataset(ScanDataset):
    def __init__(self, folders, sample_shape=(96,96,5), means=None, norm=None, is_train=False):
        self.name = 'ISLES'
        super(ISLESDataset, self).__init__(folders, sample_shape, means, norm, is_train)

    def load_data(self, folders):
        data_list = [LoadOnePersonNii(folder) for folder in folders]
        data_list = [StackData(data) for data in data_list]
        label_list = [data[1] for data in data_list]
        data_list = [data[0] for data in data_list]
        return data_list, label_list

class BRATSDataset(ScanDataset):
    def __init__(self, folders, sample_shape=(96,96,5), means=None, norm=None, is_train=False):
        self.name = 'BRATS'
        means = np.array([[[[ 51.95236969,  74.40973663,  81.23361206,  95.90114594]]]], dtype=np.float32)
        norm = np.array([[[[ 89.12859344,  124.9729538 ,  137.86834717,  154.61538696]]]], dtype=np.float32)
        super(BRATSDataset, self).__init__(folders, sample_shape, means, norm, is_train)

    def load_data(self, folders):
        data_list = []
        label_list = []
        for folder in folders:
            print('loading %s' % folder)
            cache_file = os.path.join('./cache', '_'.join(folder.split('/')[-2:]))
            if os.path.exists(cache_file+'.npz'):
                print('load from cache %s' % cache_file)
                npzfile = np.load(cache_file+'.npz')
                data = npzfile['data']
                label = npzfile['label'] if 'label' in npzfile else None
            else:
                data = LoadOnePersonMha(folder)
                data, label = StackData(data)
                if label is None:
                    np.savez(cache_file, data=data)
                else:
                    np.savez(cache_file, data=data, label=label)
            data_list.append(data)
            label_list.append(label)
        return data_list, label_list

def test_visulize():
    root = sys.argv[1]
    data = LoadOnePersonMha(root)
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

