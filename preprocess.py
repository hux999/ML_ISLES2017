import random
import math

import numpy as np 
import cv2

class CurriculumWrapper:
    def __init__(self, trans, prob):
        self.trans = trans
        self.prob = prob
    
    def __call__(self, *args):
        if random.random() < self.prob:
            return self.trans(*args)
        else:
            if len(args) == 1:
                args = args[0]
            return args

class ReColor:
    def __init__(self, alpha=0.05):
        self._alpha = alpha

    def __call__(self, im):
        num_chns = im.shape[3]
        # random amplify each channel
        t = np.random.uniform(-1, 1, num_chns)
        im = im.astype(np.float32)
        im *= (1 + t * self._alpha)
        return im

class SampleVolume:
    def __init__(self, dst_shape=[96, 96, 5], pos_ratio=-1):
        self.dst_shape = dst_shape
        self.pos_ratio = pos_ratio

    def __call__(self, data, label):
        src_h,src_w,src_d,_ = data.shape
        dst_h,dst_w,dst_d = self.dst_shape
        if type(dst_d) is list:
            dst_d = random.choice(dst_d)
        if self.pos_ratio<0:
            h = random.randint(0, src_h-dst_h)
            w = random.randint(0, src_w-dst_w)
            d = random.randint(0, src_d-dst_d)
        else:
            select = label>0 if random.random() < self.pos_ratio else label==0
            h, w, d = np.where(select)
            select_idx = random.randint(0, len(h)-1)
            h = h[select_idx] - int(dst_h/2)
            w = w[select_idx] - int(dst_w/2)
            d = d[select_idx] - int(dst_d/2)
            h = min(max(h,0), src_h-dst_h)
            w = min(max(w,0), src_w-dst_w)
            d = min(max(d,0), src_d-dst_d)
        sub_volume = data[h:h+dst_h,w:w+dst_w,d:d+dst_d,:]
        sub_label = label[h:h+dst_h,w:w+dst_w,d:d+dst_d]
        return sub_volume,sub_label

class ScaleAndPad:
    def __init__(self, dst_size, rand_pad=False):
        self.dst_size = dst_size
        self.rand_pad = rand_pad

    def __call__(self, im, mask):
        org_h = im.shape[0]
        org_w = im.shape[1]
        fx = math.floor(self.dst_size/org_w)
        fy = math.floor(self.dst_size/org_h)
        new_im = np.zeros((self.dst_size, self.dst_size, 3), im.dtype)
        org_h = int(org_h*fy)
        org_w = int(org_w*fx)
        offset_x = 0 if self.rand_pad is False else random.randint(0, self.dst_size-org_w)
        offset_y = 0 if self.rand_pad is False else random.randint(0, self.dst_size-org_h)
        new_im[offset_y:offset_y+org_h, offset_x:offset_x+org_w, :] = cv2.resize(im, None, fx=fx, fy=fy)
        new_mask = np.zeros((self.dst_size, self.dst_size), mask.dtype)
        new_mask[offset_y:offset_y+org_h, offset_x:offset_x+org_w] = cv2.resize(mask, None, fx=fx, fy=fy)
        return new_im, new_mask

class RandomJitter:
    def __init__(self, max_angle=180, max_scale=0.1):
        self._max_angle = max_angle
        self._max_scale = max_scale

    def __call__(self, im, mask):
        h,w,_ = im.shape
        center = w/2.0, h/2.0
        angle = np.random.uniform(-self._max_angle, self._max_angle)
        scale = np.random.uniform(-self._max_scale, self._max_scale) + 1.0
        m = cv2.getRotationMatrix2D(center, angle, scale)
        im = cv2.warpAffine(im, m, (w,h))
        mask = cv2.warpAffine(mask, m, (w,h))
        return im, mask

class RandomCrop:
    def __init__(self, crop_size, rotation=False):
        self.crop_size = crop_size

    def __call__(self, im, mask):
        x = random.randint(0, im.shape[1]-self.crop_size[0])
        y = random.randint(0, im.shape[0]-self.crop_size[1])
        crop_im = im[y:y+self.crop_size[1], x:x+self.crop_size[0], :]
        crop_mask = mask[y:y+self.crop_size[1], x:x+self.crop_size[0]]
        return crop_im, crop_mask

class RandomFlip:
    def __init__(self):
        pass

    def __call__(self, im, mask):
        if random.random() > 0.5:
            im = im[:, ::-1, :]
            mask = mask[:, ::-1]
        return im, mask

class RandomRotate:
    def __init__(self, random_flip=True):
        self.random_flip = random_flip

    def __call__(self, im, mask):
        rotate = random.randint(0, 3)
        if self.random_flip and random.random() > 0.5:
            im = im[:, ::-1, :]
            mask = mask[:, ::-1]
        if rotate > 0:
            im = np.rot90(im, rotate)
            mask = np.rot90(mask, rotate)
        return im.copy(), mask.copy()

