#!/usr/bin/env python
# coding:utf-8
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from module_pytorch import utils

class CustomDataAugmentation(object):
    def __init__(self, data_size, img_size, scale_crop=True, flip=False, rotate=False, brightness=True):
        self.img_size = img_size
        
        # scale & crop
        self.np_rx = np.zeros((data_size), dtype=np.int32)
        self.np_ry = np.zeros((data_size), dtype=np.int32)
        self.scale_add = int(img_size * 0.05) if scale_crop == True else 0
        self.np_rx += self.scale_add
        self.np_ry += self.scale_add
        
        # flip
        flip_rand = 2 if flip == True else 1
        self.np_rf = np.random.randint(flip_rand, size=data_size) # 1: no-flip, 2: flip
        
        # rotate
        rotate_rand = 4 if rotate == True else 1
        self.np_rr = np.random.randint(rotate_rand, size=data_size) # 1: no-rotate, 4:rotate
        
        # brightness
        brightness_rand = [0.5, 2.0] if brightness == True else [1.0, 1.0]
        self.np_rb = np.random.uniform(brightness_rand[0], brightness_rand[1], data_size)

    def __call__(self, idx, img):
        
        rx = self.np_rx[idx]
        ry = self.np_ry[idx]
        rf = self.np_rf[idx]
        rr = self.np_rr[idx]
        rb = self.np_rb[idx]

        minval = np.min(img)
        maxval = np.max(img)
        
        # norm
        if maxval - minval > 0.0:
            img = (img - minval) / (maxval - minval)
        else:
            img = np.clip(img, 0.0, 1.0)
        
        # scale crop
        add_scale_size = self.img_size+self.scale_add
        if img.shape[2] == 1:
            img_resize = cv2.resize(np.uint8(img.reshape(self.img_size, self.img_size)*255), dsize=(add_scale_size, add_scale_size), interpolation=cv2.INTER_CUBIC)
            img_crop = img_resize[ry:ry+self.img_size, rx:rx+self.img_size]
        else:
            img_resize = cv2.resize(np.uint8(img*255), dsize=(add_scale_size, add_scale_size), interpolation=cv2.INTER_CUBIC)
            img_crop = img_resize[ry:ry+self.img_size, rx:rx+self.img_size, :]
        
        # flip
        if rf == 0:
            img_cf = img_crop
        else:
            img_cf = cv2.flip(img_crop, 1) # mirror
        
        # rotation
        img_cfr = np.rot90(img_cf, rr)
        img = img_cfr.reshape(self.img_size, self.img_size, img.shape[2])/255.0 * (maxval - minval) + minval
        
        # brightness
        img *= rb
        
        return img

class CustomMixUp(object):
    def __init__(self, data_size):
        self.l = np.random.beta(0.2, 0.2, data_size)

    def __call__(self, idx, base_img, base_label, mixup_img, mixup_label):
        img = self.l[idx] * base_img + (1.0 - self.l[idx]) * mixup_img
        label = self.l[idx] * base_label + (1.0 - self.l[idx]) * mixup_label
        return img, label

class DatasetAIAF(Dataset):
    def __init__(self, args, np_data, column_list, channels_org, transform_aug=None, transform_mixup=None, transform_twocrop=None):
        self.columns_list = column_list
        self.sample_var_names = ["sample_name1", "sample_name2", "sample_name3"]
        self.img_size = args.IMG_SIZE
        self.channels_org = channels_org
        self.input_16bit_grayscale_tiff = args.INPUT_16BIT_GRAYSCALE_TIFF
        self.normalize_input_img = args.NORMALIZE_INPUT_IMG
        self.separate_abs_sign = args.SEPARATE_ABS_SIGN
        if self.separate_abs_sign == 3:
            self.num_class = args.NUM_CLASSES
            self.offset = args.OFFSET
        
        # transform
        self.transform_aug = transform_aug
        self.transform_mixup = transform_mixup
        self.transform_twocrop = transform_twocrop
        
        # x, y, e
        self.x_np_data = np.stack([np_data[self.columns_list["sample_name1"]], np_data[self.columns_list["sample_name2"]], np_data[self.columns_list["sample_name3"]]], -1)
        self.y_data = np.array(np_data[self.columns_list["target"]], dtype=np.float32)
        self.e_data = np.array(np_data[self.columns_list["error"]], dtype=np.int32)
        self.class_list = np.unique(self.y_data)

    def __len__(self):
        return len(self.x_np_data)

    def _read_image(self, idx):
        img_list = []
        for i, sample_var_name in enumerate(self.sample_var_names):
            img_name = self.x_np_data[idx][i]
            
            if self.input_16bit_grayscale_tiff == True:
                img = np.uint16(utils.imread(img_name, cv2.IMREAD_ANYDEPTH)) # I;16
                interpolation = cv2.INTER_NEAREST
                div = 65535.0
            else:
                img = np.uint8(np.array(Image.open(img_name), dtype='float32')) # L or RGB
                interpolation = cv2.INTER_CUBIC
                div = 255.0
                
            if self.channels_org == 1:
                img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=interpolation) / div
            else: # self.channels_org == 3
                img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=interpolation).reshape(self.img_size, self.img_size, self.channels_org) / div
                
            img_list.append(img)

        # stack image
        X_data = np.dstack(img_list)
        
        # normalize image
        if self.normalize_input_img == True:
            X_data = (X_data - np.average(X_data)) / (np.std(X_data) + 1.0e-10)
        
        # augmentation (numpy)
        if self.transform_aug is not None:
            X_data = self.transform_aug(idx, X_data)
        
        return X_data

    def _read_label(self, idx):
        y_data = self.y_data[idx]
        
        if self.separate_abs_sign == 3:
            y_data_dec = int(y_data) + self.offset
            Y_data = np.zeros(self.num_class, dtype=np.float32)
            
            if y_data_dec < 0:
                Y_data[0] = 1.0
            elif y_data_dec >= self.num_class:
                Y_data[self.num_class - 1] = 1.0
            else:
                Y_data[[y_data_dec]] = 1.0
        elif self.separate_abs_sign == 1:
            y_data_abs = np.abs(y_data)
            y_data_sign = (np.sign(y_data) + 1.0) / 2.0 # np.sign: 1,-1,0
            Y_data = np.hstack([y_data_abs, y_data_sign])
        else:
            Y_data = y_data
        
        return Y_data
    
    def __getitem__(self, idx):
        X_data = self._read_image(idx)
        Y_data = self._read_label(idx)
        E_data = self.e_data[idx]
        
        # mixup (numpy)
        if self.transform_mixup is not None:
            next_idx = idx+1 if idx+1 < len(self.x_np_data) else 0
            mixup_X_data = self._read_image(next_idx)
            mixup_Y_data = self._read_label(next_idx)
            X_data, Y_data = self.transform_mixup(idx, X_data, Y_data, mixup_X_data, mixup_Y_data)
        
        # two crop (for Supcon)
        if self.transform_twocrop is not None:
            X_data = self.transform_twocrop(idx, X_data)
        else:
            # HWC -> CHW
            X_data = X_data.transpose((2,0,1))
            
            # float64 -> float32
            X_data = X_data.astype(np.float32)
            
            # numpy(float) -> tensor
            X_data = torch.from_numpy(X_data)
        
        return X_data, Y_data, E_data


class DatasetEBHI(Dataset):
    def __init__(self, args, np_data, column_list, channels_org, transform_aug=None, transform_mixup=None, transform_twocrop=None):
        self.columns_list = column_list
        self.img_size = args.IMG_SIZE
        self.channels_org = channels_org
        self.normalize_input_img = args.NORMALIZE_INPUT_IMG
        
        # transform
        self.transform_aug = transform_aug
        self.transform_mixup = transform_mixup
        self.transform_twocrop = transform_twocrop
        
        # x, y
        self.x_data = np_data[self.columns_list["filepath"]]
        self.y_data = self._convert_ids(np.array(np_data[self.columns_list["class"]]))

    def __len__(self):
        return len(self.x_data)

    def _convert_ids(self, data_list):
        self.class_list = np.unique(data_list)
        id_data_list = [self._convert_id(y) for y in data_list] # convert str -> id
        return id_data_list
    
    def _convert_id(self, key, exclude=False, replace=0, check_use_in=True):
        replace = replace if exclude == False else None
        for i, label in enumerate(self.class_list):
            if check_use_in == True:
                if label in key:
                    return i
            else:
                if label == key:
                    return i
        return replace
    
    def _read_image(self, idx):
        img_name = self.x_data[idx]
        img = np.uint8(np.array(Image.open(img_name), dtype='float32')) # L or RGB
        img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        X_data = img
        
        # normalize image
        if self.normalize_input_img == True:
            X_data = (X_data - np.average(X_data)) / (np.std(X_data) + 1.0e-10)
        
        # augmentation (numpy)
        if self.transform_aug is not None:
            X_data = self.transform_aug(idx, X_data)
        
        return X_data

    def __getitem__(self, idx):
        X_data = self._read_image(idx)
        Y_data = self.y_data[idx]
        dummy_data = []
        
        # mixup (numpy)
        if self.transform_mixup is not None:
            next_idx = idx+1 if idx+1 < len(self.x_data) else 0
            mixup_X_data = self._read_image(next_idx)
            mixup_Y_data = self.y_data[next_idx]
            X_data, Y_data = self.transform_mixup(idx, X_data, Y_data, mixup_X_data, mixup_Y_data)
        
        # two crop (for Supcon)
        if self.transform_twocrop is not None:
            X_data = self.transform_twocrop(X_data)
        else:
            # HWC -> CHW
            X_data = X_data.transpose((2,0,1))
            
            # float64 -> float32
            X_data = X_data.astype(np.float32)
            
            # numpy(float) -> tensor
            X_data = torch.from_numpy(X_data)
        
        return X_data, Y_data, dummy_data
