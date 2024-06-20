#!/usr/bin/env python
# coding:utf-8
import os
import sys
import gc
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from module_pytorch import utils

def convert_df_to_np_aiaf(df, mode:str):

    temp_buff = []
    column_list = {}
    idx = 0
    
    for column in df.columns.values:
        if column != "Unnamed: 0" and column != "type":
            if column == "target":
                dtype = np.float32
            elif column == "error":
                dtype = np.int32
            else:
                dtype = np.object
            temp_buff.append(df[column].to_numpy(dtype=dtype, copy=True))
            column_list[column] = idx
            idx += 1
    temp_buff = np.array(temp_buff)
    print("{}: ({}) numpy dataset = {} byte".format(mode, temp_buff.shape, sys.getsizeof(temp_buff)))
    
    del df
    gc.collect()
    
    return temp_buff, column_list


def create_path_aiaf(df, mode:str, path_mode:str, img_type:str, sample_var_names:list):
    
    df_mode = df[df['type']==mode].copy()
    
    for sample_var_name in sample_var_names:
        mode_img_names = list(df_mode[sample_var_name])
        df_mode.loc[:, sample_var_name] = [os.path.join(path_mode, "{}.{}".format(n, img_type)) for n in mode_img_names]
    
    df_mode.reset_index(drop=True, inplace=True)
    
    return df_mode


def read_data_aiaf(args):
    
    print("reading aiaf images...")
    
    img_type = args.IMG_TYPE
    path_source_train = os.path.join(args.source_dataset_path, 'image/train')
    path_source_val = os.path.join(args.source_dataset_path, 'image/val')
    path_source_test = os.path.join(args.source_dataset_path, 'image/test')
    path_target_train = os.path.join(args.target_dataset_path, 'image/train')
    path_target_val = os.path.join(args.target_dataset_path, 'image/val')
    path_target_test = os.path.join(args.target_dataset_path, 'image/test')
    
    source_csv_file = args.source_dataset_label_path
    target_csv_file = args.target_dataset_label_path
    
    sample_var_names = ["sample_name1", "sample_name2", "sample_name3"]
    
    img_source_train = glob.glob(os.path.join(path_source_train, "*.{}".format(img_type)))
    img_source_val = glob.glob(os.path.join(path_source_val, "*.{}".format(img_type)))
    img_source_test = glob.glob(os.path.join(path_source_test, "*.{}".format(img_type)))
    img_target_train = glob.glob(os.path.join(path_target_train, "*.{}".format(img_type)))
    img_target_val = glob.glob(os.path.join(path_target_val, "*.{}".format(img_type)))
    img_target_test = glob.glob(os.path.join(path_target_test, "*.{}".format(img_type)))
    
    assert len(img_source_train) != 0, print("train source domain image files dose not exist.")
    assert len(img_source_val) != 0, print("valid source domain image files dose not exist.")
    assert len(img_source_test) != 0, print("test source domain image files dose not exist.")
    assert len(img_target_train) != 0, print("train target domain image files dose not exist.")
    assert len(img_target_val) != 0, print("valid target domain image files dose not exist.")
    assert len(img_target_test) != 0, print("test target domain image files dose not exist.")
    assert os.path.exists(source_csv_file) == True, print("source domain csv file does not exist.")
    assert os.path.exists(target_csv_file) == True, print("target domain csv file does not exist.")
    
    if args.INPUT_16BIT_GRAYSCALE_TIFF == True:
        img = utils.imread(img_source_train[0], cv2.IMREAD_ANYDEPTH)
    else:
        img = np.array(Image.open(img_source_train[0]), dtype='float32')
    
    if img.ndim==2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    
    width, height, channels_org = img.shape
    channels = channels_org * len(sample_var_names)
    
    print("width: ", width)
    print("height: ", height)
    print("channels: ", channels)
    
    del img_source_train, img_source_val, img_source_test, img_target_train, img_target_val, img_target_test
    del img
    gc.collect()
    
    try:
        source_df = pd.read_csv(source_csv_file, encoding="UTF-8")
    except Exception as e:
        source_df = pd.read_csv(source_csv_file, encoding="shift-jis")
        print("[Warning] {}. encoding change 'shift-jis'".format(e))
    
    try:
        target_df = pd.read_csv(target_csv_file, encoding="UTF-8")
    except Exception as e:
        target_df = pd.read_csv(target_csv_file, encoding="shift-jis")
        print("[Warning] {}. encoding change 'shift-jis'".format(e))
    
    # Create a file path list
    # train
    df_source_train = create_path_aiaf(source_df, 'train', path_source_train, img_type, sample_var_names)
    df_target_train = create_path_aiaf(target_df, 'train', path_target_train, img_type, sample_var_names)
    
    # validation
    use_val = True
    if os.path.exists(path_source_val):
        files_val = utils.list_normal_files(path_source_val)
        if len(files_val) == 0:
            use_val = False
    else:
        use_val = False
    
    if use_val == True:
        df_source_val = create_path_aiaf(source_df, 'val', path_source_val, img_type, sample_var_names)
        df_target_val = create_path_aiaf(target_df, 'val', path_target_val, img_type, sample_var_names)
    
    # test
    df_source_test = create_path_aiaf(source_df, 'test', path_source_test, img_type, sample_var_names)
    df_target_test = create_path_aiaf(target_df, 'test', path_target_test, img_type, sample_var_names)
    
    del source_df, target_df
    gc.collect()
    print("read dataset done.")
    
    return df_source_train, df_source_val, df_source_test, df_target_train, df_target_val, df_target_test, channels_org


def convert_df_to_np_ebhi(df, mode:str):

    temp_buff = []
    column_list = {}
    idx = 0
    
    for column in df.columns.values:
        if column != "type":
            temp_buff.append(df[column].to_numpy(copy=True))
            column_list[column] = idx
            idx += 1
    temp_buff = np.array(temp_buff)
    print("{}: numpy dataset = {} byte".format(mode, sys.getsizeof(temp_buff)))
    
    del df
    gc.collect()
    
    return temp_buff, column_list


def create_path_ebhi(df, mode:str, path_mode:str):
    
    df_mode = df[df['type']==mode].copy()
    
    img_names = list(df_mode["filepath"])
    df_mode.loc[:, "filepath"] = [os.path.join(path_mode, file_path) for file_path in img_names]
    
    df_mode.reset_index(drop=True, inplace=True)
    
    return df_mode


def read_data_ebhi(args):
    
    print("reading ebhi images...")
    
    img_type = args.IMG_TYPE
    source_csv_file = args.source_dataset_label_path
    target_csv_file = args.target_dataset_label_path
    
    img_source_dataset = glob.glob(os.path.join(args.source_dataset_path, "*", "*.{}".format(img_type)))
    img_target_dataset = glob.glob(os.path.join(args.target_dataset_path, "*", "*.{}".format(img_type)))

    assert len(img_source_dataset) != 0, print("source domain image files dose not exist.")
    assert len(img_target_dataset) != 0, print("target domain image files dose not exist.")
    assert os.path.exists(source_csv_file) == True, print("source domain csv file does not exist.")
    assert os.path.exists(target_csv_file) == True, print("target domain csv file does not exist.")
    
    img = np.array(Image.open(img_source_dataset[0]), dtype='float32')
    
    if img.ndim==2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    
    width, height, channels_org = img.shape
    
    print("width: ", width)
    print("height: ", height)
    print("channels: ", channels_org)
    
    del img_source_dataset, img_target_dataset
    del img
    gc.collect()
    
    source_df = pd.read_csv(source_csv_file, encoding="UTF-8")
    target_df = pd.read_csv(target_csv_file, encoding="UTF-8")
    
    # Create a file path list
    # train
    df_source_train = create_path_ebhi(source_df, 'train', args.source_dataset_path)
    df_target_train = create_path_ebhi(target_df, 'train', args.target_dataset_path)
    
    # validation
    df_source_val = create_path_ebhi(source_df, 'val', args.source_dataset_path)
    df_target_val = create_path_ebhi(target_df, 'val', args.target_dataset_path)
    
    # test
    df_source_test = create_path_ebhi(source_df, 'test', args.source_dataset_path)
    df_target_test = create_path_ebhi(target_df, 'test', args.target_dataset_path)
    
    del source_df, target_df
    gc.collect()
    print("read dataset done.")
    
    return df_source_train, df_source_val, df_source_test, df_target_train, df_target_val, df_target_test, channels_org
