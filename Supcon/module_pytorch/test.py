#!/usr/bin/env python
# coding:utf-8
import os
import sys
import glob
import time
import gc
import copy
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from module_pytorch.dataset import read_data_aiaf, read_data_ebhi, convert_df_to_np_aiaf, convert_df_to_np_ebhi
from module_pytorch.dataloader import DatasetAIAF, DatasetEBHI
from module_pytorch.model import create_model_full_scratch
from module_pytorch.postprocess import postprocess
from module_pytorch.utils import accuracy, plot_confusion_matrix

def test_aiaf(args, test_loader, model, np_test_name2, save_dir):

    start = time.perf_counter()
    elapsed_time2 = 0.0
    
    y_test = np.zeros((len(test_loader.dataset), args.NUM_CLASSES), dtype=np.float32)
    y_pred = np.zeros((len(test_loader.dataset), args.NUM_CLASSES), dtype=np.float32)
    e_test = np.zeros(len(test_loader.dataset), dtype=np.int32)
    
    model.eval()
    
    # test
    with torch.no_grad():
        for idx, (data, label, error) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.to(args.DEVICE)
            
            start2 = time.perf_counter()
            
            # predict
            output = model(data)
            pred = output
        
            elapsed_time2 += time.perf_counter() - start2
            
            y_test[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = label.detach().numpy().astype(np.float32)
            y_pred[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = pred.cpu().detach().numpy().astype(np.float32)
            e_test[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = error.detach().numpy().astype(np.int32)
            
    elapsed_time = time.perf_counter() - start
    elapsed_time_arr = np.array([elapsed_time])
    elapsed_time_arr2 = np.array([elapsed_time2])
    
    np.savetxt(os.path.join(save_dir, "read_apply_cost_sec.txt"), elapsed_time_arr)
    np.savetxt(os.path.join(save_dir, "apply_cost_sec.txt"), elapsed_time_arr2)
    print("read apply cost {} sec".format(str(elapsed_time)))
    print("apply cost {} sec".format(str(elapsed_time2)))
    
    # post-processing
    postprocess(args, save_dir, np_test_name2, y_test, y_pred, e_test)


def test_ebhi(args, test_loader, model, class_list, save_dir):

    start = time.perf_counter()
    elapsed_time2 = 0.0
    
    y_output = np.zeros((len(test_loader.dataset), args.NUM_CLASSES), dtype=np.float32)
    y_test = np.zeros(len(test_loader.dataset), dtype=np.int32)
    y_pred = np.zeros(len(test_loader.dataset), dtype=np.int32)
    
    model.eval()
    
    # test
    with torch.no_grad():
        for idx, (data, label, error) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.to(args.DEVICE)
            
            start2 = time.perf_counter()
            
            # predict
            output = model(data)
            pred = torch.argmax(output, dim=-1)
        
            elapsed_time2 += time.perf_counter() - start2
            
            y_output[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = output.cpu().detach().numpy().astype(np.float32)
            y_test[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = label.detach().numpy().astype(np.int32)
            y_pred[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = pred.cpu().detach().numpy().astype(np.int32)
            
    elapsed_time = time.perf_counter() - start
    elapsed_time_arr = np.array([elapsed_time])
    elapsed_time_arr2 = np.array([elapsed_time2])
    
    np.savetxt(os.path.join(save_dir, "read_apply_cost_sec.txt"), elapsed_time_arr)
    np.savetxt(os.path.join(save_dir, "apply_cost_sec.txt"), elapsed_time_arr2)
    print("read apply cost {} sec".format(str(elapsed_time)))
    print("apply cost {} sec".format(str(elapsed_time2)))
    
    # accuracy
    acc1 = accuracy(torch.from_numpy(y_output), torch.from_numpy(y_test), topk=(1,))
    # confusion matrix
    plot_confusion_matrix(save_dir, class_list, y_test, y_pred, extra_msg="accuracy={:.2f}%".format(acc1[0].item()))
    np.savetxt(os.path.join(save_dir, "test_accuracy.txt"), acc1[0].numpy())
    np.savetxt(os.path.join(save_dir, "test_y_pred.txt"), y_pred)
    print("accuracy top1 = {}".format(acc1[0].item()))


def test(args):
    
    Select_ReadData = {"aiaf": read_data_aiaf, "ebhi": read_data_ebhi}
    Select_Converter = {"aiaf": convert_df_to_np_aiaf, "ebhi": convert_df_to_np_ebhi}
    Select_Dataset = {"aiaf": DatasetAIAF, "ebhi": DatasetEBHI}
    
    read_data = Select_ReadData[args.DATASET]
    convert_df_to_np = Select_Converter[args.DATASET]
    Dataset = Select_Dataset[args.DATASET]
    
    save_dir = args.save_dir
    
    # get dataset
    _, _, df_test, channels_org = read_data(args)
    
    print("df_test = {}".format(len(df_test)))
    
    # [debug]
    if args.DEBUG_DATAFRAME_NUM > 0:
        slice_test_num = min(args.DEBUG_DATAFRAME_NUM, len(df_test))
        df_test = copy.deepcopy(df_test[0:slice_test_num])
    
    # dataframe to numpy
    np_test, column_test = convert_df_to_np(df_test, "test")
    
    del df_test
    gc.collect()

    # dataset
    test_dataset = Dataset(args, np_test, column_test, channels_org, None, None, None)
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE_TEST, num_workers=args.NUM_TEST_WORKERS, pin_memory=False, persistent_workers=False, shuffle=False, drop_last=False)
    class_list = test_dataset.class_list
    
    np_test_name2 = copy.deepcopy(np_test[1]) # "sample_name2"
    
    print("test_dataset = {}".format(len(test_dataset)))
    print("test_loader = {}".format(len(test_loader)))
    print("class_list = {}".format(class_list))
    
    # model
    model = create_model_full_scratch(args.NETWORK, args.NUM_CLASSES, args.SEPARATE_ABS_SIGN)
    
    # load weight
    filepath_weights = glob.glob(os.path.join(save_dir, "weights-*.pth"))
    assert len(filepath_weights) > 0, print("[Error] not exist weights files. searched directory = {}".format(save_dir))
    checkpoint = torch.load(filepath_weights[-1])
    model.load_state_dict(checkpoint["model"]) # load latest weights file
    model = model.to(args.DEVICE)
    model.eval()
    print("loaded weights file. {}".format(filepath_weights[-1]))
    
    # warm-up
    model.eval()
    warmup_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    channels_org = channels_org * 3 if args.DATASET == "aiaf" else channels_org
    with torch.no_grad():
        for i in range(10):
            rand_image = np.random.randint(0, 255, (args.IMG_SIZE, args.IMG_SIZE, channels_org)).astype(np.uint8) # HWC
            tensor_image = warmup_transform(rand_image) # CHW
            inputs = tensor_image.unsqueeze(0).to(args.DEVICE) # NCHW
            model(inputs)
    print("warmup done.")
    
    if args.DATASET == "aiaf":
       test_aiaf(args, test_loader, model, np_test_name2, save_dir)
    else:
       test_ebhi(args, test_loader, model, class_list, save_dir)
    
    gc.collect()
    print("done.")
