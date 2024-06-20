#!/usr/bin/env python
# coding:utf-8
import os
import sys
import glob
import time
import gc
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from module_pytorch.dataset import read_data_aiaf, read_data_ebhi, convert_df_to_np_aiaf, convert_df_to_np_ebhi
from module_pytorch.dataloader import CustomDataAugmentation, CustomMixUp, DatasetAIAF, DatasetEBHI
from module_pytorch.model import create_model_full_scratch, create_model_fine_tuning
from module_pytorch.loss import CCEPlusMSELoss, DistSignLoss, MSETotLoss, WeightedMSELoss
from module_pytorch.scheduler import ReduceLROnPlateau, StepDecayLRScheduler
from module_pytorch.utils import AverageMeter, accuracy

def train(args):
    
    Select_ReadData = {"aiaf": read_data_aiaf, "ebhi": read_data_ebhi}
    Select_Converter = {"aiaf": convert_df_to_np_aiaf, "ebhi": convert_df_to_np_ebhi}
    Select_Dataset = {"aiaf": DatasetAIAF, "ebhi": DatasetEBHI}
    
    read_data = Select_ReadData[args.DATASET]
    convert_df_to_np = Select_Converter[args.DATASET]
    Dataset = Select_Dataset[args.DATASET]
    
    # get dataset
    df_train, df_valid, _, channels_org = read_data(args)
    
    # split dataset
    def train_val_split(df_train, df_valid):
        if len(df_valid)==0:
            num_train = len(df_train)
            if num_train==1:
                df_valid = copy.deepcopy(df_train)
            else:
                df_train_val = copy.deepcopy(df_train)
                num_val = np.max([int(num_train * args.VALIDATION_SPLIT), 1])
                num_train = num_train - num_val
                df_train = df_train_val[0:num_train]
                df_valid = df_train_val[num_train:num_train + num_val]
                del df_train_val
            gc.collect()
        return df_train, df_valid
    
    df_train, df_valid = train_val_split(df_train, df_valid)

    print("df_train = {}".format(len(df_train)))
    print("df_valid = {}".format(len(df_valid)))
    
    # [debug]
    if args.DEBUG_DATAFRAME_NUM > 0:
        slice_train_num = min(args.DEBUG_DATAFRAME_NUM, len(df_train))
        slice_valid_num = min(args.DEBUG_DATAFRAME_NUM, len(df_valid))
        df_train = copy.deepcopy(df_train[0:slice_train_num])
        df_valid = copy.deepcopy(df_valid[0:slice_valid_num])
    
    # dataframe to numpy
    np_train, column_train = convert_df_to_np(df_train, "train")
    np_valid, column_valid = convert_df_to_np(df_valid, "valid")
    
    del df_train, df_valid
    gc.collect()

    # transform 
    if args.DATA_AUGMENTATION == True:
        train_transform_aug = CustomDataAugmentation(np_train.shape[1], args.IMG_SIZE, scale_crop=args.SCALE_CROP_AUGMENTATION, 
                                                     flip=args.ROTATE_AUGMENTATION, rotate=args.FLIP_AUGMENTATION, brightness=args.BRIGHTNESS_AUGMENTATION)
    else:
        train_transform_aug = None
    
    if args.MIX_UP == True:
        transform_mixup = CustomMixUp(np_train.shape[1])
    else:
        transform_mixup = None
    
    # dataset
    train_dataset = Dataset(args, np_train, column_train, channels_org, train_transform_aug, transform_mixup, None)
    valid_dataset = Dataset(args, np_valid, column_valid, channels_org, None, None, None)
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=args.NUM_TRAIN_WORKERS, pin_memory=args.PIN_MEMORY, persistent_workers=args.PERSISTENT_WORKERS, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE_TEST, num_workers=args.NUM_VALID_WORKERS, pin_memory=False, persistent_workers=False, shuffle=False, drop_last=False)
    
    print("train_dataset = {}".format(len(train_dataset)))
    print("valid_dataset = {}".format(len(valid_dataset)))
    print("train_loader = {}".format(len(train_loader)))
    print("valid_loader = {}".format(len(valid_loader)))
    
    # create model
    if args.FINE_TUNING == False:
        model = create_model_full_scratch(args.NETWORK, args.NUM_CLASSES, args.SEPARATE_ABS_SIGN, args.TRAINABLE_ONLY_BN)
    else:
        model = create_model_fine_tuning(args.NETWORK, args.NUM_CLASSES, args.SEPARATE_ABS_SIGN)
    model = model.to(args.DEVICE)
    
    # optimizer
    if args.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.BASE_LR, momentum=0.9)
    elif args.OPTIMIZER == "asgd":
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.BASE_LR)
    elif args.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.BASE_LR)
    elif args.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.BASE_LR)

    # scheduler
    if args.SCHEDULER == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer)
    elif args.SCHEDULER == "step_decay":
        scheduler = StepDecayLRScheduler(optimizer, args.BASE_LR, args.CYCLE)

    # loss
    if args.DATASET == "aiaf":
        if args.SEPARATE_ABS_SIGN == 3:
            criterion = CCEPlusMSELoss(args.DEVICE, args.NUM_CLASSES, args.OFFSET)
        elif args.SEPARATE_ABS_SIGN == 1:
            criterion = DistSignLoss(args.LOSS_WEIGHT)
            metrics = MSETotLoss() # for evaluate
        else:
            criterion = WeightedMSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    start_epoch = 0
    history = {"loss": list(), "val_loss": list(), "mse_tot": list(), "val_mse_tot": list()}
    if args.DATASET == "ebhi":
        history.update({"acc": list(), "val_acc": list()})
    
    # fine-tuning (load initial_weights)
    init_weights_paths = glob.glob(os.path.join(args.root_dir, args.initial_weights_folder, "weights*.pth"))
    if len(init_weights_paths)>0:
        checkpoint = torch.load(init_weights_paths[-1])
        model.load_state_dict(checkpoint["model"])
        if args.RESUME == True:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
        print("loaded checkpoint.")
    
    # multi-gpu
    if args.MULTI_GPU:
        model = torch.nn.DataParallel(model)
    
    # log
    log = open(os.path.join(args.save_dir, "training.log"), mode="w")
    if args.DATASET == "aiaf":
        log.write("epoch,loss,val loss\n")
    else:
        log.write("epoch,loss,val loss,accuracy(%),val accuracy(%)\n")
    
    train_loss_per_epoch = AverageMeter()
    valid_loss_per_epoch = AverageMeter()
    if args.SEPARATE_ABS_SIGN == 1:
        train_mse_tot_per_epoch = AverageMeter()
        valid_mse_tot_per_epoch = AverageMeter()
    if args.DATASET == "ebhi":
        train_acc_per_epoch = AverageMeter()
        valid_acc_per_epoch = AverageMeter()
    
    print("training...")
    start = time.perf_counter()
    
    # training
    for epoch in range(start_epoch, args.NUM_EPOCHS):
        
        train_loss_per_epoch.reset()
        valid_loss_per_epoch.reset()
        if args.SEPARATE_ABS_SIGN == 1:
            train_mse_tot_per_epoch.reset()
            valid_mse_tot_per_epoch.reset()
        if args.DATASET == "ebhi":
            train_acc_per_epoch.reset()
            valid_acc_per_epoch.reset()
        
        # train data
        model.train()
        with tqdm(train_loader) as pbar:
            
            pbar.set_description("[Train Epoch {}/{}]".format(epoch, args.NUM_EPOCHS))
            
            for step, (data, label, _) in enumerate(pbar):
                data = data.to(args.DEVICE)
                label = label.to(args.DEVICE)
                
                pred, _ = model(data)
                loss = criterion(pred, label)
                train_loss_per_epoch.update(loss.item())
                
                if args.SEPARATE_ABS_SIGN == 1:
                    mse_tot = metrics(label, pred)
                    train_mse_tot_per_epoch.update(mse_tot.item())
                
                if args.DATASET == "ebhi":
                    acc = accuracy(pred, label, topk=(1,))
                    train_acc_per_epoch.update(acc[0].item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # validation data
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as pbar:
                
                pbar.set_description("[Valid Epoch {}/{}]".format(epoch, args.NUM_EPOCHS))
                
                for step, (data, label, _) in enumerate(pbar):
                    data = data.to(args.DEVICE)
                    label = label.to(args.DEVICE)
                    
                    pred = model(data)
                    loss = criterion(pred, label)
                    valid_loss_per_epoch.update(loss.item())
                    
                    if args.SEPARATE_ABS_SIGN == 1:
                        mse_tot = metrics(label, pred)
                        valid_mse_tot_per_epoch.update(mse_tot.item())
                    
                    if args.DATASET == "ebhi":
                        acc = accuracy(pred, label, topk=(1,))
                        valid_acc_per_epoch.update(acc[0].item())
        
        # scheduler
        scheduler.step(valid_loss_per_epoch.avg)
        
        # log
        if args.DATASET == "aiaf":
            log.write("{},{},{}\n".format(epoch, train_loss_per_epoch.avg, valid_loss_per_epoch.avg))
        else:
            log.write("{},{},{},{},{}\n".format(epoch, train_loss_per_epoch.avg, valid_loss_per_epoch.avg, train_acc_per_epoch.avg, valid_acc_per_epoch.avg))
        
        # add history
        history["loss"].append(train_loss_per_epoch.avg)
        history["val_loss"].append(valid_loss_per_epoch.avg)
        if args.SEPARATE_ABS_SIGN == 1:
            history["mse_tot"].append(train_mse_tot_per_epoch.avg)
            history["val_mse_tot"].append(valid_mse_tot_per_epoch.avg)
        if args.DATASET == "ebhi":
            history["acc"].append(train_acc_per_epoch.avg)
            history["val_acc"].append(valid_acc_per_epoch.avg)
    
        # save
        if (epoch % args.SAVE_FREQ == 0) or (epoch == args.NUM_EPOCHS-1):
            if args.MULTI_GPU:
                state = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch+1,
                    "history": history,
                }
            else:
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch+1,
                    "history": history,
                }
            torch.save(state, os.path.join(args.save_dir, "weights-{:0>4}.pth".format(epoch)))
            del state
        
    elapsed_time = time.perf_counter() - start
    elapsed_time_arr = np.array([elapsed_time])
    print("training cost {} sec".format(elapsed_time))
    
    log.close()
    
    del model
    gc.collect()
    
    loss_train = history["loss"]
    loss_val = history["val_loss"]
    if args.SEPARATE_ABS_SIGN == 1:
        mse_tot_train = history["mse_tot"]
        mse_tot_val = history["val_mse_tot"]
    if args.DATASET == "ebhi":
        acc_train = history["acc"]
        acc_val = history["val_acc"]
    
    # save text
    np.savetxt(os.path.join(args.save_dir, "training_cost_sec.txt"), elapsed_time_arr)
    np.savetxt(os.path.join(args.save_dir, "loss_train.txt"), loss_train)
    np.savetxt(os.path.join(args.save_dir, "loss_val.txt"), loss_val)
    if args.SEPARATE_ABS_SIGN == 1:
        np.savetxt(os.path.join(args.save_dir, "mse_tot_train.txt"), mse_tot_train)
        np.savetxt(os.path.join(args.save_dir, "mse_tot_val.txt"), mse_tot_val)
    if args.DATASET == "ebhi":
        np.savetxt(os.path.join(args.save_dir, "acc_train.txt"), acc_train)
        np.savetxt(os.path.join(args.save_dir, "acc_val.txt"), acc_val)
    
    # plot learning curve
    plt.plot(loss_train, label="loss for training")
    plt.plot(loss_val, label="loss for validation")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(args.save_dir, "loss.png"), dpi=100, orientation="portrait", transparent=False, pad_inches=0.0)
    plt.clf()
    plt.close()
    
    if args.DATASET == "ebhi":
        plt.plot(acc_train, label="accuracy for training")
        plt.plot(acc_val, label="accuracy for validation")
        plt.title("model accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(args.save_dir, "accuracy.png"), dpi=100, orientation="portrait", transparent=False, pad_inches=0.0)
        plt.clf()
        plt.close()
