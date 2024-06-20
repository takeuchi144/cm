from __future__ import print_function

import os
import sys
import argparse
import time
import glob
import math
import gc
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as multiprocessing
import tensorboard_logger as tb_logger
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
from distutils.util import strtobool
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from util import AverageMeter, warmup_learning_rate, adjust_learning_rate, accuracy, plot_confusion_matrix
from module_pytorch.utils import get_current_dir, device_check
from module_pytorch.dataloader import CustomDataAugmentation, CustomMixUp, DatasetAIAF, DatasetEBHI
from module_pytorch.dataset import read_data_aiaf, read_data_ebhi, convert_df_to_np_aiaf, convert_df_to_np_ebhi
from module_pytorch.model_separete import create_model_full_scratch, create_classifier_full_scratch
from module_pytorch.loss import CCEPlusMSELoss
from module_pytorch.scheduler import StepDecayLRScheduler, ReduceLROnPlateau
from module_pytorch.postprocess import postprocess

def parse_option():

    parser = argparse.ArgumentParser("argument for training")
    
    # - switch
    parser.add_argument("--DATASET", type=str, default="aiaf", choices=["aiaf", "ebhi"])
    parser.add_argument("--USE_DATA", type=str, default="target", choices=["sourse", "target"])
    parser.add_argument("--source_dataset_path", type=str, default=r"./datasets/data_10x_AIAF_3cells_rev2(train_val_half)")
    parser.add_argument("--target_dataset_path", type=str, default=r"./datasets/data_20x_AIAF_3cells_rev2(train_val_half)")
    parser.add_argument("--supcon_result_folder", type=str, default="results_supcon_aiaf", help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--linear_result_folder", type=str, default="results_linear_aiaf")
    parser.add_argument("--NUM_CLASSES", type=int, default=(350+650)+1)
    parser.add_argument("--OFFSET", type=int, default=650-10)
    # - freq
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency") # default:10
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency") # default:50
    parser.add_argument("--num_train_workers", type=int, default=4, help="num of workers to use") # default:4
    parser.add_argument("--num_valid_workers", type=int, default=2, help="num of workers to use") # default:2
    parser.add_argument("--num_test_workers", type=int, default=2, help="num of workers to use") # default:2
    parser.add_argument("--pin_memory", type=strtobool, default=True, help="pin memory") # default:True
    parser.add_argument("--persistent_workers", type=strtobool, default=True, help="pin memory") # default:True
    # - optimization
    parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")
    parser.add_argument("--lr_decay_epochs", type=str, default="2,4,6", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.2, help="decay rate for learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    # - other setting
    parser.add_argument("--cosine", type=strtobool, default=False, help="using cosine annealing")
    parser.add_argument("--warm", type=strtobool, default=False, help="warm-up for large batch training")
    # - dataset
    parser.add_argument("--IMG_TYPE", type=str, default="tif", choices=["tif", "png", "jpg"])
    parser.add_argument("--IMG_SIZE", type=int, default=300) # 画像のxyサイズをリサイズして使用(>= 139)
    parser.add_argument("--VALIDATION_SPLIT", type=float, default=0.1) # default 0.2 バリデーション画像の割合（バリデーション画像を用意していない場合にのみ適用）
    parser.add_argument("--INPUT_16BIT_GRAYSCALE_TIFF", type=strtobool, default=True) # 入力する画像が16bitの場合、引数に追加する。
    parser.add_argument("--NORMALIZE_INPUT_IMG", type=strtobool, default=True) # normalize by average and stdev
    parser.add_argument("--DATA_AUGMENTATION", type=strtobool, default=True) # データ拡張する場合はTrue
    parser.add_argument("--SCALE_CROP_AUGMENTATION", type=strtobool, default=True) # 5%画像を拡大してランダムシフト拡張
    parser.add_argument("--ROTATE_AUGMENTATION", type=strtobool, default=True) # 回転拡張（上下のある画像ではオフにする）
    parser.add_argument("--FLIP_AUGMENTATION", type=strtobool, default=True) # 反転拡張
    parser.add_argument("--BRIGHTNESS_AUGMENTATION", type=strtobool, default=True) # 明るさを0.5～2.0倍で拡張
    parser.add_argument("--MIX_UP", type=strtobool, default=False) # Mix Upする場合はTrue
    # - archtecture
    parser.add_argument("--NETWORK", type=str, default="ResNet50", choices=["InceptionV3", "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7", "ResNet50", "VGG19_bn"])
    parser.add_argument("--SEPARATE_ABS_SIGN", type=int, default=3, choices=[0,1,2,3]) # 0固定:ラベルをZ値のまま取得する為。モデルはSEPARATE_ABS_SIGNで分岐する前のところの重みが学習対象。
    parser.add_argument("--TRAINABLE_ONLY_BN", type=strtobool, default=False) # ★ BN層のみ重みを更新する
    # - train/test
    parser.add_argument("--BATCH_SIZE", type=int, default=32) # default 10  バッチサイズ
    parser.add_argument("--BATCH_SIZE_TEST", type=int, default=4) # default 4
    parser.add_argument("--NUM_EPOCHS", type=int, default=5) # default 10  エポック数
    # - add
    parser.add_argument("--SUB_PROCESS", type=str, default="fork", choices=["fork", "spawn"])
    parser.add_argument("--RUN_MODE", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--CYCLE", type=int, default=10)
    parser.add_argument("--GPU_ID", type=str, default="0", help="GPU ID")
    parser.add_argument("--MULTI_GPU", type=strtobool, default=False)
    parser.add_argument("--OPTIMIZER", type=str, default="sgd", choices=["sgd", "asgd", "adam", "adamw"])
    parser.add_argument("--SCHEDULER", type=str, default="step_decay", choices=["step_decay", "reduce_on_plateau"])
    parser.add_argument("--LOAD_CKPT_FOLDER", type=str, default="supcon", choices=["supcon", "liner"])
    parser.add_argument("--DEBUG_DATAFRAME_NUM", type=int, default=-1, help="fix the max number of using data")
    
    args = parser.parse_args()
    
    args.root_dir = get_current_dir()
    
    if args.DATASET == "aiaf":
        args.source_dataset_label_path = os.path.join(args.source_dataset_path, "label/label.csv")
        args.target_dataset_label_path = os.path.join(args.target_dataset_path, "label/label.csv")
    else:
        args.source_dataset_label_path = os.path.join(args.source_dataset_path, "labels.csv")
        args.target_dataset_label_path = os.path.join(args.target_dataset_path, "labels.csv")
    
    args.model_dir = os.path.join(args.root_dir, args.supcon_result_folder)
    args.save_dir = os.path.join(args.root_dir, args.linear_result_folder)
    
    # torch.save対策（日本語パス非対応）
    assert len(args.save_dir) == len(args.save_dir.encode('utf-8')), print("[Error] パスに日本語が含まれています. save_dir = {}\n".format(args.save_dir))
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "tensorboard"), exist_ok=True)
    
    iterations = args.lr_decay_epochs.split(",")
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if args.BATCH_SIZE > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 1
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (1 + math.cos(math.pi * args.warm_epochs / args.NUM_EPOCHS)) / 2
        else:
            args.warmup_to = args.learning_rate
        print("warmup_to = {}".format(args.warmup_to))

    if "EfficientNet" in args.NETWORK:
        efficientnet_input_size = [224, 240, 260, 300, 380, 456, 528, 600] # b0 ~ b7
        args.IMG_SIZE = efficientnet_input_size[int(args.NETWORK[-1])]

    if args.MULTI_GPU:
        args.GPU_ID ="0,1"
    args.DEVICE, args.GPU_ID, args.MULTI_GPU = device_check("cuda", args.GPU_ID)

    # multi-process : create sub-process mode
    if args.SUB_PROCESS == "spawn" and multiprocessing.get_start_method() == "fork":
        multiprocessing.set_start_method("spawn", force=True)
        print("{} setup done".format(multiprocessing.get_start_method()))
    
    # コマンドライン引数の表示(+テキスト保存)
    command_log = open(os.path.join(args.save_dir, "command.txt"), mode="w")
    for kwarg in args._get_kwargs():
        print("--{} : {}".format(kwarg[0], kwarg[1]))
        command_log.write("--{} : {}\n".format(kwarg[0], kwarg[1]))
    command_log.close()

    return args


def set_loader(args):
    
    Select_ReadData = {"aiaf": read_data_aiaf, "ebhi": read_data_ebhi}
    Select_Converter = {"aiaf": convert_df_to_np_aiaf, "ebhi": convert_df_to_np_ebhi}
    Select_Dataset = {"aiaf": DatasetAIAF, "ebhi": DatasetEBHI}
    
    read_data = Select_ReadData[args.DATASET]
    convert_df_to_np = Select_Converter[args.DATASET]
    Dataset = Select_Dataset[args.DATASET]
    
    # dataset
    df_train_source, df_valid_source, df_test_source, df_train_target, df_valid_target, df_test_target, channels_org = read_data(args)
    
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
        return df_train, df_valid
            
    df_train_source, df_valid_source = train_val_split(df_train_source, df_valid_source)
    df_train_target, df_valid_target = train_val_split(df_train_target, df_valid_target)
    
    print("df_train_source = {}".format(len(df_train_source)))
    print("df_train_target = {}".format(len(df_train_target)))
    print("df_valid_source = {}".format(len(df_valid_source)))
    print("df_valid_target = {}".format(len(df_valid_target)))
    print("df_test_source = {}".format(len(df_test_source)))
    print("df_test_target = {}".format(len(df_test_target)))
    
    # [debug] 
    if args.DEBUG_DATAFRAME_NUM > 0:
        slice_train_source_num = min(args.DEBUG_DATAFRAME_NUM, len(df_train_source))
        slice_train_target_num = min(args.DEBUG_DATAFRAME_NUM, len(df_train_target))
        slice_valid_source_num = min(args.DEBUG_DATAFRAME_NUM, len(df_valid_source))
        slice_valid_target_num = min(args.DEBUG_DATAFRAME_NUM, len(df_valid_target))
        slice_test_source_num = min(args.DEBUG_DATAFRAME_NUM, len(df_test_source))
        slice_test_target_num = min(args.DEBUG_DATAFRAME_NUM, len(df_test_target))
        df_train_source = copy.deepcopy(df_train_source[0:slice_train_source_num])
        df_train_target = copy.deepcopy(df_train_target[0:slice_train_target_num])
        df_valid_source = copy.deepcopy(df_valid_source[0:slice_valid_source_num])
        df_valid_target = copy.deepcopy(df_valid_target[0:slice_valid_target_num])
        df_test_source = copy.deepcopy(df_test_source[0:slice_test_source_num])
        df_test_target = copy.deepcopy(df_test_target[0:slice_test_target_num])

    # dataframe to numpy
    np_train_source, column_train_source = convert_df_to_np(df_train_source, "train source")
    np_train_target, column_train_target = convert_df_to_np(df_train_target, "train target")
    np_valid_source, column_valid_source = convert_df_to_np(df_valid_source, "valid source")
    np_valid_target, column_valid_target = convert_df_to_np(df_valid_target, "valid target")
    np_test_source, column_test_source = convert_df_to_np(df_test_source, "test source")
    np_test_target, column_test_target = convert_df_to_np(df_test_target, "test target")
    
    del df_train_source, df_valid_source, df_test_source, df_train_target, df_valid_target, df_test_target
    gc.collect()

    # transform 
    if args.DATA_AUGMENTATION == True:
        train_source_transform_aug = CustomDataAugmentation(np_train_source.shape[1], args.IMG_SIZE, scale_crop=args.SCALE_CROP_AUGMENTATION, 
                                                            flip=args.ROTATE_AUGMENTATION, rotate=args.FLIP_AUGMENTATION, brightness=args.BRIGHTNESS_AUGMENTATION)
        train_target_transform_aug = CustomDataAugmentation(np_train_target.shape[1], args.IMG_SIZE, scale_crop=args.SCALE_CROP_AUGMENTATION, 
                                                            flip=args.ROTATE_AUGMENTATION, rotate=args.FLIP_AUGMENTATION, brightness=args.BRIGHTNESS_AUGMENTATION)
    else:
        train_source_transform_aug = None
        train_target_transform_aug = None
    
    if args.MIX_UP == True:
        train_source_transform_mixup = CustomMixUp(np_train_source.shape[1])
        train_target_transform_mixup = CustomMixUp(np_train_target.shape[1])
    else:
        train_source_transform_mixup = None
        train_target_transform_mixup = None
    
    # dataset
    if args.USE_DATA == "source":
        train_dataset = Dataset(args, np_train_source, column_train_source, channels_org, train_source_transform_aug, train_source_transform_mixup, None)
        valid_dataset = Dataset(args, np_valid_source, column_valid_source, channels_org, None, None, None)
        test_dataset = Dataset(args, np_test_source, column_test_source, channels_org, None, None, None)
        np_test_name2 = copy.deepcopy(np_test_source[1]) # "sample_name2"
        del np_train_target, np_valid_target, np_test_target
    elif  args.USE_DATA == "target":
        train_dataset = Dataset(args, np_train_target, column_train_target, channels_org, train_target_transform_aug, train_target_transform_mixup, None)
        valid_dataset = Dataset(args, np_valid_target, column_valid_target, channels_org, None, None, None)
        test_dataset = Dataset(args, np_test_target, column_test_target, channels_org, None, None, None)
        np_test_name2 = copy.deepcopy(np_test_target[1]) # "sample_name2"
        del np_train_source, np_valid_source, np_test_source
    
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=args.num_train_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE_TEST, num_workers=args.num_valid_workers, pin_memory=False, persistent_workers=False, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE_TEST, num_workers=args.num_test_workers, pin_memory=False, persistent_workers=False, shuffle=False, drop_last=False)
    class_list = test_dataset.class_list
    
    gc.collect()
    
    return train_loader, valid_loader, test_loader, np_test_name2, class_list, channels_org


def set_model(args):

    model = create_model_full_scratch(args.NETWORK, args.TRAINABLE_ONLY_BN)
    classifier = create_classifier_full_scratch(args.NUM_CLASSES, args.SEPARATE_ABS_SIGN, model.in_features)
    if args.DATASET == "aiaf":
        criterion = CCEPlusMSELoss("cuda" if torch.cuda.is_available() else "cpu", args.NUM_CLASSES, args.OFFSET)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        if args.MULTI_GPU ==True and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            classifier = torch.nn.DataParallel(classifier)
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = False

    return model, classifier, criterion


def set_optimizer(args, model):
    
    if args.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(model.get_parameters(base_lr=args.learning_rate), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.OPTIMIZER == "asgd":
        optimizer = torch.optim.ASGD(model.get_parameters(base_lr=args.learning_rate), lr=args.learning_rate)
    elif args.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.get_parameters(base_lr=args.learning_rate), lr=args.learning_rate)
    elif args.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(model.get_parameters(base_lr=args.learning_rate), lr=args.learning_rate)
    
    return optimizer


def set_scheduler(args, optim):
    
    if args.SCHEDULER == "step_decay":
        scheduler = StepDecayLRScheduler(optim, args.learning_rate, args.CYCLE)
    elif args.SCHEDULER == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer)
    
    return scheduler


def train_aiaf(train_loader, model, classifier, criterion, optimizer, epoch, args):
    
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model._forward(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)
        
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print("Train: [{0}][{1}/{2}]\t"
                  "Iter time {batch_time.val:.3f} (avg. {batch_time.avg:.3f})\t"
                  "Loss {loss.val:.3f} (avg. {loss.avg:.3f})\t"
                  .format(epoch, idx + 1, len(train_loader), batch_time=batch_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def train_ebhi(train_loader, model, classifier, criterion, optimizer, epoch, args):
    
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model._forward(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # accuracy
        acc1 = accuracy(output, labels, topk=(1,))
        
        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc1[0].item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print("Train: [{0}][{1}/{2}]\t"
                  "Iter time {batch_time.val:.3f} (avg. {batch_time.avg:.3f})\t"
                  "Loss {loss.val:.3f} (avg. {loss.avg:.3f})\t"
                  "Acc@1 {top1.val:.3f} (avg. {top1.avg:.3f})"
                  .format(epoch, idx + 1, len(train_loader), batch_time=batch_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate_aiaf(valid_loader, model, classifier, criterion, args):
    
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, _) in enumerate(valid_loader):
            
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward
            output = classifier(model._forward(images))
            loss = criterion(output, labels)
            
            # update metric
            losses.update(loss.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print("Valid: [{0}/{1}]\t"
                      "Iter time {batch_time.val:.3f} (avg. {batch_time.avg:.3f})\t"
                      "Loss {loss.val:.4f} (avg. {loss.avg:.4f})"
                      .format(idx, len(valid_loader), batch_time=batch_time, loss=losses))
                sys.stdout.flush()

    return losses.avg


def validate_ebhi(valid_loader, model, classifier, criterion, args):
    
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, _) in enumerate(valid_loader):
            
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward
            output = classifier(model._forward(images))
            loss = criterion(output, labels)
            
            # accuracy
            acc1 = accuracy(output, labels, topk=(1,))

            # update metric
            losses.update(loss.item(), bsz)
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print("Valid: [{0}/{1}]\t"
                      "Iter time {batch_time.val:.3f} (avg. {batch_time.avg:.3f})\t"
                      "Loss {loss.val:.4f} (avg. {loss.avg:.4f})\t"
                      "Acc@1 {top1.val:.3f} (avg. {top1.avg:.3f})"
                      .format(idx, len(valid_loader), batch_time=batch_time, loss=losses, top1=top1))
                sys.stdout.flush()

    return losses.avg, top1.avg


def test_aiaf(test_loader, np_test, model, classifier, args):

    y_test = np.zeros((len(test_loader.dataset), args.NUM_CLASSES), dtype=np.float32)
    y_pred = np.zeros((len(test_loader.dataset), args.NUM_CLASSES), dtype=np.float32)
    e_test = np.zeros(len(test_loader.dataset), dtype=np.int32)
    
    model.eval()
    classifier.eval()
    
    # source
    start = time.perf_counter()
    elapsed_time2 = 0.0

    with torch.no_grad():
        for idx, (images, label, error) in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            
            start2 = time.perf_counter()
            
            # predict
            pred = classifier(model._forward(images))
        
            elapsed_time2 += time.perf_counter() - start2
            
            y_test[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = label.detach().numpy().astype(np.float32)
            y_pred[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = pred.cpu().detach().numpy().astype(np.float32)
            e_test[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = error.detach().numpy().astype(np.int32)

    elapsed_time = time.perf_counter() - start
    elapsed_time_arr = np.array([elapsed_time])
    elapsed_time_arr2 = np.array([elapsed_time2])
    np.savetxt(os.path.join(args.save_dir, "read_apply_cost_sec.txt"), elapsed_time_arr)
    np.savetxt(os.path.join(args.save_dir, "apply_cost_sec.txt"), elapsed_time_arr2)
    print("[{}] read apply cost {} sec".format(args.USE_DATA, str(elapsed_time)))
    print("[{}] apply cost {} sec".format(args.USE_DATA, str(elapsed_time2)))
    print("[{}] length: y_test={}, y_pred={}, e_test={}".format(args.USE_DATA, len(y_test), len(y_pred), len(e_test)))

    postprocess(args, args.save_dir, np_test, y_test, y_pred, e_test)
    print("[{}] complete postprocess".format(args.USE_DATA))
    
    print("done.")


def test_ebhi(test_loader, class_list, model, classifier, args):

    y_output = np.zeros((len(test_loader.dataset), args.NUM_CLASSES), dtype=np.float32)
    y_test = np.zeros(len(test_loader.dataset), dtype=np.int32)
    y_pred = np.zeros(len(test_loader.dataset), dtype=np.int32)
    
    model.eval()
    classifier.eval()
    
    # source
    start = time.perf_counter()
    elapsed_time2 = 0.0

    with torch.no_grad():
        for idx, (images, label, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            
            start2 = time.perf_counter()
            
            # predict
            output = classifier(model._forward(images))
            pred = torch.argmax(output, dim=-1)
            
            elapsed_time2 += time.perf_counter() - start2
            
            y_output[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = output.cpu().detach().numpy().astype(np.float32)
            y_test[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = label.detach().numpy().astype(np.int32)
            y_pred[idx*args.BATCH_SIZE_TEST:(idx+1)*args.BATCH_SIZE_TEST] = pred.cpu().detach().numpy().astype(np.int32)

    elapsed_time = time.perf_counter() - start
    elapsed_time_arr = np.array([elapsed_time])
    elapsed_time_arr2 = np.array([elapsed_time2])
    
    # accuracy (source)
    acc1 = accuracy(torch.from_numpy(y_output), torch.from_numpy(y_test), topk=(1,))
    # confusion matrix (source)
    plot_confusion_matrix(args.save_dir, class_list, y_test, y_pred, extra_msg="accuracy={:.2f}%".format(acc1[0].item()))
    # save text
    np.savetxt(os.path.join(args.save_dir, "read_apply_cost_sec.txt"), elapsed_time_arr)
    np.savetxt(os.path.join(args.save_dir, "apply_cost_sec.txt"), elapsed_time_arr2)
    np.savetxt(os.path.join(args.save_dir, "test_accuracy.txt"), acc1[0].numpy())
    np.savetxt(os.path.join(args.save_dir, "test_y_pred.txt"), y_pred)
    print("[{}] read apply cost {} sec".format(args.USE_DATA, str(elapsed_time)))
    print("[{}] apply cost {} sec".format(args.USE_DATA, str(elapsed_time2)))
    print("[{}] accuracy top1 = {}".format(args.USE_DATA, acc1[0].item()))
    
    print("done.")


def save_model(model, classifier, optimizer, scheduler, args, epoch, history, save_file):
    print("==> Saving...")
    if args.MULTI_GPU:
        state = {
            "model_backbone": model.module.state_dict(),
            "model_classifier": classifier.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch+1,
            "history": history,
        }
    else:
        state = {
            "model_backbone": model.state_dict(),
            "model_classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch+1,
            "history": history,
        }
    torch.save(state, save_file)
    del state


def train(args, train_loader, valid_loader, model, classifier, criterion, optimizer, scheduler):

    start_epoch = 0
    best_loss = float("inf")
    if args.DATASET == "aiaf":
        history = {"loss": list(), "val_loss": list()}
    else:
        history = {"loss": list(), "val_loss": list(), "acc": list(), "val_acc": list()}
    
    # tensorboard
    logger = tb_logger.Logger(logdir=os.path.join(args.save_dir, "tensorboard"), flush_secs=2)
    
    # load checkpoint
    if args.LOAD_CKPT_FOLDER == "supcon":
        if os.path.exists(os.path.join(args.model_dir, "best.pth")):
            print("loading supcon best checkpoint")
            ckpt = torch.load(os.path.join(args.model_dir, "best.pth"), map_location="cpu")
            state_dict = ckpt["model"]
            model.load_state_dict(state_dict)
        else:
            print("loading supcon last checkpoint")
            ckpt = torch.load(os.path.join(args.model_dir, "last.pth"), map_location="cpu")
            state_dict = ckpt["model"]
            model.load_state_dict(state_dict)
    else: # liner
        if os.path.exists(os.path.join(args.save_dir, "best.pth")):
            print("loading liner best checkpoint")
            ckpt = torch.load(os.path.join(args.save_dir, "best.pth"), map_location="cpu")
            state_dict_model = ckpt["model_backbone"]
            state_dict_classifier = ckpt["model_classifier"]
            optimizer_dict = ckpt["optimizer"]
            scheduler_dict = ckpt["scheduler"]
            start_epoch = ckpt["epoch"]
            history = ckpt["history"]
            model.load_state_dict(state_dict_model)
            classifier.load_state_dict(state_dict_classifier)
            optimizer.load_state_dict(optimizer_dict)
            scheduler.load_state_dict(scheduler_dict)
        else:
            init_weights_paths = glob.glob(os.path.join(args.save_dir, "ckpt_epoch_*.pth"))
            if len(init_weights_paths)>0:
                print("loading liner epoch {} checkpoint".format(len(init_weights_paths) - 1))
                ckpt = torch.load(init_weights_paths[-1], map_location="cpu")
                state_dict_model = ckpt["model_backbone"]
                state_dict_classifier = ckpt["model_classifier"]
                optimizer_dict = ckpt["optimizer"]
                scheduler_dict = ckpt["scheduler"]
                start_epoch = ckpt["epoch"]
                history = ckpt["history"]
                model.load_state_dict(state_dict_model)
                classifier.load_state_dict(state_dict_classifier)
                optimizer.load_state_dict(optimizer_dict)
                scheduler.load_state_dict(scheduler_dict)
    
    # log
    log = open(os.path.join(args.save_dir, "finetuning.log"), mode="w")
    log.write("epoch,loss,val loss\n")

    start = time.perf_counter()
    
    # training routine
    for epoch in range(start_epoch, args.NUM_EPOCHS):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        if args.DATASET == "aiaf":
            loss = train_aiaf(train_loader, model, classifier, criterion, optimizer, epoch, args)
        else:
            loss, acc = train_ebhi(train_loader, model, classifier, criterion, optimizer, epoch, args)
        
        # scheduler update
        scheduler.step()

        # eval for one epoch
        if args.DATASET == "aiaf":
            val_loss = validate_aiaf(valid_loader, model, classifier, criterion, args)
        else:
            val_loss, val_acc = validate_ebhi(valid_loader, model, classifier, criterion, args)
        
        # tensorboard logger
        logger.log_value("loss", loss, epoch)
        logger.log_value("val loss", val_loss, epoch)
        logger.log_value("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        history["loss"].append(loss)
        history["val_loss"].append(val_loss)
        if args.DATASET == "ebhi":
            history["acc"].append(acc)
            history["val_acc"].append(val_acc)
        
        log.write(f"{epoch},{loss},{val_loss}\n")
            
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_dir, "ckpt_epoch_{epoch}.pth".format(epoch=epoch))
            save_model(model, classifier, optimizer, scheduler, args, epoch, history, save_file)
            if val_loss < best_loss:
                save_file = os.path.join(args.save_dir, "best.pth")
                save_model(model, classifier, optimizer, scheduler, args, epoch, history, save_file)
                best_loss = val_loss
            
    elapsed_time = time.perf_counter() - start
    elapsed_time_arr = np.array([elapsed_time])
    print("training cost " + str(elapsed_time) + " sec")
    
    # save the last model
    save_file = os.path.join(args.save_dir, "last.pth")
    save_model(model, classifier, optimizer, scheduler, args, args.NUM_EPOCHS, history, save_file)
    
    print("best loss: {:.2f}".format(best_loss))

    log.close()

    # plot learning curve
    loss = history["loss"]
    val_loss = history["val_loss"]
    if args.DATASET == "ebhi":
        acc = history["acc"]
        val_acc = history["val_acc"]
    
    np.savetxt(os.path.join(args.save_dir, "train_loss.txt"), loss)
    np.savetxt(os.path.join(args.save_dir, "val_loss.txt"), val_loss)
    if args.DATASET == "ebhi":
        np.savetxt(os.path.join(args.save_dir, "train_accuracy.txt"), acc)
        np.savetxt(os.path.join(args.save_dir, "val_accuracy.txt"), val_acc)
    np.savetxt(os.path.join(args.save_dir, "training_cost_sec.txt"), elapsed_time_arr)
    
    plt.plot(loss, label="train cls loss")
    plt.plot(val_loss, label="valid cls loss")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(args.save_dir, "train_loss.png"), dpi=100, orientation="portrait", transparent=False, pad_inches=0.0)
    plt.clf()
    plt.close()
    
    if args.DATASET == "ebhi":
        plt.plot(acc, label="train cls accuracy")
        plt.plot(val_acc, label="valid cls accuracy")
        plt.title("model accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(args.save_dir, "train_accuracy.png"), dpi=100, orientation="portrait", transparent=False, pad_inches=0.0)
        plt.clf()
        plt.close()


def test(args, test_loader, np_test, class_list, channels_org, model, classifier):

    # load best model for test
    if os.path.exists(os.path.join(args.save_dir, "best.pth")):
        ckpt = torch.load(os.path.join(args.save_dir, "best.pth"), map_location="cpu")
        print("load best checkpoint")
    else:
        ckpt = torch.load(os.path.join(args.save_dir, "last.pth"), map_location="cpu")
        print("load last checkpoint")
    state_dict_model = ckpt["model_backbone"]
    state_dict_classifier = ckpt["model_classifier"]
    model.load_state_dict(state_dict_model)
    classifier.load_state_dict(state_dict_classifier)
    
    # warm-up
    warmup_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    channels_org = channels_org * 3 if args.DATASET == "aiaf" else channels_org
    with torch.no_grad():
        for i in range(10):
            rand_image = np.random.randint(0, 255, (args.IMG_SIZE, args.IMG_SIZE, channels_org)).astype(np.uint8) # HWC
            tensor_image = warmup_transform(rand_image) # CHW
            inputs = tensor_image.unsqueeze(0) # NCHW
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
            classifier(model._forward(inputs))
    print("warmup done.")
    
    # test
    if args.DATASET == "aiaf":
        test_aiaf(test_loader, np_test, model, classifier, args)
    else:
        test_ebhi(test_loader, class_list, model, classifier, args)


def main():
    
    # argparser
    args = parse_option()

    # build data loader
    train_loader, valid_loader, test_loader, np_test, class_list, channels_org = set_loader(args)
    print("train dataset = {}".format(len(train_loader)))
    print("valid dataset = {}".format(len(valid_loader)))
    print("test dataset = {}".format(len(test_loader)))

    # build model and criterion
    model, classifier, criterion = set_model(args)
    
    # build optimizer
    optimizer = set_optimizer(args, classifier)
    
    # build scheduler
    scheduler = set_scheduler(args, optimizer)
    
    # train / test
    if args.RUN_MODE == "train":
        train(args, train_loader, valid_loader, model, classifier, criterion, optimizer, scheduler)
        test(args, test_loader, np_test, class_list, channels_org, model, classifier)
    else:
        test(args, test_loader, np_test, class_list, channels_org, model, classifier)


if __name__ == '__main__':
    main()
