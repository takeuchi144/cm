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
from torch.utils.data import DataLoader
from losses import SupConLoss
from util import TwoCropTransform, AverageMeter, warmup_learning_rate, adjust_learning_rate
from module_pytorch.utils import get_current_dir, device_check
from module_pytorch.dataloader import CustomDataAugmentation, CustomMixUp, DatasetAIAF, DatasetEBHI
from module_pytorch.dataset import read_data_aiaf, read_data_ebhi, convert_df_to_np_aiaf, convert_df_to_np_ebhi
from module_pytorch.model_separete import create_model_full_scratch
from module_pytorch.scheduler import StepDecayLRScheduler, ReduceLROnPlateau

def parse_option():
    
    parser = argparse.ArgumentParser("argument for training")
    
    # - switch
    parser.add_argument("--DATASET", type=str, default="aiaf", choices=["aiaf", "ebhi"])
    parser.add_argument("--USE_DATA", type=str, default="source", choices=["all", "source", "target"])
    parser.add_argument("--source_dataset_path", type=str, default=r"./datasets/data_10x_AIAF_3cells_rev2(train_val_half)")
    parser.add_argument("--target_dataset_path", type=str, default=r"./datasets/data_20x_AIAF_3cells_rev2(train_val_half)")
    parser.add_argument("--result_folder", type=str, default="results_supcon_aiaf", help="Where to save logs, checkpoints and debugging images.")
    # - freq
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency") # default:10
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency") # default:50
    parser.add_argument("--num_train_workers", type=int, default=4, help="num of workers to use") # default:4
    parser.add_argument("--pin_memory", type=strtobool, default=True, help="pin memory") # default:True
    parser.add_argument("--persistent_workers", type=strtobool, default=True, help="pin memory") # default:True
    # - optimization
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate") # default:0.2
    parser.add_argument("--lr_decay_epochs", type=str, default="2,4,6", help="where to decay lr, can be a list") # default:700,800,900
    parser.add_argument("--lr_decay_rate", type=float, default=1e-3, help="decay rate for learning rate") # default:0.1
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay") # default:1e-4
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum") # default: 0.9
    parser.add_argument("--method", type=str, default="SupCon", choices=["SupCon", "SimCLR"], help="choose method")
    parser.add_argument("--temp", type=float, default=0.07, help="temperature for loss function") # default:0.07
    # - other setting
    parser.add_argument("--cosine", type=strtobool, default=False, help="using cosine annealing")
    parser.add_argument("--syncBN", type=strtobool, default=False, help="using synchronized batch normalization")
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
    parser.add_argument("--SEPARATE_ABS_SIGN", type=int, default=0, choices=[0,1,2,3]) # 0固定:ラベルをZ値のまま取得する為。モデルはSEPARATE_ABS_SIGNで分岐する前のところの重みが学習対象。
    parser.add_argument("--TRAINABLE_ONLY_BN", type=strtobool, default=False) # ★ BN層のみ重みを更新する
    # - train/test
    parser.add_argument("--BATCH_SIZE", type=int, default=32) # default 10  バッチサイズ
    parser.add_argument("--NUM_EPOCHS", type=int, default=5) # default 10  エポック数
    # - add
    parser.add_argument("--SUB_PROCESS", type=str, default="fork", choices=["fork", "spawn"])
    parser.add_argument("--CYCLE", type=int, default=10)
    parser.add_argument("--GPU_ID", type=str, default="0", help="GPU ID")
    parser.add_argument("--MULTI_GPU", type=strtobool, default=False)
    parser.add_argument("--ACCUM_ITER", type=int, default=1, help="batch accumlate. 1 or more")
    parser.add_argument("--OPTIMIZER", type=str, default="sgd", choices=["sgd", "asgd", "adam", "adamw"])
    parser.add_argument("--SCHEDULER", type=str, default="cosine_annealing", choices=["cosine_annealing", "step_decay", "reduce_on_plateau"])
    parser.add_argument("--LOAD_CKPT", type=strtobool, default=False)
    parser.add_argument("--DEBUG_DATAFRAME_NUM", type=int, default=-1, help="fix the max number of using data")
    
    args = parser.parse_args()

    args.root_dir = get_current_dir()
    if args.DATASET == "aiaf":
        args.source_dataset_label_path = os.path.join(args.source_dataset_path, "label/label.csv")
        args.target_dataset_label_path = os.path.join(args.target_dataset_path, "label/label.csv")
    else:
        args.source_dataset_label_path = os.path.join(args.source_dataset_path, "labels.csv")
        args.target_dataset_label_path = os.path.join(args.target_dataset_path, "labels.csv")
    args.save_dir = os.path.join(args.root_dir, args.result_folder)
    
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
    command_log = open(os.path.join(args.save_dir, 'command.txt'), mode='w')
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
    df_train_source, _, _, df_train_target, _, _, channels_org = read_data(args)
    
    print("df_train_source = {}".format(len(df_train_source)))
    print("df_train_target = {}".format(len(df_train_target)))
    
    # [debug] 
    if args.DEBUG_DATAFRAME_NUM > 0:
        slice_train_source_num = min(args.DEBUG_DATAFRAME_NUM, len(df_train_source))
        slice_train_target_num = min(args.DEBUG_DATAFRAME_NUM, len(df_train_target))
        df_train_source = copy.deepcopy(df_train_source[0:slice_train_source_num])
        df_train_target = copy.deepcopy(df_train_target[0:slice_train_target_num])

    # dataframe to numpy
    np_train_source, column_train_source = convert_df_to_np(df_train_source, "train source")
    np_train_target, column_train_target = convert_df_to_np(df_train_target, "train target")
    
    del df_train_source, df_train_target
    gc.collect()

    # transform 
    if args.DATA_AUGMENTATION == True:
        train_source_transform_aug_1 = CustomDataAugmentation(np_train_source.shape[1], args.IMG_SIZE, scale_crop=args.SCALE_CROP_AUGMENTATION, 
                                                              flip=args.ROTATE_AUGMENTATION, rotate=args.FLIP_AUGMENTATION, brightness=args.BRIGHTNESS_AUGMENTATION)
        train_source_transform_aug_2 = CustomDataAugmentation(np_train_source.shape[1], args.IMG_SIZE, scale_crop=args.SCALE_CROP_AUGMENTATION, 
                                                              flip=args.ROTATE_AUGMENTATION, rotate=args.FLIP_AUGMENTATION, brightness=args.BRIGHTNESS_AUGMENTATION)
        train_target_transform_aug_1 = CustomDataAugmentation(np_train_target.shape[1], args.IMG_SIZE, scale_crop=args.SCALE_CROP_AUGMENTATION, 
                                                              flip=args.ROTATE_AUGMENTATION, rotate=args.FLIP_AUGMENTATION, brightness=args.BRIGHTNESS_AUGMENTATION)
        train_target_transform_aug_2 = CustomDataAugmentation(np_train_target.shape[1], args.IMG_SIZE, scale_crop=args.SCALE_CROP_AUGMENTATION, 
                                                              flip=args.ROTATE_AUGMENTATION, rotate=args.FLIP_AUGMENTATION, brightness=args.BRIGHTNESS_AUGMENTATION)
    else:
        train_source_transform_aug_1 = None
        train_source_transform_aug_2 = None
        train_target_transform_aug_1 = None
        train_target_transform_aug_2 = None
    
    if args.MIX_UP == True:
        train_source_transform_mixup = CustomMixUp(np_train_source.shape[1])
        train_target_transform_mixup = CustomMixUp(np_train_target.shape[1])
    else:
        train_source_transform_mixup = None
        train_target_transform_mixup = None
    
    train_source_transform_twocrop = TwoCropTransform(train_source_transform_aug_1, train_source_transform_aug_2)
    train_target_transform_twocrop = TwoCropTransform(train_target_transform_aug_1, train_target_transform_aug_2)

    if args.USE_DATA == "all":
        train_source_dataset = Dataset(args, np_train_source, column_train_source, channels_org, None, train_source_transform_mixup, train_source_transform_twocrop)
        train_target_dataset = Dataset(args, np_train_target, column_train_target, channels_org, None, train_target_transform_mixup, train_target_transform_twocrop)
        # concat
        train_dataset = torch.utils.data.ConcatDataset([train_source_dataset, train_target_dataset])
    elif args.USE_DATA == "source":
        train_dataset = Dataset(args, np_train_source, column_train_source, channels_org, None, train_source_transform_mixup, train_source_transform_twocrop)
        del np_train_target, column_train_target
    elif  args.USE_DATA == "target":
        train_dataset = Dataset(args, np_train_target, column_train_target, channels_org, None, train_target_transform_mixup, train_target_transform_twocrop)
        del np_train_source, column_train_source
    gc.collect()
    
    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, num_workers=args.num_train_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers, shuffle=True, drop_last=True)
    
    return train_loader


def set_model(args):
    
    model = create_model_full_scratch(args.NETWORK, args.TRAINABLE_ONLY_BN)
    criterion = SupConLoss(temperature=args.temp)
    
    # enable synchronized Batch Normalization
    if args.syncBN:
        try:
            model = apex.parallel.convert_syncbn_model(model)
        except Exception as e:
            print("[Warning] {}".format(str(e.args)))

    if torch.cuda.is_available():
        if args.MULTI_GPU == True and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = False

    return model, criterion


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
    
    if args.SCHEDULER == "cosine_annealing":
        # https://github.com/Spijkervet/SimCLR
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.NUM_EPOCHS, eta_min=0, last_epoch=-1)
    elif args.SCHEDULER == "step_decay":
        scheduler = StepDecayLRScheduler(optim, args.learning_rate, args.CYCLE)
    elif args.SCHEDULER == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer)
    
    return scheduler


def train_per_epoch(args, train_loader, model, criterion, optimizer, epoch):
    
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
   
    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # forward
        features = model(images)
        
        # compute loss
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if args.method == "SupCon":
            loss = criterion(features, labels)
        elif args.method == "SimCLR":
            loss = criterion(features)
        else:
            raise ValueError("contrastive method not supported: {}".format(args.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        loss = loss / args.ACCUM_ITER
        loss.backward()
        if ((idx + 1) % args.ACCUM_ITER == 0) or ((idx + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if ((idx + 1) % args.print_freq == 0) or ((idx + 1) == len(train_loader)):
            print("Train: [{0}][{1}/{2}]\t"
                  "Iter time {batch_time.val:.3f} (avg. {batch_time.avg:.3f})\t"
                  "loss {loss.val:.3f} (avg. {loss.avg:.3f})\t"
                  .format(epoch, idx + 1, len(train_loader), batch_time=batch_time, loss=losses))
            sys.stdout.flush()

    return  losses.avg


def save_model(model, optimizer, scheduler, args, epoch, history, save_file):
    print("==> Saving...")
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
    torch.save(state, save_file)
    del state


def train(args, train_loader, model, criterion, optimizer, scheduler):

    best_loss = float("inf")
    start_epoch = 0
    history = {"loss": list()}
    
    # tensorboard
    logger = tb_logger.Logger(logdir=os.path.join(args.save_dir, "tensorboard"), flush_secs=2)
    
    # load checkpoint
    if args.LOAD_CKPT == True:
        if os.path.exists(os.path.join(args.save_dir, "last.pth")):
            print("loading last checkpoint")
            checkpoint = torch.load(os.path.join(args.save_dir, "last.pth"))
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
        else:
            init_weights_paths = glob.glob(os.path.join(args.save_dir, "ckpt_epoch_*.pth"))
            if len(init_weights_paths)>0:
                print("loading epoch {} checkpoint".format(len(init_weights_paths) - 1))
                checkpoint = torch.load(init_weights_paths[-1])
                model.load_state_dict(checkpoint["model"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                start_epoch = checkpoint["epoch"]
                history = checkpoint["history"]

    # log
    log = open(os.path.join(args.save_dir, "pretrain.log"), mode="a")
    log.write("epoch,loss\n")
    
    start = time.perf_counter()

    # training routine
    for epoch in range(start_epoch, args.NUM_EPOCHS):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        loss = train_per_epoch(args, train_loader, model, criterion, optimizer, epoch)

        # scheduler update
        scheduler.step()

        # tensorboard logger
        logger.log_value("loss", loss, epoch)
        logger.log_value("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        
        history["loss"].append(loss)
        
        log.write(f"{epoch},{loss}\n")

        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_dir, "ckpt_epoch_{epoch}.pth".format(epoch=epoch))
            save_model(model, optimizer, scheduler, args, epoch, history, save_file)
            if loss < best_loss:
                save_file = os.path.join(args.save_dir, "best.pth")
                save_model(model, optimizer, scheduler, args, epoch, history, save_file)
                best_loss = loss

    elapsed_time = time.perf_counter() - start
    elapsed_time_arr = np.array([elapsed_time])
    print("training cost " + str(elapsed_time) + " sec")
    
    # save the last model
    save_file = os.path.join(args.save_dir, "last.pth")
    save_model(model, optimizer, scheduler, args, args.NUM_EPOCHS, history, save_file)
    
    print("best loss: {:.2f}".format(best_loss))

    log.close()

    # plot learning curve
    loss_value = history["loss"]
    np.savetxt(os.path.join(args.save_dir, "loss.txt"), loss_value)
    np.savetxt(os.path.join(args.save_dir, "training_cost_sec.txt"), elapsed_time_arr)
    
    plt.plot(loss_value, label="supcon loss")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(args.save_dir, "loss.png"), dpi=100, orientation="portrait", transparent=False, pad_inches=0.0)
    plt.clf()
    plt.close()

    print("done.")


def main():
    
    # argparser
    args = parse_option()
    
    # build data loader
    train_loader = set_loader(args)
    print("train dataset = {}".format(len(train_loader)))
    
    # build model and criterion
    model, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(args, model)
    
    # build optimizer
    scheduler = set_scheduler(args, optimizer)
    
    # train
    train(args, train_loader, model, criterion, optimizer, scheduler)


if __name__ == '__main__':
    main()
