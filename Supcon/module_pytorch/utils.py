#!/usr/bin/env python
# coding:utf-8
import os
import pathlib
import numpy as np
import cv2
import torch

def get_current_dir():
    current_dir = os.getcwd()
    if "DOCKER_WORK_DIR" in os.environ:
        if os.path.exists(os.environ['DOCKER_WORK_DIR']):
            current_dir = os.environ['DOCKER_WORK_DIR']
    work_path = pathlib.Path(current_dir)
    if not work_path.is_absolute():
        current_dir = work_path.resolve()
    return current_dir


def list_normal_files(path):
    files = os.listdir(path)
    # return [filename for filename in os.listdir(path) if not filename.startswith('.') and not filename.startswith('Thumbs.db')]
    if os.path.exists(os.path.join(path, 'Thumbs.db')):
        files.remove('Thumbs.db')
    return files


def device_check(device, gpu_id="0"):
    if device == "cuda" and torch.cuda.is_available():
        device_num = torch.cuda.device_count()
        gpu_id = [int(id) for id in gpu_id.split(",")]
        
        # "over device num" or "over specified gpu id"
        if len(gpu_id) > device_num or any([bool(int(id) >= device_num) for id in gpu_id]):
            old_gpu_id = gpu_id
            gpu_id = [id for id in range(min(len(gpu_id), device_num))]
            print("[Warning] change specified gpu id. ({} -> {})".format(old_gpu_id, gpu_id))
        
        for id in gpu_id:
            print("use gpu #{} : {}".format(id, torch.cuda.get_device_name(id)))
        
        str_gpu_id = ",".join(map(str, gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_id
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if len(gpu_id) == 1: # single (specified gpu id)
            return torch.device("cuda:{}".format(gpu_id[0])), str_gpu_id, False
        else: # multi
            return torch.device("cuda"), str_gpu_id, True
    else:
        str_gpu_id = "None"
        return torch.device("cpu"), str_gpu_id, False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://qiita.com/SKYS/items/cbde3775e2143cad7455
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
