#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/20 11:24
# @Author  : zhangchao
# @File    : base.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import torch
import numpy as np
import random

from spatialign.utils import get_format_time


class spatiAlignBase:
    def __init__(self, gpu=None, torch_thread=24, seed=42, save_path="./output"):
        self.set_seed(seed, torch_thread)
        self.device = self.set_device(gpu=gpu)
        self.ckpt_path, self.res_path = self.init_path(save_path)

    def set_seed(self, seed=42, n_thread=24):
        torch.set_num_threads(n_thread)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def set_device(self, gpu=None):
        if torch.cuda.is_available() and gpu is not None:
            if float(gpu) >= 0:
                device = torch.device(f"cuda:{gpu}")
            else:
                print(f"{get_format_time}  Got an invalid GPU device ids, can not using GPU device to training...")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        return device

    def init_path(self, save_path):
        assert save_path is not None, "Error, Got an invalid save path"
        ckpt_path = osp.join(save_path, "ckpt")
        res_path = osp.join(save_path, "res")
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(res_path, exist_ok=True)
        return ckpt_path, res_path

    def save_checkpoint(self, model):
        assert osp.exists(self.ckpt_path)
        torch.save(model.state_dict(), osp.join(self.ckpt_path, "spatialign.bgi"))

    def load_checkpoint(self, model):
        ckpt_file = osp.join(self.ckpt_path, "spatialign.bgi")
        assert osp.exists(ckpt_file)
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        state_dict = model.state_dict()
        trained_dict = {k: v for k, v in checkpoint.items() if k in state_dict}
        state_dict.update(trained_dict)
        model.load_state_dict(state_dict)
        # return model
