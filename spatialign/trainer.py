#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/12/23 9:53 AM
# @Author  : zhangchao
# @File    : trainer.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import random
import scipy.sparse as sp
from anndata import AnnData
from typing import Union
from collections import defaultdict
from module import contrast_loss, trivial_entropy, cross_instance_loss
from spatialign import DGIAlignment
from utils import get_format_time, Dataset, get_running_time, EarlyStopping


class Spatialign:
    """
    Spatialign Model
    :param data_path:
        Input dataset path.
    :param min_genes:
         Minimum number of genes expressed required for a cell to pass filtering, default 20.
    :param min_cells:
        Minimum number of cells expressed required for a gene to pass filtering, default 20.
    :param batch_key:
        The batch annotation to :attr:`obs` using this key, default, 'batch'.
    :param is_norm_log:
        Whether to perform 'sc.pp.normalize_total' and 'sc.pp.log1p' processing, default, True.
    :param is_scale:
        Whether to perform 'sc.pp.scale' processing, default, False.
    :param is_hvg:
        Whether to perform 'sc.pp.highly_variable_genes' processing, default, False.
    :param is_reduce:
        Whether to perform PCA reduce dimensional processing, default, False.
    :param n_pcs:
        PCA dimension reduction parameter, valid when 'is_reduce' is True, default, 100.
    :param n_hvg:
        'sc.pp.highly_variable_genes' parameter, valid when 'is_reduce' is True, default, 2000.
    :param n_neigh:
        The number of neighbors selected when constructing a spatial neighbor graph. default, 15.
    :param is_undirected:
        Whether the constructed spatial neighbor graph is undirected graph, default, True.
    :param latent_dims:
        The number of embedding dimensions, default, 100.
    :param seed:
        Random seed.
    :param gpu:
        Whether the GPU device is using to train spatialign.
    :param save_path:
        The path of alignment dataset and saved spatialign.
    :return:
    """
    def __init__(self,
                 *data_path: str,
                 min_genes: int = 20,
                 min_cells: int = 20,
                 batch_key: str = "batch",
                 is_norm_log: bool = True,
                 is_scale: bool = False,
                 is_hvg: bool = False,
                 is_reduce: bool = False,
                 n_pcs: int = 100,
                 n_hvg: int = 2000,
                 n_neigh: int = 15,
                 is_undirected: bool = True,
                 latent_dims: int = 100,
                 seed: int = 42,
                 gpu: Union[int, str, None] = None,
                 save_path: str = None):
        self.set_seed(seed)
        self.device = self.set_device(gpu)
        self.ckpt_path, self.res_path = self.init_path(save_path)
        self.dataset = Dataset(*data_path,
                               min_genes=min_genes,
                               min_cells=min_cells,
                               batch_key=batch_key,
                               is_norm_log=is_norm_log,
                               is_scale=is_scale,
                               is_hvg=is_hvg,
                               is_reduce=is_reduce,
                               n_pcs=n_pcs,
                               n_hvg=n_hvg,
                               n_neigh=n_neigh,
                               is_undirected=is_undirected)
        self.model = self.set_model(latent_dims=latent_dims, n_domain=self.dataset.n_domain)
        self.header_bank = self.init_bank()

    def set_seed(self, seed=42, n_thread=24):
        torch.set_num_threads(n_thread)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def set_device(self, gpu):
        if torch.cuda.is_available() and gpu is not None:
            if (float(gpu) % 2 == 0) and float(gpu) >= 0:
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

    def set_model(self, latent_dims, n_domain):
        model = DGIAlignment(input_dims=self.dataset.inner_dims,
                             output_dims=latent_dims,
                             n_domain=n_domain,
                             act=nn.ELU(),
                             p=0.2)
        print(f"{get_format_time()} {model.__class__.__name__}: \n{model}")
        model.to(self.device)
        return model

    def save_checkpoint(self):
        assert osp.exists(self.ckpt_path)
        torch.save(self.model.state_dict(), osp.join(self.ckpt_path, "spatialign.bgi"))

    def load_checkpoint(self):
        ckpt_file = osp.join(self.ckpt_path, "spatialign.bgi")
        assert osp.exists(ckpt_file)
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        state_dict = self.model.state_dict()
        trained_dict = {k: v for k, v in checkpoint.items() if k in state_dict}
        state_dict.update(trained_dict)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    @get_running_time
    def init_bank(self):
        self.model.eval()
        header_bank = defaultdict()
        for idx, data in enumerate(self.dataset.loader_list):
            data = data.to(self.device)
            graph_loss, dgi_loss, recon_loss, latent_x, recon_x = self.model(
                x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, domain_idx=data.domain_idx,
                neigh_mask=data.neigh_graph)
            header_bank[idx] = latent_x.detach()
        return header_bank

    @torch.no_grad()
    def update_bank(self, idx, feat, alpha=0.5):
        self.header_bank[idx] = feat * alpha + (1 - alpha) * self.header_bank[idx]

    @get_running_time
    def train(self,
              lr: float = 1e-3,
              max_epoch: int = 500,
              alpha: float = 0.5,
              patient: int = 15,
              tau1: float = 0.2,
              tau2: float = 1.,
              tau3: float = 0.5):
        """
        Training spatialign
        :param lr:
            Learning rate, default, 1e-3.
        :param max_epoch:
            The number of maximum epochs, default, 500.
        :param alpha:
            The momentum parameter, default, 0.5
        :param patient:
            Early stop parameter, default, 15.
        :param tau1:
            Instance level and pseudo prototypical cluster level contrastive learning parameters, default, 0.2
        :param tau2:
            Pseudo prototypical cluster entropy parameter, default, 1.
        :param tau3:
            Cross-batch instance self-supervised learning parameter, default, 0.5
        :return:
        """
        early_stop = EarlyStopping(patience=patient)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
        # scaler = torch.cuda.amp.GradScaler()
        self.model.train()

        for eph in range(max_epoch):
            epoch_loss = []
            for idx, data in enumerate(self.dataset.loader_list):
                data = data.to(self.device, non_blocking=True)
                # with torch.cuda.amp.autocast():
                graph_loss, dgi_loss, recon_loss, latent_x, recon_x = self.model(
                    x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight, domain_idx=data.domain_idx,
                    neigh_mask=data.neigh_graph)

                # update memory bank
                self.update_bank(idx, latent_x, alpha=alpha)
                intra_inst = contrast_loss(feat1=latent_x, feat2=self.header_bank[idx], tau=tau1, weight=1.)
                intra_clst = contrast_loss(feat1=latent_x.T, feat2=self.header_bank[idx].T, tau=tau1, weight=1.)
                # Maximize clustering entropy to avoid all data clustering into the same class
                entropy_clst = trivial_entropy(feat=latent_x, tau=tau2, weight=1.)
                loss = graph_loss + dgi_loss + recon_loss + intra_inst + entropy_clst + intra_clst
                for i in np.delete(range(len(self.dataset.data_list)), idx):
                    inter_loss = cross_instance_loss(
                        feat1=latent_x, feat2=self.header_bank[i], tau=tau3, weight=1.)
                    loss += inter_loss

                epoch_loss.append(loss)
            optimizer.zero_grad()
            # scaler.scale(sum(epoch_loss)).backward()
            # scaler.step(optimizer)
            # scaler.update()
            sum(epoch_loss).backward()
            optimizer.step()
            scheduler.step()

            early_stop(sum(epoch_loss).detach().cpu().numpy())
            print(f"\r  {get_format_time()} "
                  f"Epoch: {eph} "
                  f"Loss: {sum(epoch_loss).detach().cpu().numpy():.4f} "
                  f"Loss min: {early_stop.loss_min:.4f} "
                  f"EarlyStopping counter: {early_stop.counter} out of {patient}",
                  flush=True, end="")
            if early_stop.counter == 0:
                self.save_checkpoint()
            if early_stop.stop_flag:
                print(f"\n  {get_format_time()} Model Training Finished!")
                print(f"  {get_format_time()} Trained checkpoint file has been saved to {self.ckpt_path}")
                break

    @get_running_time
    @torch.no_grad()
    def alignment(self):
        self.load_checkpoint()
        self.model.eval()
        data_list = []
        for idx, dataset in enumerate(self.dataset.loader_list):
            dataset = dataset.to(self.device)
            _, _, _, latent_x, recon_x = self.model(
                x=dataset.x, edge_index=dataset.edge_index, edge_weight=dataset.edge_weight,
                domain_idx=dataset.domain_idx, neigh_mask=dataset.neigh_graph)
            data = AnnData(sp.csr_matrix(recon_x.detach().cpu().numpy()))
            data.obsm["correct"] = latent_x.detach().cpu().numpy()

            data.obs = self.dataset.data_list[idx].obs

            data.obsm["spatial"] = dataset.pos.detach().cpu().numpy()
            data.obs.index = self.dataset.data_list[idx].obs.index
            data.var_names = self.dataset.merge_data.var_names
            data.write_h5ad(osp.join(self.res_path, f"correct_data{idx}.h5ad"))
            data_list.append(data)
        print(f"{get_format_time()} Batch Alignment Finished!")
        print(f"{get_format_time()} Alignment data saved in: {self.res_path}")
