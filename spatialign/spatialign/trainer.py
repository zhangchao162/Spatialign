#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/12/23 9:53 AM
# @Author  : zhangchao
# @File    : trainer.py
# @Email   : zhangchao5@genomics.cn
# cython: language_level=3
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

from spatialign.module import contrast_loss, trivial_entropy, cross_instance_loss
from spatialign.utils import Dataset, get_format_time, get_running_time, EarlyStopping
from spatialign.spatialign import DGIAlignment, spatiAlignBase


class Spatialign(spatiAlignBase):
    """
    spatialign Model
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
    :param mask_rate:
        The rate of training size.
    :param is_undirected:
        Whether the constructed spatial neighbor graph is undirected graph, default, True.
    :param latent_dims:
        The number of embedding dimensions, default, 100.
    :param is_verbose:
        Whether the detail information is print, default, True.
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
                 mask_rate: Union[float, list] = .5,
                 is_undirected: bool = True,
                 latent_dims: int = 100,
                 is_verbose: bool = True,
                 seed: int = 42,
                 gpu: Union[int, str, None] = None,
                 save_path: str = None):
        super(Spatialign, self).__init__(gpu=gpu, torch_thread=24, seed=seed, save_path=save_path)
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
                               mask_rate=mask_rate,
                               is_undirected=is_undirected)
        self.latent_dims = latent_dims
        self.model = self.set_model(latent_dims=latent_dims, n_domain=self.dataset.n_domain, is_verbose=is_verbose)
        self.header_bank = self.init_bank()

    def set_model(self, latent_dims, n_domain, is_verbose):
        model = DGIAlignment(input_dims=self.dataset.inner_dims,
                             output_dims=latent_dims,
                             n_domain=n_domain,
                             act=nn.ELU(),
                             p=0.2)
        if is_verbose:
            print(f"{get_format_time()} {model.__class__.__name__}: \n{model}")
        model.to(self.device)
        return model

    @torch.no_grad()
    @get_running_time
    def init_bank(self):
        self.model.eval()
        header_bank = defaultdict()

        for data_idx, loader in enumerate(self.dataset.trainer_list):
            data = next(iter(loader)).to(self.device)
            neigh_graph = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float, device=self.device)
            neigh_graph[data.edge_index[0], data.edge_index[1]] = 1.
            latent_x, neg_x, pos_summary, recon_x = self.model(x=data.x,
                                                               edge_index=data.edge_index,
                                                               edge_weight=data.edge_weight,
                                                               domain_idx=data.domain_idx,
                                                               neigh_mask=neigh_graph)
            header_bank[data_idx] = latent_x[:data.batch_size].detach()
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
            for idx, loader in enumerate(self.dataset.trainer_list):
                data = next(iter(loader)).to(self.device, non_blocking=True)
                neigh_graph = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float, device=self.device)
                neigh_graph[data.edge_index[0], data.edge_index[1]] = 1.
                # with torch.cuda.amp.autocast():
                latent_x, neg_x, pos_summary, recon_x = self.model(x=data.x,
                                                                   edge_index=data.edge_index,
                                                                   edge_weight=data.edge_weight,
                                                                   domain_idx=data.domain_idx,
                                                                   neigh_mask=neigh_graph)

                graph_loss, dgi_loss, recon_loss = self.model.loss(x=data.x[:data.batch_size],
                                                                   recon_x=recon_x[:data.batch_size],
                                                                   latent_x=latent_x[:data.batch_size],
                                                                   neg_x=neg_x[:data.batch_size],
                                                                   pos_summary=pos_summary[:data.batch_size])

                # update memory bank
                self.update_bank(idx, latent_x[:data.batch_size], alpha=alpha)

                intra_inst = contrast_loss(feat1=latent_x[:data.batch_size],
                                           feat2=self.header_bank[idx],
                                           tau=tau1,
                                           weight=1.)
                intra_clst = contrast_loss(feat1=latent_x[:data.batch_size].T,
                                           feat2=self.header_bank[idx][:data.batch_size].T,
                                           tau=tau1,
                                           weight=1.)
                # Maximize clustering entropy to avoid all data clustering into the same class
                entropy_clst = trivial_entropy(feat=latent_x[:data.batch_size], tau=tau2, weight=1.)

                loss = graph_loss + dgi_loss + recon_loss + intra_inst + entropy_clst + intra_clst
                for i in np.delete(range(len(self.dataset.data_list)), idx):
                    inter_loss = cross_instance_loss(
                        feat1=latent_x[:data.batch_size], feat2=self.header_bank[i], tau=tau3, weight=1.)
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
                self.save_checkpoint(model=self.model)
            if early_stop.stop_flag:
                print(f"\n  {get_format_time()} Model Training Finished!")
                print(f"  {get_format_time()} Trained checkpoint file has been saved to {self.ckpt_path}")
                break

    @get_running_time
    @torch.no_grad()
    def alignment(self):
        self.load_checkpoint(model=self.model)
        self.model.eval()
        data_list = []
        for idx, loader in enumerate(self.dataset.tester_list):
            dataset = next(iter(loader)).to(self.device)
            neigh_graph = torch.zeros((dataset.num_nodes, dataset.num_nodes), dtype=torch.float, device=self.device)
            neigh_graph[dataset.edge_index[0], dataset.edge_index[1]] = 1.
            latent_x, neg_x, pos_summary, recon_x = self.model(x=dataset.x,
                                                               edge_index=dataset.edge_index,
                                                               edge_weight=dataset.edge_weight,
                                                               domain_idx=dataset.domain_idx,
                                                               neigh_mask=neigh_graph)
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
