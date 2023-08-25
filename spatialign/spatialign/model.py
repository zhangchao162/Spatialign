#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/12/23 9:50 AM
# @Author  : zhangchao
# @File    : spatialign.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn

from spatialign.module import DGI, EmbeddingLayer, scale_mse


class DGIAlignment(nn.Module):
    def __init__(self, input_dims, output_dims, n_domain, act, p):
        super().__init__()
        self.dgi = DGI(input_dims, output_dims, n_domain, act, p)
        self.decoder = EmbeddingLayer(
            input_dims=output_dims, output_dims=input_dims, n_domain=n_domain, act=act, drop_rate=p)

    def forward(self, x, edge_index, edge_weight, domain_idx, neigh_mask):
        latent_x, neg_x, pos_summary, graph_loss = self.dgi(x, edge_index, edge_weight, domain_idx, neigh_mask)
        dgi_loss = self.dgi.loss(latent_x, neg_x, pos_summary)
        recon_x = self.decoder(latent_x, domain_idx)
        recon_loss = scale_mse(recon_x=recon_x, x=x)
        return graph_loss, dgi_loss, recon_loss, latent_x, recon_x
