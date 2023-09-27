#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/23 6:02 PM
# @Author  : zhangchao
# @File    : graph_vae.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import VGAE, SAGEConv
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling


class GraphEncoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(GraphEncoder, self).__init__()
        self.sage = SAGEConv(input_dims, output_dims)
        self.mu = SAGEConv(output_dims, output_dims)
        self.var = SAGEConv(output_dims, output_dims)

    def forward(self, x, edge_index, edge_weight):
        feat_x = self.sage(x, edge_index).relu_()
        mu = self.mu(feat_x, edge_index)
        var = self.var(feat_x, edge_index)
        return mu, var


class GraphVAE(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(GraphVAE, self).__init__()
        self.edge_index = None
        self.edge_weight = None
        self.feat_x = None
        self.vgae = VGAE(GraphEncoder(input_dims, output_dims))

    def forward(self, x, edge_index, edge_weight):
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.feat_x = self.vgae.encode(x, edge_index, edge_weight)
        return self.feat_x
