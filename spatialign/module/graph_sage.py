#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/21 16:15
# @Author  : zhangchao
# @File    : graph_sage.py
# @Software: PyCharm
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class SAGELayer(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SAGELayer, self).__init__()
        self.layer = SAGEConv(input_dims, output_dims)
        self.drop = nn.Dropout()

    def forward(self, x, edge_index):
        x = self.layer(x, edge_index).relu_()
        x = self.drop(x)
        return x


class SAGE(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SAGE, self).__init__()
        self.layer1 = SAGELayer(input_dims, output_dims)
        self.layer2 = SAGELayer(output_dims, output_dims)
        self.layer3 = SAGELayer(output_dims, output_dims)

    def forward(self, x, edge_index, edge_weight):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        return x

