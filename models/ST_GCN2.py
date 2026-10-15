#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y


# ========================
# Step 6: Main GCN Refinement Module (FIXED)
# ========================
# class GCN_main(nn.Module):
#     def __init__(self, cfg):
#         super(GCN_main, self).__init__()
#         self.cfg = cfg
#
#         self.gcn_spatial = GCN(
#             input_feature=42,
#             hidden_feature=128,#后续做消融
#             p_dropout=0.3,
#             num_stage=2,
#             node_n=48,
#         ).to(cfg.device)
#
#         self.gcn_temporal = GCN(
#             input_feature=48,
#             hidden_feature=128,#后续做消融
#             p_dropout=0.3,
#             num_stage=2,
#             node_n=42,
#         ).to(cfg.device)
#
#     def forward(self, src, diffusion_out):
#         """
#         :param src: (B, 20, 48) — initial DCT coefficients (after permute)
#         :param diffusion_out: (B, 20, 48) — predicted trajectory in DCT domain
#         """
#         # First refinement
#         dct_in_tmp = torch.cat([src.permute(0, 2, 1), diffusion_out.permute(0, 2, 1)], dim=-1)  # (B, 48, 40)
#         spatial_gcn_out_1 = self.gcn_spatial(dct_in_tmp)
#         temporal_gcn_out_1 = self.gcn_temporal(spatial_gcn_out_1.permute(0, 2, 1))[:, :21, :] # (B, 20, 48)
#
#         # Second refinement
#         dct_in_tmp = torch.cat([src.permute(0, 2, 1), temporal_gcn_out_1.permute(0, 2, 1)], dim=-1)  # (B, 48, 40)
#         spatial_gcn_out_2 = self.gcn_spatial(dct_in_tmp)
#         temporal_gcn_out_2 = self.gcn_temporal(spatial_gcn_out_2.permute(0, 2, 1))[:, :21, :]# (B, 20, 48)
#         output = temporal_gcn_out_2
#
#         return output
class GCN_main(nn.Module):
    def __init__(self, cfg):
        super(GCN_main, self).__init__()
        self.cfg = cfg

        self.gcn_spatial = GCN(
            input_feature=20,
            hidden_feature=128,#后续做消融
            p_dropout=0.3,
            num_stage=2,
            node_n=48,
        ).to(cfg.device)

        self.gcn_temporal = GCN(
            input_feature=48,
            hidden_feature=128,#后续做消融
            p_dropout=0.3,
            num_stage=2,
            node_n=20,
        ).to(cfg.device)

    def forward(self, x):
        """
        :param src: (B, 20, 48) — initial DCT coefficients (after permute)
        :param diffusion_out: (B, 20, 48) — predicted trajectory in DCT domain
        """
        # First refinement
        spatial_gcn_out_1 = self.gcn_spatial(x.permute(0, 2, 1))
        temporal_gcn_out_1 = self.gcn_temporal(spatial_gcn_out_1.permute(0, 2, 1)) # (B, 20, 48)

        # Second refinement
        spatial_gcn_out_2 = self.gcn_spatial(temporal_gcn_out_1.permute(0, 2, 1))
        temporal_gcn_out_2 = self.gcn_temporal(spatial_gcn_out_2.permute(0, 2, 1))# (B, 20, 48)

        return temporal_gcn_out_2