#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-03-01 22:49:23
# @Author  : Tuo Feng (fengt@stu.xidian.edu.cn)
# @Link    : https://blog.csdn.net/taifengzikai/
# @Version : $1.0$

import os
import sys
import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'shapemeasure'))

from pointnet2_utils import *
from pytorch_utils import *

from typing import List, Tuple

class point_rnn(nn.Module):
    def __init__(self, radius, nsample, feat_channels, out_channels, knn=False, pooling='max'):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling
        #a=self.feat_channels+self.out_channels+3                                                   #nn.ReLU(inplace=False)
        self.fc = Conv2d((self.feat_channels+self.out_channels+3), self.out_channels,kernel_size=1, stride=1,activation=None,
                         init=nn.init.xavier_uniform_)

    def forward(self,P1, P2, X1, S2):
        """
        Input:
            P1:     (batch_size, npoint, 3)
            P2:     (batch_size, npoint, 3)
            X1:     (batch_size, npoint, feat_channels)
            S2:     (batch_size, npoint, out_channels)
        Output:
            S1:     (batch_size, npoint, out_channels)
        """
        # 1. Sample points
        if self.knn:
            _, idx = knn_point(self.nsample, P2, P1)
        else:
            idx, cnt = ball_query(self.radius, self.nsample, P2, P1)
            _, idx_knn = knn_point(self.nsample, P2, P1)
            cnt = cnt.unsqueeze(-1).repeat([1, 1, self.nsample])
            idx = torch.where(cnt > (self.nsample-1), idx, idx_knn)

        # 2.1 Group P2 points
        P2_trans = P2.contiguous()
        P2_grouped = grouping_operation(P2_trans, idx).contiguous()# (B, 3, npoint, nsample)

        # 2.2 Group P2 states
        S2_trans = S2.contiguous()
        S2_grouped = grouping_operation(S2_trans, idx).contiguous()  # (B, out_channels, npoint, nsample)

        # 3. Calcaulate displacements

        P1_expanded = P1.unsqueeze(2) #
        displacement = P2_grouped - P1_expanded                 # batch_size, 3, npoint, nsample

        # 4. Concatenate X1, S2 and displacement
        if X1 is not None:
            X1_expanded = X1.unsqueeze(2).repeat([1, 1, self.nsample,1 ])                # batch_size, feat_channels, npoint, sample,
            correlation = torch.cat((S2_grouped, X1_expanded), 3)                      # batch_size,  feat_channels+out_channels, npoint, nsample
            correlation = torch.cat((correlation, displacement), 3)                    # batch_size, feat_channels+out_channels+3, npoint, nsample
        else:
            correlation = torch.cat((S2_grouped, displacement), 3) #4,512,24,131               # batch_size, out_channels+3, npoint, nsample

        # 5. Fully-connected layer (the only parameters)
        S1 = self.fc(correlation.transpose(2, 3).contiguous().transpose(1, 2).contiguous())

        S1 = S1.transpose(1, 2).contiguous().transpose(2, 3).contiguous()
        # 6. Pooling
        if self.pooling=='max':
            return torch.max(S1, 2, False)
        elif self.pooling=='avg':
            return torch.mean(S1, 2, False)

class PointRNNCell(nn.Module):
    def __init__(self,
                 radius,
                 nsample,
                 feat_channels,
                 out_channels,
                 knn=False,
                 pooling='max'):
        super().__init__()

        self.radius = radius
        self.nsample = nsample
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling
        self.point_rnn_cell = point_rnn(radius=self.radius, nsample=self.nsample,feat_channels=self.feat_channels, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
    
    def init_state(self, inputs):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, X = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.shape[0]
        inferred_npoints = P.shape[1]
        inferred_xyz_dimensions = P.shape[2]

        P = torch.zeros([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        S = torch.zeros([inferred_batch_size, inferred_npoints, self.out_channels], dtype=torch.float32)

        return (P, S)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, S2 = states

        S1 = self.point_rnn_cell(P1, P2, X1, S2)

        return (P1, S1)

class PointGRUCell(nn.Module):
    def __init__(self,
                 radius,
                 nsample,
                 feat_channels,
                 out_channels,
                 knn=False,
                 pooling='max'):
        super().__init__(radius, nsample,feat_channels, out_channels, knn, pooling)
        self.point_gru_cell1 = point_rnn(radius=self.radius, nsample=self.nsample, feat_channels=self.feat_channels, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
        self.point_gru_cell2 = point_rnn(radius=self.radius, nsample=self.nsample, feat_channels=self.feat_channels, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
        self.point_gru_cell3 = point_rnn(radius=self.radius, nsample=self.nsample, feat_channels=self.feat_channels, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
        self.GRUFC = Conv1d(in_size , out_size=self.out_channels, activation=None, name='new_state') #in_size=？

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, S2 = states

        Z = self.point_gru_cell1(P1, P2, X1, S2)
        R = self.point_gru_cell2(P1, P2, X1, S2)
        Z = torch.sigmoid(Z)
        R = torch.sigmoid(R)

        S_old = self.point_gru_cell3(P1, P2, None, S2)

        if X1 is None:
            S_new = R*S_old
        else:
            S_new = torch.cat((X1, R*S_old), 2)

        S_new = self.GRUFC(S_new)
        S_new = torch.tanh(S_new)

        S1 = Z * S_old + (1 - Z) * S_new

        return (P1, S1)

class PointLSTMCell(nn.Module):
    def __init__(self,
                 radius,
                 nsample,
                 feat_channels,
                 out_channels,
                 knn=False,
                 pooling='max'):

        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling

        self.input_gate = point_rnn(radius=self.radius, nsample=self.nsample, feat_channels=self.feat_channels, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
        self.forget_gate = point_rnn(radius=self.radius, nsample=self.nsample, feat_channels=self.feat_channels, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
        self.output_gate = point_rnn(radius=self.radius, nsample=self.nsample, feat_channels=self.feat_channels, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
        self.new_cell = point_rnn(radius=self.radius, nsample=self.nsample, feat_channels=self.feat_channels, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)
        self.old_cell = point_rnn(radius=self.radius, nsample=self.nsample, feat_channels=0, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)

    def init_state(self, inputs):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, X = inputs

        inferred_batch_size = P.shape[0]
        inferred_npoints = P.shape[1]
        inferred_xyz_dimensions = P.shape[2]

        P = torch.zeros([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype).cuda()
        H = torch.zeros([inferred_batch_size, inferred_npoints, self.out_channels], dtype=torch.float32).cuda()
        C= torch.zeros([inferred_batch_size, inferred_npoints, self.out_channels], dtype=torch.float32).cuda()

        return (P, H, C)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, H2, C2 = states

        I = self.input_gate(P1, P2, X1, H2)
        F = self.forget_gate(P1, P2, X1, H2)
        O = self.output_gate(P1, P2, X1, H2)

        C_new = self.new_cell(P1, P2, X1, H2)
        C_old = self.old_cell(P1, P2, None, C2)

        I = torch.sigmoid(I.values)
        F = torch.sigmoid(F.values)
        O = torch.sigmoid(O.values)
        C_new = torch.tanh(C_new.values)

        C1 = F * (C_old.values) + I * C_new
        H1 = O * torch.tanh(C1)

        return (P1, H1, C1)
