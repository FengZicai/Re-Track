"""
PointNet++ Operations and Layers
Modified by LXX
Date February  2020
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'shapemeasure'))

from pointnet2_utils import *
import pytorch_utils as pt_utils

class SampleAndGroup(nn.Module):
    def __init__(self, nsample , knn, use_xyz):
        super().__init__()

        self.nsample = nsample
        self.knn = False
        self.use_xyz = False
    def forward(self, npoint, radius, xyz, points):
        '''
        Input:
            npoint:         int32
            radius:         float32
            nsample:        int32
            xyz:            (batch_size, ndataset, 3) TF tensor
            points:         (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
            knn:            bool, if True use kNN instead of radius search
            use_xyz:        bool, if True concat XYZ with local point features, otherwise just use point features
        Output:
            new_xyz:        (batch_size, npoint, 3) TF tensor
            new_points:     (batch_size, npoint, nsample, 3+channel) TF tensor
            idx:            (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
            grouped_xyz:    (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs (subtracted by seed point XYZ) in local regions
        '''
        xyz = xyz.contiguous()
        new_xyz = gather_operation(xyz.contiguous(), furthest_point_sample(xyz, npoint)).contiguous()
        # new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))         # (batch_size, npoint, 3)
        if self.knn:  # no go
            _,idx = knn_point(nsample, xyz, new_xyz)
        else:  # go
            idx, pts_cnt = ball_query(radius, self.nsample, xyz, new_xyz)

        # grouped_xyz = group_point(xyz, idx)                                     # (batch_size, npoint, nsample, 3) (B, C, npoint, nsample)
        grouped_xyz = grouping_operation(xyz.contiguous(), idx).contiguous()
        grouped_xyz -= new_xyz.unsqueeze(2).repeat(1, 1, self.nsample, 1)  # ??????????

        if points is not None:  # no go -> go
            grouped_points = grouping_operation(points.contiguous(), idx).contiguous() # (batch_size, npoint, nsample, channel)
            if self.use_xyz:  # no go
                new_points = torch.concat([grouped_xyz, grouped_points],
                                          dim=-1)  # (batch_size, npoint, nample, 3+channel)
            else:  # go
                new_points = grouped_points
        else:  # go
            new_points = grouped_xyz

        return new_xyz, new_points, idx, grouped_xyz


class PointnetFPModule(nn.Module):
    def __init__(self, mlp=[256], last_mlp_activation=True):
        super().__init__()
        self.last_mlp_activation = last_mlp_activation
        self.mlp = pt_utils.SharedMLP(mlp, bn=False,activation=nn.ReLU(inplace=False))

    def forward(self,xyz1, xyz2, points1,points2):

        ''' PointNet Feature Propogation (FP) Module
            Input:
                xyz1:       (batch_size, ndataset1, 3) TF tensor
                xyz2:       (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
                points1:    (batch_size, ndataset1, nchannel1) TF tensor
                points2:    (batch_size, ndataset2, nchannel2) TF tensor
                mlp:        list of int32 -- output size for MLP on each point
            Return:
                new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
        '''

        dist, idx = three_nn(xyz1, xyz2)
        dist = torch.max(dist, torch.tensor([1e-8]).cuda())
        norm = torch.sum((1.0/dist),2, True)
        norm = norm.repeat([1,1,3])
        weight = (1.0/dist) / norm #????      #保留问题 pintrnn

        interpolated_points = three_interpolate(points2.transpose(1, 2).contiguous(), idx, weight).transpose(1, 2).contiguous()
#4,256,52
        if points1 is not None:
            #new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
            new_points1 = torch.cat([interpolated_points, points1], dim=2)
        else:
            new_points1 = interpolated_points
        #new_points1 = tf.expand_dims(new_points1, 2)
        new_points1 = new_points1.unsqueeze(2)
        new_points1=self.mlp(new_points1.transpose(2, 3).contiguous().transpose(1, 2).contiguous())
        new_points1=new_points1.transpose(1, 2).contiguous().transpose(2, 3).contiguous()
        '''
        for i, num_out_channel in enumerate(self.mlp):
            if i == len(self.mlp)-1 and not(self.last_mlp_activation):
                activation_fn = None
            else:
                activation_fn = 1#tf.nn.relu#wuyiyi
            con = torch.nn.Conv2d(new_points1.shape[-1],num_out_channel,kernel_size=1,stride=1)
            new_points1 = con(new_points1)
            #new_points1 = conv2d(inputs=new_points1, filters=num_out_channel, name='mlp_%d'%(i+1))

        '''
        new_points1 = new_points1.squeeze(2)                                  # B,ndataset1,mlp[-1]
        return new_points1
