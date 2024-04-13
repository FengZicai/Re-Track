'''
Created on February 4, 2017

@author: optas

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from pyquaternion import Quaternion
import copy
from tools.AEModel import PCAutoEncoder, DPAutoEncoder
from tools.data_classes import PointCloud
from lib.rpn.proposal_target_layer import ProposalTargetLayer

from pyquaternion import Quaternion
from lib.config import cfg
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
import lib.utils.iou3d.iou3d_utils as iou3d_utils
import tools.utils as utils


class Model(nn.Module):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, bneck_size=128, mode='TRAIN'):
        super(Model, self).__init__()
        self.AE = PCAutoEncoder(bneck_size)
        self.input_size = self.AE.input_size
        self.mode = mode
        self.bneck_size = bneck_size
        self.score = nn.Linear(bneck_size * 2, 1)

        self.offset_BB = 0
        self.scale_BB = 1.25
        self.model_inputsize = 2048
        self.proposal_target_layer = ProposalTargetLayer()

    def forward(self, siamese_input_info):
        # roi_boxes3d = siamese_input_info['roi_boxes3d']
        # gt_boxes3d = siamese_input_info['gt_boxes3d']
        # pts_input = siamese_input_info['pts_input']
        # if self.mode == 'TRAIN':
        #     batch_rois, batch_gt_of_rois, batch_roi_iou = self.proposal_target_layer.sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d) #测试发现使用roi_sample的效果更好
        #     sample_PCs = self.cropAndCenterPC(pts_input, batch_rois, offset=self.offset_BB, scale=self.scale_BB)
        # elif self.mode == "TEST":
        #     sample_PCs = self.cropAndCenterPC(pts_input, roi_boxes3d, offset=self.offset_BB, scale=self.scale_BB)
        # else:
        #     raise ModuleNotFoundError

        if not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            sample_PC, model_PC, target = siamese_input_info['sample_PC'], siamese_input_info['model_PC'], siamese_input_info['target']
            X = self.AE.encode(sample_PC)
            Y = self.AE.encode(model_PC)
            Y_AE = self.AE.forward(model_PC)
            Sim = F.cosine_similarity(X, Y, dim=1)
            return {'Sim': Sim, 'Y_AE': Y_AE, 'target': target, 'model_PC': model_PC}
        # elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED and self.mode == "TEST":
        #     roi_boxes3d = siamese_input_info['roi_boxes3d']
        #     gt_boxes3d = siamese_input_info['gt_boxes3d']
        #     pts_input = siamese_input_info['pts_input']
        #     batch_rois, batch_gt_of_rois, batch_roi_iou = self.proposal_target_layer.sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d)  # 测试发现使用roi_sample的效果更好
        #     sample_PCs = self.cropAndCenterPC(pts_input, batch_rois, offset=self.offset_BB, scale=self.scale_BB)
        #
        #     model_PCs = siamese_input_info['model_PC']
        #     '''
        #     已知sample_PCs与model_PCs,将矩阵展开为sample_PC与model_PC,2019年10月10日20:06:35
        #     '''
        #     sample_PC = sample_PCs.view(-1, 3, 2048)
        #     # for i in range(sample_PC.shape[0]):
        #     #     np.savetxt('/media/fengzicai/fzc/3Dsiamesetracker/show3/' + str(i) + '.txt', sample_PC[i].cpu().numpy().T, fmt='%.2f,%.2f,%.2f')
        #
        #     repeat_shape = np.ones(len(sample_PC.shape) + 1, dtype=np.int32)
        #     repeat_shape[1] = sample_PCs.shape[1]
        #     model_PC = model_PCs.unsqueeze(1).repeat(tuple(repeat_shape)).view(-1, 3, 2048)
        #     X = self.AE.encode(sample_PC)
        #     Y = self.AE.encode(model_PC)
        #     Y_AE = self.AE.forward(model_PC)
        #     Sim = F.cosine_similarity(X, Y, dim=1)
        #     # X = self.score(torch.cat((X,Y),dim=1)).squeeze()
        #
        #     return {'Sim': Sim, 'Y_AE': Y_AE, 'gt_boxes3d': gt_boxes3d, 'model_PC': model_PC, 'roi_boxes3d': batch_rois}
        #     #     return {'Sim': Sim, 'Y_AE': Y_AE, 'gt_boxes3d': gt_boxes3d, 'model_PC': model_PC, 'roi_boxes3d': roi_boxes3d}
        else:
            raise ModuleNotFoundError

    def encode(self, X):
        return self.AE.encode(X)

    def decode(self, X):
        return self.AE.decode(X)

    def corners(self, center, rois_tmp_hwl, rotation_matrix, wlh_factor: float = 1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        rois_tmp_hwl = rois_tmp_hwl * wlh_factor
        h, w, l = rois_tmp_hwl
        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        corners = torch.tensor(np.array([l.cpu().numpy() / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1]),
                                         w.cpu().numpy() / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1]),
                                         h.cpu().numpy() / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])])).cuda().float()
        # Rotate
        corners = torch.mm(rotation_matrix, corners)
        # Translate
        corners = center.float() + corners.float()
        return corners

    def cropPC(self, pts_tmp_input, center, rois_tmp_hwl, rotation_matrix, offset=0, scale=1.0):
        rois_tmp_hwl = rois_tmp_hwl * scale
        corners = self.corners(center, rois_tmp_hwl, rotation_matrix)

        maxi = (torch.max(corners, 1).values.cuda() + torch.tensor(np.array([offset, offset, offset])).float().cuda()).reshape((3, 1))
        mini = (torch.min(corners, 1).values.cuda() - torch.tensor(np.array([offset, offset, offset])).float().cuda()).reshape((3, 1))
        x_filt_max = pts_tmp_input[0, :] < maxi[0]
        x_filt_min = pts_tmp_input[0, :] > mini[0]
        y_filt_max = pts_tmp_input[1, :] < maxi[1]
        y_filt_min = pts_tmp_input[1, :] > mini[1]
        z_filt_max = pts_tmp_input[2, :] < maxi[2]
        z_filt_min = pts_tmp_input[2, :] > mini[2]

        close = x_filt_min & x_filt_max
        close = close & y_filt_min
        close = close & y_filt_max
        close = close & z_filt_min
        close = close & z_filt_max
        new_PC = pts_tmp_input[:, close]
        return new_PC

    def cropAndCenterPC(self, pts_input, batch_rois, offset=0, scale=1.0, normalize=False):
        batch_size = batch_rois.shape[0]
        candidate_num = batch_rois.shape[1]
        new_PCs = torch.zeros((batch_size, candidate_num, 3, self.model_inputsize))
        for k in range(batch_size):
            for c in range(candidate_num):
                pts_tmp_input = pts_input[k, :, :].t()
                rois_tmp_xyz = batch_rois[k, c, 0:3]
                rois_tmp_hwl = batch_rois[k, c, 3:6]
                rois_tmp_ry = batch_rois[k, c, -1]

                center = torch.tensor([rois_tmp_xyz[0], rois_tmp_xyz[1] - rois_tmp_hwl[0] / 2, rois_tmp_xyz[2]]).reshape((3, 1)).cuda().float()
                rotation_matrix = torch.tensor((Quaternion(axis=[0, 1, 0], radians=float(rois_tmp_ry)) *
                                               Quaternion(axis=[1, 0, 0], radians=np.pi / 2)).rotation_matrix).cuda().float()
                new_PC = self.cropPC(pts_tmp_input, center, rois_tmp_hwl, rotation_matrix, offset=2 * offset,
                                     scale=4 * scale)

                new_center = copy.deepcopy(center)
                new_rotation_matrix = copy.deepcopy(rotation_matrix)
                rot_mat = new_rotation_matrix.t()
                trans = -new_center

                # align data
                new_PC = trans + new_PC
                new_center = new_center + trans
                if new_PC.shape[1] == 0:
                    new_PCs[k, c, :, :] = torch.zeros((3, self.model_inputsize)).cuda().float()
                    continue
                new_PC = torch.mm(rot_mat, new_PC)
                new_center = torch.mm(rot_mat, new_center)
                new_rotation_matrix = torch.mm(rot_mat, new_rotation_matrix)

                # crop around box
                new_PC = self.cropPC(new_PC, new_center, rois_tmp_hwl, new_rotation_matrix, offset=offset, scale=scale)

                if normalize:
                    new_PC = self.normalize(new_PC, rois_tmp_hwl)
                new_PC = self.regularizePC(new_PC, self.model_inputsize)[0]
                new_PCs[k, c, :, :] = new_PC
        return new_PCs.cuda().float()

    def regularizePC(self, new_PC, model_input_size):
        if new_PC.shape[1] > 20:
            if new_PC.shape[0] > 3:
                new_PC = new_PC[0:3, :]
            if new_PC.shape[1] != model_input_size:
                new_pts_idx = torch.randint(low=0, high=new_PC.shape[1], size=(model_input_size,))
                new_PC = new_PC[:, new_pts_idx]
            new_PC = new_PC.reshape((3, model_input_size))
        else:
            new_PC = torch.zeros((3, model_input_size))
        return new_PC.float()

    def normalize(self, new_PC, rois_tmp_hwl):
        h, w, l = rois_tmp_hwl
        normalizer = torch.tensor(np.array([l.numpy(), w.numpy(), h.numpy()])).reshape((3, -1))
        return new_PC / normalizer


class DenseModel(nn.Module):
    def __init__(self, bneck_size=128, mode='TRAIN'):
        super(DenseModel, self).__init__()
        self.AE = DPAutoEncoder(bneck_size)
        self.input_size = self.AE.input_size
        self.mode = mode
        self.bneck_size = bneck_size
        # self.score = nn.Linear(1280, 1) four
        self.score = nn.Linear(256, 1)

    def forward(self, siamese_input_info):
        if not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            sample_PC, model_PC, target = siamese_input_info['sample_PC'], siamese_input_info['model_PC'], siamese_input_info['target']

            X = self.AE.encode(sample_PC)
            Y = self.AE.encode(model_PC)
            Y_AE = self.AE.forward(model_PC)
            Sim = F.cosine_similarity(X, Y, dim=1)
            # Sim = self.score(torch.cat((X, Y), dim=1)).squeeze()
            return {'Sim': Sim, 'Y_AE': Y_AE, 'target': target, 'model_PC': model_PC}
        else:
            raise ModuleNotFoundError

    def encode(self, X):
        return self.AE.encode(X)

