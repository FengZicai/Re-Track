import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.config import cfg
from tools.SiameseModel import Model, DenseModel
from lib.rpn.proposal_target_layer import ProposalTargetLayer


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN', bneck_size=None, DenseAutoEncoder=False):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)
        elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            if DenseAutoEncoder:
                self.model = DenseModel(bneck_size, mode=mode)
            else:
                self.model = Model(bneck_size, mode=mode)
        elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)
            if DenseAutoEncoder:
                self.model = DenseModel(bneck_size, mode=mode)
            else:
                self.model = Model(bneck_size, mode=mode)
        else:
            raise ModuleNotFoundError

        self.proposal_target_layer = ProposalTargetLayer()


    def forward(self, input_data):
        output = {}
        if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED:
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)

            with torch.no_grad():  # 里面的数据不需要计算梯度
                rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                rpn_scores_raw = rpn_cls[:, :, 0]
                rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                pts_depth = torch.norm(backbone_xyz, p=2, dim=2)
                # proposal layer
                rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

            gt_boxes3d = input_data['gt_boxes3d']
            batch_rois, batch_gt_of_rois, batch_roi_iou = self.proposal_target_layer.sample_rois_for_rcnn(
                rois, gt_boxes3d)

            output['sample_id'] = input_data['sample_id']
            output['pts_input'] = input_data['pts_input']
            output['roi_boxes3d'] = batch_rois
            # output['roi_scores_raw'] = roi_scores_raw
            # output['seg_result'] = seg_mask

        elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            # rcnn inference
            model_output = self.model(input_data)
            output.update(model_output)
        elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)

            with torch.no_grad():  # 里面的数据不需要计算梯度
                rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                rpn_scores_raw = rpn_cls[:, :, 0]
                rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                # proposal layer
                rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
                # output['roi_boxes3d'] = rois
                # output['roi_scores_raw'] = roi_scores_raw
                # output['seg_result'] = seg_mask

            rcnn_input_info = {'roi_boxes3d': rois}  #'rpn_xyz': backbone_xyz, 'rpn_features': backbone_features.permute((0, 2, 1)), 'seg_mask': seg_mask, , 'pts_depth': pts_depth}
            rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']
            rcnn_input_info['model_PC'] = input_data['model_PC']
            rcnn_input_info['pts_input'] = input_data['pts_input']

            model_output = self.model(rcnn_input_info)
            output.update(model_output)
        else:
            raise NotImplementedError

        return output
