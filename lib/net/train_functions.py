import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
from collections import namedtuple
from tools.PCLosses import ChamferLoss
from tools.utils import getScoreIoU, getScoreHingeIoU
import numpy as np

def model_joint_fn_decorator():
    ModelReturn = namedtuple("ModelReturn", ['loss', 'tb_dict', 'disp_dict'])
    MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()

    def model_fn(model, data, lambda_completion, ckpt_dir):

        # pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
        # gt_boxes3d = data['gt_boxes3d']

        # if not cfg.RPN.FIXED:
        #     rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
        #     rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).long()
        #     rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking=True).float()
        #     inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        #     gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking=True).float()
        #     input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d}
        # else:
        #     model_PC = data['model_PC']
        #     inputs = pts_input.cuda(non_blocking=True).float()
        #     gt_boxes3d = gt_boxes3d.cuda(non_blocking=True).float()
        #     model_PC = model_PC.cuda(non_blocking=True).float()
        #     input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d, 'model_PC': model_PC}

        if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_reg_label']
            rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).long()
            rpn_reg_label = torch.from_numpy(rpn_reg_label).cuda(non_blocking=True).float()
            inputs = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking=True).float()
            input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d, 'sample_id': data['sample_id']}
        elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            sample_PC, model_PC, score = data['sample_PC'], data['model_PC'], data['score']
            sample_PC = sample_PC.view(-1, 3, 2048).cuda()
            model_PC = model_PC.view(-1, 3, 2048).cuda()
            target = score.float().cuda(non_blocking=True).view(-1)
            input_data = {'sample_PC': sample_PC, 'model_PC': model_PC, 'target': target}
        elif cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            pts_rect, pts_features, pts_input = data['pts_rect'], data['pts_features'], data['pts_input']
            gt_boxes3d = data['gt_boxes3d']
            model_PC = data['model_PC']
            inputs = pts_input.cuda(non_blocking=True).float()
            gt_boxes3d = gt_boxes3d.cuda(non_blocking=True).float()
            model_PC = model_PC.cuda(non_blocking=True).float()
            input_data = {'pts_input': inputs, 'gt_boxes3d': gt_boxes3d, 'model_PC': model_PC}
        else:
            raise ModuleNotFoundError

        ret_dict = model(input_data)

        tb_dict = {}
        disp_dict = {}
        loss = 0

        if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED and not cfg.RPN.FIXED:
            rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
            rpn_loss = get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict)
            loss += rpn_loss
            disp_dict['rpn_loss'] = rpn_loss.item()

            #describe the 3D box
            import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
            sample_id = ret_dict['sample_id']
            pts_input = ret_dict['pts_input'].cpu()
            roi_boxes3d = ret_dict['roi_boxes3d'].cpu()
            # for i in range(roi_boxes3d.shape[0]):
            #     savepath = os.path.join('/media/fengzicai/fzc/3Dsiamesetracker/show4/', sample_id[i])
            #     os.makedirs(savepath, exist_ok=True)
            #     np.savetxt(savepath + '.txt', pts_input[i, :, :].numpy(), fmt='%.2f,%.2f,%.2f')

                # boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(pts_input[i, :, :], roi_boxes3d[i, :, :])
                #
                # for k in range(roi_boxes3d.shape[1]):
                #     pt_mask_flag = (boxes_pts_mask_list[k].numpy() == 1)
                #     cur_pts = pts_input[i, pt_mask_flag, :].float().numpy()
                #     np.savetxt(os.path.join(savepath, str(k) + '.txt'), cur_pts,
                #                fmt='%.2f,%.2f,%.2f')

        if not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
            output, Y_AE, target, model_PC = ret_dict['Sim'], ret_dict['Y_AE'], ret_dict['target'], ret_dict['model_PC']
            criterion_tracking = torch.nn.MSELoss()
            criterion_completion = ChamferLoss()
            if lambda_completion < 1:
                loss1 = criterion_tracking(output, target)
            else:
                loss1 = torch.tensor([0]).float().cuda()
            if lambda_completion != 0:
                loss2 = criterion_completion(Y_AE, model_PC)
            else:
                loss2 = torch.tensor([0]).float().cuda()
            siamese_loss = loss1 + lambda_completion * loss2
            disp_dict['loss_tracking'] = loss1.item()
            disp_dict['loss_completion'] = loss2.item()
            loss += siamese_loss
            disp_dict['siamese_loss'] = siamese_loss.item()



        # if cfg.RCNN.ENABLED and cfg.RCNN.ENABLED and cfg.RPN.FIXED:
        #     loss_tracking, loss_completion, siamese_loss = get_siamese_loss(model, ret_dict, tb_dict, lambda_completion, regress)
        #     disp_dict['loss_tracking'] = loss_tracking.item()
        #     disp_dict['loss_completion'] = loss_completion.item()
        #     loss += siamese_loss
        #     disp_dict['siamese_loss'] = siamese_loss.item()

        disp_dict['loss'] = loss.item()

        return ModelReturn(loss, tb_dict, disp_dict)

    def get_rpn_loss(model, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
        if isinstance(model, nn.DataParallel):
            rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        else:
            rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
            rpn_loss_cls = rpn_cls_loss_func(rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rpn_cls_flat), rpn_cls_label_target,
                                                    weight=weight, reduction='none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                loss_utils.get_reg_loss(rpn_reg.view(point_num, -1)[fg_mask],
                                        rpn_reg_label.view(point_num, 7)[fg_mask],
                                        loc_scope=cfg.RPN.LOC_SCOPE,
                                        loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                        num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                        anchor_size=MEAN_SIZE,
                                        get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                        get_y_by_bin=False,
                                        get_ry_fine=False)

            loss_size = 3 * loss_size  # consistent with old codes
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({'rpn_loss_cls': rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                        'rpn_loss': rpn_loss.item(), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(),
                        'rpn_loss_angle': loss_angle.item(), 'rpn_loss_size': loss_size.item()})

        return rpn_loss

    def get_siamese_loss(model, ret_dict, tb_dict, lambda_completion, regress):
        # {'Sim': Sim, 'Y_AE': Y_AE, 'gt_boxes3d': gt_boxes3d, 'model_PC': model_PC}
        Sim = ret_dict['Sim']
        Y_AE = ret_dict['Y_AE']
        model_PC = ret_dict['model_PC']
        '''
        gt_boxes3d batch * 1 * 7
        roi_boxes3d batch2 * m 64 * 7
        '''
        gt_boxes3d = ret_dict['gt_boxes3d']
        roi_boxes3d = ret_dict['roi_boxes3d']
        batch_size = roi_boxes3d.shape[0]
        candidate_num = roi_boxes3d.shape[1]
        criterion_tracking = torch.nn.MSELoss()
        criterion_completion = ChamferLoss()
        target = torch.zeros([batch_size*candidate_num, ]).cuda().float()
        if "IOU" in regress.upper():
            for k in range(batch_size):  #
                for c in range(candidate_num):
                    target[k * candidate_num + c] = getScoreIoU(roi_boxes3d[k, c, :], gt_boxes3d[k, 0, :])
        elif "HINGE" in regress.upper():
            for k in range(batch_size):
                for c in range(candidate_num):
                    target[k * candidate_num + c] = getScoreHingeIoU(roi_boxes3d[k, c, :], gt_boxes3d[k, 0, :])
        else:
            raise NotImplementedError

        if lambda_completion < 1:
            loss1 = criterion_tracking(Sim, target)
        else:
            loss1 = torch.tensor([0]).float().cuda()

        if lambda_completion != 0:
            loss2 = criterion_completion(Y_AE, model_PC)
        else:
            loss2 = torch.tensor([0]).float().cuda()

        loss = loss1 + lambda_completion * loss2

        return loss1, loss2, loss

    return model_fn

