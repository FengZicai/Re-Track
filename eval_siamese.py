import tools._init_path
import os
import time
import logging
from datetime import datetime
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.cluster import KMeans
import copy

import tools.utils as utils
from tools.PCLosses import ChamferLoss

from tools.metrics import AverageMeter, Success, Precision, Accuracy_Completeness
from tools.metrics import estimateOverlapwithboxes3d, estimateAccuracywithboxes3d
from tools.metrics import estimateOverlap, estimateAccuracy
from pyquaternion import Quaternion
from tools.data_classes import PointCloud, Box

import numpy as np
import re
import lib.utils.kitti_utils as kitti_utils
from lib.net.point_rcnn import PointRCNN
import lib.utils.calibration as calibration
import lib.net.train_functions as train_functions
from lib.config import cfg, cfg_from_file, save_config_to_file, cfg_from_list
import tools.train_utils.train_utils as train_utils
#from tools.rpn_train_dataset import Train_RPN
from tools.searchspace import KalmanFiltering
from tools.ModelUpdate import ModelUpdate
from tools.train_utils.fastai_optim import OptimWrapper
from tools.train_utils import learning_schedules_fastai as lsf



def create_logger(log_file):
    log_format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(level=logging.NOTSET, format=log_format)
    file = logging.FileHandler(log_file)
    file.setLevel(logging.DEBUG)
    logging.getLogger(__name__).addHandler(file)
    return logging.getLogger(__name__)


def load_part_ckpt(model, filename, logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        if '.tar' in filename:
            model_state = checkpoint['state_dict']
            update_model_state = {'model.' + key: val for key, val in model_state.items() if 'model.' + key in model.state_dict()}
        else:
            model_state = checkpoint['model_state']
            update_model_state = {key: val for key, val in model_state.items() if key in model.state_dict()}
        state_dict = model.state_dict()
        state_dict.update(update_model_state)
        model.load_state_dict(state_dict)

        update_keys = update_model_state.keys().__len__()
        if update_keys == 0:
            raise RuntimeError
        logger.info("==> Done (loaded %d/%d)" % (update_keys, total_keys))
    else:
        raise FileNotFoundError


def load_ckpt_based_on_args(model, logger):
    if args.chkpt_file is not None:
        train_utils.load_checkpoint(model, filename=args.chkpt_file, logger=logger)
    total_keys = model.state_dict().keys().__len__()
    if cfg.RPN.ENABLED and args.rpn_ckpt is not None:
        load_part_ckpt(model, filename=args.rpn_ckpt, logger=logger, total_keys=total_keys)
    if cfg.RCNN.ENABLED and args.siam_ckpt is not None:
        load_part_ckpt(model, filename=args.siam_ckpt, logger=logger, total_keys=total_keys)


def eval_one_epoch_joint_offline(model, dataloader, result_dir, logger, epoch =-1,  shape_aggregation="all",
                                 model_fusion="pointcloud", IoU_Space=3, DetailedMetrics=False, max_iter=-1):
    # for reproducibility
    torch.manual_seed(12)
    np.random.seed(666)

    #S3D
    batch_time = AverageMeter()
    data_time = AverageMeter()

    Success_main_all = Success()
    Precision_main_all = Precision()

    Precision_occluded = [Precision(), Precision()]
    Success_occluded = [Success(), Success()]

    Precision_dynamic = [Precision(), Precision()]
    Success_dynamic = [Success(), Success()]

    #PR
    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)
    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()
    end = time.time()

    index = 0
    dataset = dataloader.dataset
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataset.list_of_tracklet_anno), leave=True, desc='eval', ncols=220)
    for datas in dataloader:

        # measure data loading time
        data_time.update((time.time() - end))
        for data in datas:

            Success_main = Success()
            Precision_main = Precision()

            # S3D
            results_BBs = []
            results_scores = []

            list_of_sample_id, list_of_anno, list_of_Sample_PC, list_of_Sample_BB = \
                 data['sample_id'], data['tracklet_anno'], data['list_of_Sample_PC'], data['list_of_Sample_BB']

            BBs = data['list_of_BB']
            PCs = data['list_of_pc']
            index_size = len(list_of_anno)
            index += 1
            for i in range(index_size):
                this_anno = list_of_anno[i]
                this_BB = BBs[i]
                this_PC = PCs[i]

                sample_id = list_of_sample_id[i]
                Sample_PCs = list_of_Sample_PC[i]
                Sample_BBs = list_of_Sample_BB[i]


                # IS THE POINT CLOUD OCCLUDED?
                occluded = this_anno["occluded"]
                if occluded in [0]:  # FULLY VISIBLE
                    occluded = 0
                elif occluded in [1, 2]:  # PARTIALLY AND FULLY OCCLUDED
                    occluded = 1
                else:
                    occluded = -1

                if "pointcloud".upper() in model_fusion.upper():
                    # INITIAL FRAME
                    if i == 0:
                        best_box = BBs[i]
                        score = 1.0

                        candidate_BBs = []
                        dynamic = -1
                    else:
                        # previous_PC = PCs[i - 1]
                        previous_BB = BBs[i - 1]
                        # previous_anno = list_of_anno[i - 1]
                        # IS THE SAMPLE dynamic?
                        if (np.linalg.norm(this_BB.center - previous_BB.center) > 0.709):  # for complete set
                            dynamic = 1
                        else:
                            dynamic = 0

                        near_list = select_near_100_from_300(Sample_BBs, [this_BB], 4)
                        Sample_PCs_part = [Sample_PCs[i] for i in near_list]
                        Sample_BBs_part = [Sample_BBs[i] for i in near_list]

                        Sample_PCs_reg = [utils.regularizePC(PC, 2048) for PC in Sample_PCs_part]

                        Sample_PCs_torch = torch.cat(Sample_PCs_reg, dim=0).cuda().float()

                        # calculate using ground truth
                        # model_PC = utils.getModel(PCs[:i], BBs[:i], offset=dataset.offset_BB, scale=dataset.scale_BB)
                        # model_PC = utils.getModel([PCs[0], PCs[i - 1]], [results_BBs[0], results_BBs[i - 1]],  offset=dataset.offset_BB, scale=dataset.scale_BB)

                        model_PC = utils.getModel([PCs[0]], [results_BBs[0]], offset=dataset.offset_BB, scale=dataset.scale_BB)

                        # save_dir = os.path.join('/media/fengzicai/fzc/3Dsiamesetracker/intermediate/model_PC/',
                        #                         str(list_of_anno[0]['track_id']))
                        # os.makedirs(save_dir, exist_ok=True)
                        # np.savetxt(os.path.join(save_dir, sample_id + '.txt'),
                        #            model_PC.points.reshape(-1, 3),
                        #            fmt='%.2f,%.2f,%.2f')

                        repeat_shape = np.ones(len(Sample_PCs_torch.shape), dtype=np.int32)
                        repeat_shape[0] = len(Sample_PCs_torch)
                        model_PC_encoded = utils.regularizePC(model_PC, 2048).repeat(tuple(repeat_shape)).cuda().float()

                        # decoded_PC = model.model.AE.forward(model_PC_encoded)

                        X = model.model.AE.encode(Sample_PCs_torch)
                        Y = model.model.AE.encode(model_PC_encoded)
                        # Y = model.model.AE.encode(decoded_PC)
                        # criterion_completion = ChamferLoss()
                        # loss2 = criterion_completion(decoded_PC, model_PC_encoded)


                        # save_dir = os.path.join('/media/fengzicai/fzc/3Dsiamesetracker/intermediate/decoded_PC/', str(list_of_anno[0]['track_id']))
                        # os.makedirs(save_dir, exist_ok=True)
                        # np.savetxt(os.path.join(save_dir, sample_id + '.txt'), decoded_PC[0].cpu().numpy().reshape(-1, 3),
                        #            fmt='%.2f,%.2f,%.2f')

                        output = F.cosine_similarity(X, Y, dim=1)
                        #
                        # output = model.model.score(torch.cat((X, Y), dim=1)).squeeze()
                        # output = output + 100.0/loss2
                        scores = output.detach().cpu().numpy()
                        idx = np.argmax(scores)
                        score = scores[idx]
                        best_box = Sample_BBs_part[idx]

                else:
                    raise ModuleNotFoundError
                results_BBs.append(best_box)
                results_scores.append(score)

                # estimate overlap/accuracy fro current sample
                this_overlap = estimateOverlap(this_BB, best_box, dim=IoU_Space)
                this_accuracy = estimateAccuracy(this_BB, best_box, dim=IoU_Space)

                Success_main.add_overlap(this_overlap)
                Precision_main.add_accuracy(this_accuracy)

                Success_main_all.add_overlap(this_overlap)
                Precision_main_all.add_accuracy(this_accuracy)

                if (dynamic >= 0):
                    Success_dynamic[dynamic].add_overlap(this_overlap)
                    Precision_dynamic[dynamic].add_accuracy(this_accuracy)

                if (occluded >= 0):
                    Success_occluded[occluded].add_overlap(this_overlap)
                    Precision_occluded[occluded].add_accuracy(this_accuracy)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if Success_main.count >= max_iter and max_iter >= 0:
                    return Success_main.average, Precision_main.average

            progress_bar.update(1)

            progress_bar.set_description(f'Test {epoch}: '
                              f'Time {batch_time.avg:.3f}s '
                              f'(it:{batch_time.val:.3f}s) '
                              f'Data:{data_time.avg:.3f}s '
                              f'(it:{data_time.val:.3f}s), '
                              f'Succ/Prec:'
                              f'{Success_main.average:.1f}/'
                              f'{Precision_main.average:.1f}')
            print(f'Succ/Prec:' f'{Success_main.average:.1f}/' f'{Precision_main.average:.1f}')

    logger.info(f'Succ/Prec:' f'{Success_main_all.average:.1f}/' f'{Precision_main_all.average:.1f}')

    return Success_main_all.average, Precision_main_all.average


def eval_one_epoch_joint_offline_metric(model, dataloader, result_dir, logger, epoch =-1,  shape_aggregation="all",
                                 model_fusion="pointcloud", IoU_Space=3, DetailedMetrics=False, max_iter=-1):
    # for reproducibility
    torch.manual_seed(25)
    np.random.seed(9568)

    #S3D
    batch_time = AverageMeter()
    data_time = AverageMeter()

    Accuracy_Completeness_main = Accuracy_Completeness()
    metric_accuracy = []
    metric_completeness = []
    metric_length = []

    #PR
    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)
    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()
    end = time.time()

    index = 0
    dataset = dataloader.dataset
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataset.list_of_tracklet_anno), leave=True, desc='eval', ncols=220)
    for datas in dataloader:

        # measure data loading time
        data_time.update((time.time() - end))
        for data in datas:

            list_of_sample_id, list_of_anno, list_of_Sample_PC, list_of_Sample_BB = \
                 data['sample_id'], data['tracklet_anno'], data['list_of_Sample_PC'], data['list_of_Sample_BB']

            BBs = data['list_of_BB']
            PCs = data['list_of_pc']
            index_size = len(list_of_anno)
            index += 1
            for i in range(index_size):
                this_BB = BBs[i]
                this_PC = PCs[i]

                if "pointcloud".upper() in model_fusion.upper():
                    # INITIAL FRAME
                    if i == 0:
                        pass
                    else:
                        #  Construct GT model
                        gt_model_PC_start_idx = max(0, i - 5)
                        gt_model_PC_end_idx = min(i + 5, len(PCs))
                        gt_model_PC = utils.getModel(
                            PCs[gt_model_PC_start_idx:gt_model_PC_end_idx],
                            BBs[gt_model_PC_start_idx:gt_model_PC_end_idx],
                            offset=dataset.offset_BB,
                            scale=dataset.scale_BB)

                        if (gt_model_PC.points.shape[1] > 0):
                            gt_model_PC = gt_model_PC.convertToPytorch().float().unsqueeze(2).permute(2, 0, 1)
                            gt_candidate_PC = utils.regularizePC(
                                utils.cropAndCenterPC(
                                    this_PC,
                                    this_BB,
                                    offset=dataset.offset_BB,
                                    scale=dataset.scale_BB), 2048).cuda()
                            decoded_PC = model.model.AE.forward(gt_candidate_PC).detach().cpu()
                            Accuracy_Completeness_main.update(decoded_PC, gt_model_PC)
                else:
                    raise ModuleNotFoundError

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            if Accuracy_Completeness_main.count > 100:
                metric_accuracy.append(Accuracy_Completeness_main.average[0])
                metric_completeness.append(Accuracy_Completeness_main.average[1])
                metric_length.append(Accuracy_Completeness_main.count)
                Accuracy_Completeness_main.reset()

            progress_bar.update(1)

            progress_bar.set_description(f'Test {epoch}: '
                              f'Time {batch_time.avg:.3f}s '
                              f'(it:{batch_time.val:.3f}s) '
                              f'Data:{data_time.avg:.3f}s '
                              f'(it:{data_time.val:.3f}s)'
                              f"number:({np.sum(metric_length) + Accuracy_Completeness_main.count})")

    metric_accuracy.append(Accuracy_Completeness_main.average[0])
    metric_completeness.append(Accuracy_Completeness_main.average[1])
    metric_length.append(Accuracy_Completeness_main.count)
    count = np.sum(metric_length)
    ACC = np.sum(np.multiply(np.array(metric_accuracy), np.array(metric_length))) / count
    Com = np.sum(np.multiply(np.array(metric_completeness), np.array(metric_length))) / count
    logger.info(f"Acc/Comp ({count}):")
    logger.info(f"{ACC:.4f}/{Com:.4f}")

    return ACC, Com

def eval_one_epoch_joint_offline_visual(model, dataloader, result_dir, logger, epoch =-1,  shape_aggregation="all",
                                 model_fusion="pointcloud", IoU_Space=3, DetailedMetrics=False, max_iter=-1):
    # for reproducibility
    torch.manual_seed(2825)
    np.random.seed(2779)

    #S3D
    batch_time = AverageMeter()
    data_time = AverageMeter()

    #PR
    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()
    end = time.time()

    index = 0
    dataset = dataloader.dataset
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataset.list_of_tracklet_anno), leave=True, desc='eval', ncols=220)
    for datas in dataloader:

        # measure data loading time
        data_time.update((time.time() - end))
        for data in datas:

            list_of_sample_id, list_of_anno, list_of_Sample_PC, list_of_Sample_BB = \
                 data['sample_id'], data['tracklet_anno'], data['list_of_Sample_PC'], data['list_of_Sample_BB']
            BBs = data['list_of_BB']
            PCs = data['list_of_pc']
            index_size = len(list_of_anno)
            index += 1
            for i in range(index_size):
                this_anno = list_of_anno[i]
                sample_id = list_of_sample_id[i]
                cur_scene = sample_id[:4]
                if "pointcloud".upper() in model_fusion.upper():
                    # INITIAL FRAME
                    if i == 0:
                        pass
                    else:

                        model_PC = utils.getModel(
                            PCs[:i],
                            BBs[:i],
                            offset=dataset.offset_BB,
                            scale=dataset.scale_BB)
                        visualizemodel(i, model_PC, cur_scene, this_anno['track_id'])
                        gt_candidate_PC = utils.regularizePC(model_PC, 2048).cuda()
                        model_PC_decoded = model.model.AE.forward(gt_candidate_PC)
                        model_PC_decoded = PointCloud(model_PC_decoded.detach().cpu().numpy()[0])
                        visualizereconstruction(i, model_PC_decoded, cur_scene, this_anno['track_id'])

                else:
                    raise ModuleNotFoundError

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            progress_bar.update(1)

            progress_bar.set_description(f'Test {epoch}: '
                              f'Time {batch_time.avg:.3f}s '
                              f'(it:{batch_time.val:.3f}s) '
                              f'Data:{data_time.avg:.3f}s '
                              f'(it:{data_time.val:.3f}s)')


def select_near_100_from_300(Samples_BBs, candidate_BBs, num=8):
    Samples_BBs_array = np.array([BB.center for BB in Samples_BBs])
    candidate_BBs_array = np.array([BB.center for BB in candidate_BBs])
    distances = []
    for candidate_BB_array in candidate_BBs_array:
        distances.append(np.linalg.norm(Samples_BBs_array-candidate_BB_array, axis=1))
    distance_array = np.array(distances).sum(axis=0)
    return list(distance_array.argsort())[:num]

def visualizereconstruction(i,model_PC_decoded,cur_scene, id): #,view_BB
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    from mpl_toolkits.mplot3d import Axes3D
    params = {
        'legend.fontsize': 'x-large',
        'figure.figsize': (15, 5),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'figure.max_open_warning': 1000
    }

    pylab.rcParams.update(params)

    # Create figure for RECONSTRUCTION
    fig = plt.figure(figsize=(9, 6), facecolor="white")
    # Create axis in 3D
    ax = Axes3D(fig)

    # select 2K point to visualize
    sample = np.random.randint(
        low=0,
        high=model_PC_decoded.points.shape[1],
        size=2048,
        dtype=np.int64)
    # Scatter plot the point cloud
    ax.scatter(
        model_PC_decoded.points[0, sample],
        model_PC_decoded.points[1, sample],
        model_PC_decoded.points[2, sample],
        s=3,
        c=model_PC_decoded.points[2, sample])

    # # Plot the car BB
    # order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]
    # ax.plot(
    #     view_BB.corners()[0, order],
    #     view_BB.corners()[1, order],
    #     view_BB.corners()[2, order],
    #     color="red",
    #     alpha=0.5,
    #     linewidth=3,
    #     linestyle=":")
    # order = [6, 7, 4, 5, 6, 2, 1, 5, 1, 0, 4]
    # ax.plot(
    #     view_BB.corners()[0, order],
    #     view_BB.corners()[1, order],
    #     view_BB.corners()[2, order],
    #     color="red",
    #     linewidth=3)

    # setup axis
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels([], fontsize=10)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([], fontsize=10)
    ax.set_zticks([-1, 0, 1])
    ax.set_zticklabels([], fontsize=10)
    ax.view_init(20, -140)
    plt.tight_layout()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1.5, 1.5)

    # Save figure as Decoded Model results
    os.makedirs(os.path.join(args.path_results, cur_scene, str(id), "Reconstruction"), exist_ok=True)
    plt.savefig(os.path.join(args.path_results, cur_scene, str(id),
                "Reconstruction", f"{i}_TDAE.png"),
                format='png', dpi=100)

def visualizemodel(i,model_PC, cur_scene, id): # view_BB,
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    from mpl_toolkits.mplot3d import Axes3D
    params = {
        'legend.fontsize': 'x-large',
        'figure.figsize': (15, 5),
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'figure.max_open_warning': 1000
    }

    pylab.rcParams.update(params)

    # Create figure for MODEL PC
    fig = plt.figure(figsize=(9, 6), facecolor="white")
    ax = Axes3D(fig)


    # sample 2K Points
    sample = np.random.randint(
        low=0, high=model_PC.points.shape[1], size=2048, dtype=np.int64)
    # Scatter plot the Point cloud
    ax.scatter(
        model_PC.points[0, sample],
        model_PC.points[1, sample],
        model_PC.points[2, sample],
        s=3,
        c=model_PC.points[2, sample])

    # # Plot the Bounding Box
    # order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]
    # ax.plot(
    #     view_BB.corners()[0, order],
    #     view_BB.corners()[1, order],
    #     view_BB.corners()[2, order],
    #     color="red",
    #     alpha=0.5,
    #     linewidth=3,
    #     linestyle=":")
    # order = [6, 7, 4, 5, 6, 2, 1, 5, 1, 0, 4]
    # ax.plot(
    #     view_BB.corners()[0, order],
    #     view_BB.corners()[1, order],
    #     view_BB.corners()[2, order],
    #     color="red",
    #     linewidth=3)

    # setup axis
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels([], fontsize=10)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([], fontsize=10)
    ax.set_zticks([-1, 0, 1])
    ax.set_zticklabels([], fontsize=10)
    ax.view_init(20, -140)
    plt.tight_layout()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1.5, 1.5)

    # Save figure as Model results
    os.makedirs(
        os.path.join(args.path_results, cur_scene, str(id), "Model"),
        exist_ok=True)
    plt.savefig(
        os.path.join(args.path_results, cur_scene, str(id), "Model",
                     f"{i}_TDAE.png"),
        format='png',
        dpi=100)

def eval_one_epoch_joint_offline_with_Kalman(model, dataloader, result_dir, logger, epoch =-1,  shape_aggregation="all",
                                             model_fusion="pointcloud", IoU_Space=3, DetailedMetrics=False, max_iter=-1):
    '''
    :param model:
    :param dataloader:
    :param result_dir:
    :param logger:
    :param epoch:
    :param shape_aggregation:
    :param model_fusion:
    :param IoU_Space:
    :param DetailedMetrics:
    :param max_iter:
    :return:
    '''

    # for reproducibility
    np.random.seed(196)

    #S3D
    batch_time = AverageMeter()
    data_time = AverageMeter()

    Success_main_all = Success()
    Precision_main_all = Precision()
    Accuracy_Completeness_main = Accuracy_Completeness()

    Precision_occluded = [Precision(), Precision()]
    Success_occluded = [Success(), Success()]

    Precision_dynamic = [Precision(), Precision()]
    Success_dynamic = [Success(), Success()]

    search_space_sampler = KalmanFiltering() #bnd=[1, 1, 5]
    # gaussian = KalmanFiltering()

    #PR
    final_output_dir = os.path.join(result_dir, 'final_result', 'data')
    os.makedirs(final_output_dir, exist_ok=True)
    logger.info('---- EPOCH %s JOINT EVALUATION ----' % epoch)
    logger.info('==> Output file: %s' % result_dir)
    model.eval()

    end = time.time()
    index = 0
    dataset = dataloader.dataset
    progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataset.list_of_tracklet_anno), leave=True, desc='eval', ncols=220)
    for datas in dataloader:

        # measure data loading time
        data_time.update((time.time() - end))
        for data in datas:

            # modelupdate
            model_numbers = 2  # math.ceil(len(PCs) / 8)
            model_update = ModelUpdate(model_numbers)
            samplesf = []

            search_space_sampler.reset()
            # gaussian.reset()

            Success_main = Success()
            Precision_main = Precision()


            # S3D
            results_BBs = []
            results_scores = []
            true_BBs = []
            list_of_previous_distance = []

            list_of_sample_id, list_of_anno, list_of_Sample_PC, list_of_Sample_BB = \
                 data['sample_id'], data['tracklet_anno'], data['list_of_Sample_PC'], data['list_of_Sample_BB']

            BBs = data['list_of_BB']
            PCs = data['list_of_pc']
            index_size = len(list_of_anno)
            index += 1
            for i in range(index_size):

                this_anno = list_of_anno[i]
                this_BB = BBs[i]
                this_PC = PCs[i]

                sample_id = list_of_sample_id[i]
                Sample_PCs = list_of_Sample_PC[i]
                Sample_BBs = list_of_Sample_BB[i]


                # IS THE POINT CLOUD OCCLUDED?
                occluded = this_anno["occluded"]
                if occluded in [0]:  # FULLY VISIBLE
                    occluded = 0
                elif occluded in [1, 2]:  # PARTIALLY AND FULLY OCCLUDED
                    occluded = 1
                else:
                    occluded = -1

                if "pointcloud".upper() in model_fusion.upper():
                    # INITIAL FRAME
                    if i == 0:
                        best_box = BBs[i]
                        candidate_BBs = []
                        dynamic = -1
                        best_score = 1.0
                        distance_with_previous = 0
                        distance_with_gt = 0
                    else:
                        # previous_PC = PCs[i - 1]
                        previous_BB = BBs[i - 1]
                        # IS THE SAMPLE dynamic?
                        if (np.linalg.norm(this_BB.center - previous_BB.center) > 0.709):  # for complete set
                            dynamic = 1
                        else:
                            dynamic = 0

                        best_score = 0
                        extend_search_space = search_space_sampler.sample(40)
                        extend_candidate_BBs = utils.generate_boxes(results_BBs[-1], search_space=extend_search_space)
                        extend_Sample_PCs = [
                            utils.cropAndCenterPC(
                                this_PC,
                                box,
                                offset=dataset.offset_BB,
                                scale=dataset.scale_BB) for box in extend_candidate_BBs
                        ]

                        extend_Sample_PCs_reg = [utils.regularizePC(PC, 2048) for PC in extend_Sample_PCs]
                        extend_Sample_PCs_torch = torch.cat(extend_Sample_PCs_reg, dim=0).cuda().float()
                        extend_model_PC = utils.getModel([PCs[0]], [results_BBs[0]], offset=dataset.offset_BB,
                                                         scale=dataset.scale_BB)
                        extend_repeat_shape = np.ones(len(extend_Sample_PCs_torch.shape), dtype=np.int32)
                        extend_repeat_shape[0] = len(extend_Sample_PCs_torch)
                        extend_model_PC_encoded = utils.regularizePC(extend_model_PC, 2048).repeat(
                            tuple(extend_repeat_shape)).cuda().float()
                        # decoded_PC = model.model.AE.forward(model_PC_encoded)
                        extend_X = model.model.AE.encode(extend_Sample_PCs_torch)
                        extend_Y = model.model.AE.encode(extend_model_PC_encoded)
                        # extend_output = model.model.score(torch.cat((extend_X, extend_Y), dim=1)).squeeze()
                        extend_output = F.cosine_similarity(extend_X, extend_Y, dim=1)
                        extend_scores = extend_output.detach().cpu().numpy()

                        # extend_idx = np.argmax(extend_scores)
                        # score = extend_scores[extend_idx]
                        # box = extend_candidate_BBs[extend_idx]
                        # if score > best_score:
                        #     best_score = score
                        #     best_box = box

                        search_space_sampler.addData(data=extend_search_space, score=extend_scores.T)

                        for candidate_BB in extend_candidate_BBs:
                            # near_list = select_near_100_from_300(Sample_BBs, [candidate_BB], 1)
                            near_list = select_near_100_from_300(Sample_BBs, [this_BB])
                            Sample_PCs_part = [Sample_PCs[i] for i in near_list]
                            Sample_BBs_part = [Sample_BBs[i] for i in near_list]
                            Sample_PCs_reg = [utils.regularizePC(PC, 2048) for PC in Sample_PCs_part]
                            Sample_PCs_torch = torch.cat(Sample_PCs_reg, dim=0).cuda().float()
                            model_PC = utils.getModel([PCs[0]], [results_BBs[0]],  offset=dataset.offset_BB, scale=dataset.scale_BB)
                            repeat_shape = np.ones(len(Sample_PCs_torch.shape), dtype=np.int32)
                            repeat_shape[0] = len(Sample_PCs_torch)
                            model_PC_encoded = utils.regularizePC(model_PC, 2048).repeat(tuple(repeat_shape)).cuda().float()
                            X = model.model.AE.encode(Sample_PCs_torch)
                            Y = model.model.AE.encode(model_PC_encoded)
                            output = model.model.score(torch.cat((X, Y), dim=1)).squeeze() # new_soat
                            # output = F.cosine_similarity(X, Y, dim=1)
                            scores = output.detach().cpu().numpy().reshape(-1) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                            idx = np.argmax(scores)
                            score = scores[idx]
                            box = Sample_BBs_part[idx]
                            if score > best_score:
                                best_score = score
                                best_box = box
                        # # distance_with_gt = np.linalg.norm(this_BB.center - best_box.center)

                        distance_with_previous = np.linalg.norm(results_BBs[-1].center - best_box.center)
                        if distance_with_previous > 2:
                            print(sample_id)
                            num = 2
                            if len(results_BBs) < num:
                                BB_list = results_BBs
                            else:
                                BB_list = results_BBs[-num:]

                            for res_BB in BB_list:
                                res_search_space = search_space_sampler.sample(147)
                                res_Sample_BBs_part = utils.generate_boxes(res_BB, search_space=res_search_space)
                                res_Sample_PCs_part = [
                                    utils.cropAndCenterPC(
                                        this_PC,
                                        box,
                                        offset=dataset.offset_BB,
                                        scale=dataset.scale_BB) for box in res_Sample_BBs_part
                                ]
                                res_Sample_PCs_reg = [utils.regularizePC(PC, 2048) for PC in res_Sample_PCs_part]
                                res_Sample_PCs_torch = torch.cat(res_Sample_PCs_reg, dim=0).cuda().float()

                                cropped_PC = utils.cropAndCenterPC(PCs[0], results_BBs[0], offset=dataset.offset_BB, scale=dataset.scale_BB) #修改对比不同的模型
                                points = np.ones((PCs[0].points.shape[0], 0))
                                points = np.concatenate([points, cropped_PC.points], axis=1)
                                if (i > 1) and (len(samplesf) != 0):
                                    for j in range(len(samplesf)):
                                        points = np.concatenate([points, samplesf[j]], axis=1)
                                res_model_PC = PointCloud(points)
                                #
                                # res_model_PC = utils.getModel([PCs[0], PCs[i - 1]], [results_BBs[0], results_BBs[i - 1]], offset=dataset.offset_BB, scale=dataset.scale_BB)
                                # res_model_PC = utils.getModel(PCs[:i], results_BBs, offset=dataset.offset_BB, scale=dataset.scale_BB)
                                res_repeat_shape = np.ones(len(res_Sample_PCs_torch.shape), dtype=np.int32)
                                res_repeat_shape[0] = len(res_Sample_PCs_torch)
                                res_model_PC_encoded = utils.regularizePC(res_model_PC, 2048).repeat(
                                    tuple(res_repeat_shape)).cuda().float()
                                res_X = model.model.AE.encode(res_Sample_PCs_torch)
                                res_Y = model.model.AE.encode(res_model_PC_encoded)
                                res_output = model.model.score(torch.cat((res_X, res_Y), dim=1)).squeeze()
                                # res_output = F.cosine_similarity(res_X, res_Y, dim=1)
                                res_scores = res_output.detach().cpu().numpy()
                                search_space_sampler.addData(data=res_search_space, score=res_scores.T)
                                res_idx = np.argmax(res_scores)
                                res_score = res_scores[res_idx]
                                res_box = res_Sample_BBs_part[res_idx]
                            best_score = res_score
                            best_box = res_box
                            if np.linalg.norm(results_BBs[-1].center - best_box.center) < 2:
                                true_BBs.append(best_box)
                        else:
                            true_BBs.append(best_box)

                        new_train_sample = utils.cropAndCenterPC(this_PC, best_box, offset=dataset.offset_BB, scale=dataset.scale_BB).points

                        if new_train_sample.shape[1] != 1:
                            merged_sample, new_sample, merged_sample_id, new_sample_id = \
                                model_update.update_sample_space_model(samplesf, new_train_sample, len(samplesf))
                            if merged_sample_id == -1:
                                if len(samplesf) == model_numbers:
                                    samplesf[new_sample_id] = new_sample
                                else:
                                    samplesf.append(new_sample)
                            else:
                                if new_sample_id == -1:
                                    samplesf[merged_sample_id] = merged_sample
                                else:
                                    samplesf[new_sample_id] = new_sample
                                    samplesf[merged_sample_id] = merged_sample
                else:
                    raise ModuleNotFoundError
                #
                # from pyquaternion import Quaternion
                # cur_scene = sample_id[:4]
                # cur_sample_id = int(sample_id[4:])
                # kitti_output_dir = os.path.join('/data/3DTracking/visual2', cur_scene, str(this_anno['track_id']))
                # os.makedirs(kitti_output_dir, exist_ok=True)
                # calib = dataset.dataset.read_calib_file(
                #     os.path.join(dataset.dataset.KITTI_Folder, 'calib', '%04d.txt' % int(cur_scene)))
                # calib = calibration.Calibration(calib)
                #
                # cur_boxes3d = np.array([best_box.center[0], best_box.center[1] + best_box.wlh[2] / 2, best_box.center[2],
                #                         best_box.wlh[2], best_box.wlh[0], best_box.wlh[1], best_box.ry]).reshape(1, 7) #(best_box.orientation/Quaternion(axis=[1, 0, 0], radians=np.pi / 2)).angle
                # image_shape = dataset.dataset.get_image_shape(cur_scene, '%06d' % cur_sample_id)
                # save_kitti_format(cur_sample_id, calib, cur_boxes3d, kitti_output_dir, best_score, image_shape)

                results_BBs.append(best_box)
                results_scores.append(best_score)
                # list_of_previous_distance.append(distance_with_previous)
                # list_of_gt_distance.append(distance_with_gt)

                # estimate overlap/accuracy for current sample
                this_overlap = estimateOverlap(this_BB, best_box, dim=IoU_Space)
                this_accuracy = estimateAccuracy(this_BB, best_box, dim=IoU_Space)

                # if this_overlap == 0 and this_accuracy > 1.0:  # 如果出现跟踪丢失的情况,第二次使用真实值赋值
                #     results_BBs[-1] = BBs[i]

                Success_main.add_overlap(this_overlap)
                Precision_main.add_accuracy(this_accuracy)

                Success_main_all.add_overlap(this_overlap)
                Precision_main_all.add_accuracy(this_accuracy)

                if (dynamic >= 0):
                    Success_dynamic[dynamic].add_overlap(this_overlap)
                    Precision_dynamic[dynamic].add_accuracy(this_accuracy)

                if (occluded >= 0):
                    Success_occluded[occluded].add_overlap(this_overlap)
                    Precision_occluded[occluded].add_accuracy(this_accuracy)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if Success_main.count >= max_iter and max_iter >= 0:
                    return Success_main.average, Precision_main.average

            progress_bar.update(1)

            progress_bar.set_description(f'Test {epoch}: '
                              f'Time {batch_time.avg:.3f}s '
                              f'(it:{batch_time.val:.3f}s) '
                              f'Data:{data_time.avg:.3f}s '
                              f'(it:{data_time.val:.3f}s), '
                              f'Succ/Prec:'
                              f'{Success_main.average:.1f}/'
                              f'{Precision_main.average:.1f}')
            print(f'Succ/Prec:' f'{Success_main.average:.1f}/' f'{Precision_main.average:.1f}')

    logger.info(f'Succ/Prec:' f'{Success_main_all.average:.1f}/' f'{Precision_main_all.average:.1f}')

    if DetailedMetrics:
        logger.info(f"Succ/Prec fully visible({Success_occluded[0].count}):")
        logger.info(f"{Success_occluded[0].average:.1f}/{Precision_occluded[0].average:.1f}")

        logger.info(f"Succ/Prec occluded({Success_occluded[1].count}):")
        logger.info(f"{Success_occluded[1].average:.1f}/{Precision_occluded[1].average:.1f}")

        logger.info(f"Succ/Prec dynamic({Success_dynamic[0].count}):")
        logger.info(f"{Success_dynamic[0].average:.1f}/{Precision_dynamic[0].average:.1f}")

        logger.info(f"Succ/Prec static({Success_dynamic[1].count}):")
        logger.info(f"{Success_dynamic[1].average:.1f}/{Precision_dynamic[1].average:.1f}")

        logger.info(f"Acc/Comp ({Accuracy_Completeness_main.count}):")
        logger.info(f"{Accuracy_Completeness_main.average[0]:.4f}/{Accuracy_Completeness_main.average[1]:.4f}")

    return Success_main_all.average, Precision_main_all.average


def eval_single_ckpt(root_result_dir):
    if args.model_fusion is not None:
        root_result_dir = os.path.join(root_result_dir, 'eval', args.model_fusion)
    else:
        root_result_dir = os.path.join(root_result_dir, 'eval')

    # set epoch_id and output dir
    num_list = re.findall(r'\d+', args.chkpt_file) if args.chkpt_file is not None else []
    epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    root_result_dir = os.path.join(root_result_dir, 'epoch_%s' % epoch_id, 'rpnpred147')

    if args.extra_tag != 'default':
        root_result_dir = os.path.join(root_result_dir, args.extra_tag)
    os.makedirs(root_result_dir, exist_ok=True)

    loggerpath = os.path.join(root_result_dir, datetime.now().strftime('%Y-%m-%d %H-%M-%S.log'))
    logger = create_logger(loggerpath)
    logger.info('**********************Start logging**********************')
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
        logger.info('CUDA_VISIBLE_DEVICES=%s' % os.environ["CUDA_DEVICE_ORDER"])
    logger.info("Parameters:")
    for arg in vars(args):
        logger.info(arg.rjust(15) + " : " + str(getattr(args, arg)))
    save_config_to_file(cfg, logger=logger)
    start = time.time()

    if 'ARGO' in args.dataset.upper():
        from argo.rpn_train_dataset import Train_RPN  # ARGO
        from argo.siamese_offline_dataset import SiameseTest #gaile jiazai jinlai leibie de mingzi car->vehicle
    else:
        from tools.rpn_train_dataset import Train_RPN
        #from tools.siamese_offline_dataset import SiameseTrain
        from tools.siamese_offline_dataset import SiameseTest

    # create dataloader & network
    if args.eval_mode == 'siameseoffline':
        test_set = SiameseTest(model_inputsize=args.inputsize, path=args.dataset_path,
                               split=cfg.TEST.SPLIT, category_name=cfg.CLASSES, regress=args.regress,
                               sigma_Gaussian=args.sigma_Gaussian, offset_BB=args.offset_BB,
                               scale_BB=args.scale_BB, logger=logger, mode='TEST')
        test_loader = DataLoader(test_set, collate_fn=lambda x: x, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True)
    elif args.eval_mode == 'rpn':
        test_set = Train_RPN(path=args.dataset_path, split=cfg.TEST.SPLIT, category_name=cfg.CLASSES, #
                              npoints=cfg.RPN.NUM_POINTS, mode='EVAL', logger=logger)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, pin_memory=True,
                                  num_workers=args.workers, shuffle=True, collate_fn=test_set.collate_batch,
                                  drop_last=True)
    else:
        raise ModuleNotFoundError
    model = PointRCNN(num_classes=test_loader.dataset.num_class, use_xyz=True, mode='TEST', bneck_size=args.bneck_size, DenseAutoEncoder=args.DenseAutoEncoder)

    logger.info(model)
    model.cuda()

    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


    # copy important files to backup
    backup_dir = os.path.join(root_result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp *.py %s/' % backup_dir)
    os.system('cp ../tools/*.py %s/' % backup_dir)
    os.system('cp ../lib/net/*.py %s/' % backup_dir)
    os.system('cp ../lib/datasets/kitti_rcnn_dataset.py %s/' % backup_dir)

    load_ckpt_based_on_args(model, logger)

    if args.mgpus:
        model = nn.DataParallel(model)

    max_epoch = 5 #5#5 #3

    for epoch in range(max_epoch):
        if args.eval_mode == 'siameseoffline':
            if not args.detailed_metrics:
                Success_run = AverageMeter()
                Precision_run = AverageMeter()

                # Succ, Prec = eval_one_epoch_joint_offline(model, test_loader, root_result_dir, logger, epoch=epoch + 1,
                #                                           IoU_Space=args.IoU_Space, DetailedMetrics=args.detailed_metrics, max_iter=-1)
                #
                Succ, Prec = eval_one_epoch_joint_offline_with_Kalman(model, test_loader, root_result_dir, logger, epoch=epoch + 1,
                                                                      IoU_Space=args.IoU_Space, DetailedMetrics=args.detailed_metrics,
                                                                      max_iter=-1)
                Success_run.update(Succ)
                Precision_run.update(Prec)
                logger.info(f"mean Succ/Prec {Success_run.avg}/{Precision_run.avg}")
            else:
                # eval_one_epoch_joint_offline_metric(model, test_loader, root_result_dir, logger, epoch=epoch + 1,
                #                                     IoU_Space=args.IoU_Space, DetailedMetrics=args.detailed_metrics,
                #                                     max_iter=-1)
                eval_one_epoch_joint_offline_visual(model, test_loader, root_result_dir, logger, epoch=epoch + 1,
                                                    IoU_Space=args.IoU_Space, DetailedMetrics=args.detailed_metrics,
                                                    max_iter=-1)
        elif args.eval_mode == 'rpn':
            eval_one_epoch_rpn(model, test_loader, epoch_id, root_result_dir, logger)

    logger.info(
        'Total Execution Time is {0} seconds'.format(time.time() - start))


def save_rpn_features(seg_result, rpn_scores_raw, pts_features, backbone_xyz, backbone_features, kitti_features_dir,
                      sample_id):
    pts_intensity = pts_features[:, 0]

    output_file = os.path.join(kitti_features_dir, '%06d.npy' % sample_id)
    xyz_file = os.path.join(kitti_features_dir, '%06d_xyz.npy' % sample_id)
    seg_file = os.path.join(kitti_features_dir, '%06d_seg.npy' % sample_id)
    intensity_file = os.path.join(kitti_features_dir, '%06d_intensity.npy' % sample_id)
    np.save(output_file, backbone_features)
    np.save(xyz_file, backbone_xyz)
    np.save(seg_file, seg_result)
    np.save(intensity_file, pts_intensity)
    rpn_scores_raw_file = os.path.join(kitti_features_dir, '%06d_rawscore.npy' % sample_id)
    np.save(rpn_scores_raw_file, rpn_scores_raw)


def save_kitti_format(sample_id, calib, bbox3d, kitti_output_dir, scores, img_shape):
    corners3d = kitti_utils.boxes3d_to_corners3d(bbox3d)
    img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

    img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape[1] - 1)
    img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape[0] - 1)
    img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape[1] - 1)
    img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape[0] - 1)

    img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
    img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
    box_valid_mask = np.logical_and(img_boxes_w < img_shape[1] * 0.8, img_boxes_h < img_shape[0] * 0.8)

    kitti_output_file = os.path.join(kitti_output_dir, '%06d.txt' % sample_id)
    with open(kitti_output_file, 'w') as f:
        for k in range(bbox3d.shape[0]):
            if box_valid_mask[k] == 0:
                continue
            x, z, ry = bbox3d[k, 0], bbox3d[k, 2], bbox3d[k, 6]
            beta = np.arctan2(z, x)
            alpha = -np.sign(beta) * np.pi / 2 + beta + ry

            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f' %
                  (cfg.CLASSES, alpha, img_boxes[k, 0], img_boxes[k, 1], img_boxes[k, 2], img_boxes[k, 3],
                   bbox3d[k, 3], bbox3d[k, 4], bbox3d[k, 5], bbox3d[k, 0], bbox3d[k, 1], bbox3d[k, 2],
                   bbox3d[k, 6], scores[k]), file=f) # scores


def eval_one_epoch_rpn(model, dataloader, epoch_id, result_dir, logger):
    np.random.seed(1024)
    mode = 'EVAL'

    if args.save_result or args.save_rpn_feature:
        parent_kitti_output_dir = os.path.join(result_dir, 'detections', 'data')
        os.makedirs(parent_kitti_output_dir, exist_ok=True)

    logger.info('---- EPOCH %s RPN EVALUATION ----' % epoch_id)
    model.eval()

    dataset = dataloader.dataset
    cnt = 0

    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval')

    for data in dataloader:
        sample_id_list, pts_rect, pts_features, pts_input = \
            data['sample_id'], data['pts_rect'], data['pts_features'], data['pts_input']
        cnt += len(sample_id_list)
        rpn_cls_label, rpn_reg_label = data['rpn_cls_label'], data['rpn_cls_label']
        gt_boxes3d = data['gt_boxes3d']
        rpn_cls_label = torch.from_numpy(rpn_cls_label).cuda(non_blocking=True).long()
        if gt_boxes3d.shape[1] == 0:
            pass
        else:
            gt_boxes3d = torch.from_numpy(gt_boxes3d).cuda(non_blocking=True).float()
        input = torch.from_numpy(pts_input).cuda(non_blocking=True).float()
        input_data = {'pts_input': input, 'gt_boxes3d': gt_boxes3d, 'sample_id': sample_id_list}

        # model inference
        ret_dict = model(input_data)
        rpn_cls, rpn_reg = ret_dict['rpn_cls'], ret_dict['rpn_reg']
        backbone_xyz, backbone_features = ret_dict['backbone_xyz'], ret_dict['backbone_features']

        rpn_scores_raw = rpn_cls[:, :, 0]

        # proposal layer
        if args.mgpus:
            rois, roi_scores_raw = model.module.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
        else:
            rois, roi_scores_raw = model.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)
        batch_size = rois.shape[0]

        for bs_idx in range(batch_size):
            sample_id = sample_id_list[bs_idx]

            cur_scene = sample_id[:4]
            cur_sample_id = int(sample_id[4:])
            kitti_output_dir = os.path.join(parent_kitti_output_dir, cur_scene)
            os.makedirs(kitti_output_dir, exist_ok=True)

            cur_scores_raw = roi_scores_raw[bs_idx]
            cur_boxes3d = rois[bs_idx]

            if args.save_result or args.save_rpn_feature:

                # save as kitti format
                calib = dataset.dataset.read_calib_file(os.path.join(dataset.dataset.KITTI_Folder, 'calib', '%04d.txt' % int(cur_scene)))
                calib = calibration.Calibration(calib)
                cur_boxes3d = cur_boxes3d.cpu().numpy()
                cur_scores_raw = cur_scores_raw.cpu().numpy()
                image_shape = dataset.dataset.get_image_shape(cur_scene, '%06d' % cur_sample_id)
                save_kitti_format(cur_sample_id, calib, cur_boxes3d, kitti_output_dir, cur_scores_raw, image_shape)

        disp_dict = {'mode': mode}
        progress_bar.set_postfix(disp_dict)
        progress_bar.update()

    progress_bar.close()
    logger.info(str(datetime.now()))
    logger.info('-------------------performance of epoch %s---------------------' % epoch_id)


if __name__ == '__main__':
    parser = ArgumentParser(description='Test Shape Completion for 3D Tracking', formatter_class= ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='KITTI', help='ARGO or KITTI')
    parser.add_argument('--detailed_metrics', required=False, action='store_true')
    parser.add_argument('--dataset_path', required=False, type=str, default='/media/fengzicai/fzc/3Dsiamesetracker/data/training',
                        help='dataset Path')
    parser.add_argument('--bneck_size', required=False, type=int, default=128,
                        help='Size of the bottleneck')
    parser.add_argument('--batch_size', required=False, type=int, default=32, help='Batch size')
    parser.add_argument('--GPU', required=False, type=int, default=-1,
                        help='ID of the GPU to use')
    parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
    parser.add_argument('--model_fusion', required=False, type=str, default=None,
                        help='early or late fusion (pointcloud/latent/space)')
    parser.add_argument('--shape_aggregation', required=False, type=str, default="firstandprevious",
                        help='Aggregation of shapes (first/previous/firstandprevious/all/AVG/MEDIAN/MAX)')
    parser.add_argument('--search_space', required=False, type=str, default='Exhaustive',
                        help='Search space (Exhaustive/Kalman/Particle/GMM<N>)')
    parser.add_argument('--number_candidate', required=False, type=int, default=125,
                        help='Number of candidate for Kalman, Particle or GMM search space')
    parser.add_argument('--reference_BB', required=False, type=str, default="current_gt",
                        help='previous_result/previous_gt/current_gt')
    parser.add_argument('--regress', required=False, type=str, default='gaussian',
                        help='how to regress (IoU/Gaussian)')
    parser.add_argument('--category_name', required=False, type=str, default='Car',
                        help='Object to Track (Car/Pedetrian/Van/Cyclist)')
    parser.add_argument('--lambda_completion', required=False, type=float, default=1e-6,
                        help='lambda ratio for completion loss')
    parser.add_argument('--sigma_Gaussian', required=False, type=float, default=1,
                        help='Gaussian distance variation sigma for regression in training')
    parser.add_argument('--offset_BB', required=False, type=float, default=0, help='offset around the BB in meters')
    parser.add_argument('--scale_BB', required=False, type=float, default=1.25, help='scale of the BB before cropping')
    parser.add_argument('--IoU_Space', required=False, type=int, default=3, help='IoUBox vs IoUBEV (2 vs 3)')
    parser.add_argument('--max_num_worker', required=False, type=int, default=6, help='number of worker')
    parser.add_argument('--inputsize', required=False, type=float, default=2048,
                        help='input size of the model')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    parser.add_argument("--eval_mode", type=str, default='siamese', required=True, help="specify the evaluation mode")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")
    parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
    parser.add_argument("--extra_tag", type=str, default='default', help="extra tag for multiple evaluation")
    parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='specify the config for training')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument("--siam_ckpt", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")
    parser.add_argument("--chkpt_file", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")
    parser.add_argument("--rpn_ckpt", type=str, default=None, help="specify a ckpt directory to be evaluated if needed")
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--save_rpn_feature', action='store_true', default=False,
                        help='save features for separately rcnn training and evaluation')
    parser.add_argument('--save_result', action='store_true', default=False, help='save evaluation results to files')
    parser.add_argument('--test', action='store_true', default=False, help='evaluate without ground truth')
    parser.add_argument('--DenseAutoEncoder', action='store_true', default=False, help='use proposed model')
    parser.add_argument('--path_results', type=str, help='path to save the results')
    # parser.add_argument('--model_fusion', required=False, type=str, default="pointcloud",
    #                     help='early or late fusion (pointcloud/latent/space)')

    args = parser.parse_args()
    # merge config and log to file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.eval_mode == 'siamese' and args.model_fusion == 'pointcloud':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = cfg.RPN.FIXED = True
        root_result_dir = os.path.join('../', 'output', 'siamese', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'siamese', cfg.TAG, 'ckpt')
    elif args.eval_mode == 'siameseoffline':  # and args.model_fusion == 'latent':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'siamese', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'siamese', cfg.TAG, 'ckpt')
    elif args.eval_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
        ckpt_dir = os.path.join('../', 'output', 'rpn', cfg.TAG, 'ckpt')
    else:
        raise NotImplementedError

    if args.ckpt_dir is not None:
        ckpt_dir = args.ckpt_dir
    if args.output_dir is not None:
        root_result_dir = args.output_dir

    os.makedirs(root_result_dir, exist_ok=True)
    with torch.no_grad():
        if args.eval_all:
            pass
        else:
            eval_single_ckpt(root_result_dir)

