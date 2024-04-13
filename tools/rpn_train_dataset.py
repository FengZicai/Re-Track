from torch.utils.data import Dataset
import torch.utils.data as torch_data
from PIL import Image

import torch
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from lib.config import cfg

from tqdm import tqdm
import tools.utils as utils
from tools.utils import getModel
import numpy as np
import pandas as pd
import os
from tools.data_classes import PointCloud, Box
import pickle
from pyquaternion import Quaternion

class kittiDataset():
    def __init__(self, path):
        self.KITTI_Folder = path
        self.KITTI_velo = os.path.join(self.KITTI_Folder, "velodyne")
        self.KITTI_label = os.path.join(self.KITTI_Folder, "label_02")
        self.KITTI_imgae = os.path.join(self.KITTI_Folder, 'image_02')
        self.KITTI_pred = os.path.join(self.KITTI_Folder, 'rpnpred48')

    def getSceneID(self, split):
        if "TRAIN" in split.upper():  # Training SET
            if "TINY" in split.upper():
                sceneID = [14]
            else:
                sceneID = list(range(0, 17))
        elif "VAL" in split.upper():  # Validation Set
            if "TINY" in split.upper():
                sceneID = [3]
            else:
                sceneID = list(range(17, 19))
        elif "TEST" in split.upper():  # Testing Set
            if "TINY" in split.upper():
                sceneID = [0]
            else:
                sceneID = list(range(19, 21))

        else:  # Full Dataset
            sceneID = list(range(21))
        return sceneID

    def getBBandPC(self, anno):
        calib_path = os.path.join(self.KITTI_Folder, 'calib', anno['scene'] + ".txt")
        calib = self.read_calib_file(calib_path)
        # transf_mat = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))
        PC, box = self.getPCandBBfromPandas(anno)  #transf_mat 可以考虑修改
        # PC, box = self.getPCandBBfromPandas(anno, transf_mat)
        return PC, box, calib

    def getListOfAnno(self, sceneID, category_name="Car", if_train_rpn=False):
        list_of_scene = [
            path for path in os.listdir(self.KITTI_velo)
            if os.path.isdir(os.path.join(self.KITTI_velo, path)) and
            int(path) in sceneID
        ]
        # print(self.list_of_scene)
        list_of_tracklet_anno = []

        list_of_frame_anno = []

        self.all_frame_anno = {}
        for scene in list_of_scene:

            label_file = os.path.join(self.KITTI_label, scene + ".txt")
            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            df.insert(loc=0, column="scene", value=scene)

            self.all_frame_anno[scene] = df

            df = df[df["type"] == category_name]

            if if_train_rpn:
                # df.frame.tolist()
                for frame_index in df.frame.unique():
                    # df_frame_anno = df[df["frame"] == frame_index]
                    # df_frame_anno = df_frame_anno.reset_index(drop=True)
                    # frame_anno = [anno for index, anno in df_frame_anno.iterrows()]
                    # dict_of_frame_anno[scene + f'{frame_index:06}'] = frame_anno
                    # scene中有一些帧是缺失的,不,有一些帧的anno是缺失的,也会出现某些帧的雷达文件缺失的情况
                    list_of_frame_anno.append(scene + f'{frame_index:06}')

            else:
                for track_id in df.track_id.unique():
                    df_tracklet = df[df["track_id"] == track_id]
                    df_tracklet = df_tracklet.reset_index(drop=True)
                    tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                    list_of_tracklet_anno.append(tracklet_anno)

        return list_of_tracklet_anno, list_of_frame_anno

    def getPCandBBfromPandas(self, box):
        center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
        size = [box["width"], box["length"], box["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(box, center, size, orientation)

        try:
            # VELODYNE PointCloud
            velodyne_path = os.path.join(self.KITTI_velo, box["scene"], f'{box["frame"]:06}.bin')
            PC = PointCloud(
                np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
        except FileNotFoundError:
            # in case the Point cloud is missing
            # (0001/[000177-000180].bin)
            PC = PointCloud(np.array([[0, 0, 0, 0]]).T)
        return PC, BB

    # def getPCandBBfromPandas(self, box, calib):
    #     center = [box["x"], box["y"] - box["height"] / 2, box["z"]]
    #     size = [box["width"], box["length"], box["height"]]
    #     orientation = Quaternion(
    #         axis=[0, 1, 0], radians=box["rotation_y"]) * Quaternion(
    #             axis=[1, 0, 0], radians=np.pi / 2)
    #     BB = Box(box, center, size, orientation)
    #
    #     try:
    #         # VELODYNE PointCloud
    #         velodyne_path = os.path.join(self.KITTI_velo, box["scene"],
    #                                      f'{box["frame"]:06}.bin')
    #         PC = PointCloud(
    #             np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
    #         PC.transform(calib)
    #     except FileNotFoundError:
    #         # in case the Point cloud is missing
    #         # (0001/[000177-000180].bin)
    #         PC = PointCloud(np.array([[0, 0, 0]]).T)
    #
    #     return PC, BB

    def read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        data = {'P2': P2.reshape(3, 4), 'P3': P3.reshape(3, 4), 'R0': R0.reshape(3, 3),
        'Tr_velo_cam': Tr_velo_to_cam.reshape(3, 4)}
        return data

    def get_image_shape(self, scene, idx):
        img_file = os.path.join(self.KITTI_imgae, scene, idx + '.png')
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

class TrainRPNDataset(Dataset):

    def __init__(self, path, split, category_name="Car"):

        self.dataset = kittiDataset(path=path)
        self.split = split
        self.sceneID = list(range(21)) #self.dataset.getSceneID(split=split)
        self.getBBandPC = self.dataset.getBBandPC
        self.category_name = category_name

        if "TRAIN" in self.split.upper() or "TEST" in self.split.upper():
            if_train_rpn = True
        else:
            if_train_rpn = False

        _, self.list_of_frame_anno = self.dataset.getListOfAnno(
            self.sceneID, category_name, if_train_rpn)

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)


class Train_RPN(TrainRPNDataset):
    def __init__(self, path, split="", category_name="Car",
                 npoints=16384, mode='TRAIN', logger=None,
                 random_select=True):

        super().__init__(path=path, split=split, category_name=category_name)
        if category_name == 'Car':
            self.classes = ('Background', 'Car')
            # aug_scene_root_dir = os.path.join(path, 'KITTI', 'aug_scene')
        elif category_name == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif category_name == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
            # aug_scene_root_dir = os.path.join(path, 'KITTI', 'aug_scene_ped')
        elif category_name == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
            # aug_scene_root_dir = os.path.join(path, 'KITTI', 'aug_scene_cyclist')
        else:
            assert False, "Invalid classes: %s" % category_name

        self.num_class = self.classes.__len__()
        self.npoints = npoints
        self.sample_id_list = []
        self.random_select = random_select
        self.logger = logger
        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        if not self.random_select:
            self.logger.warning('random select is False')

        self.packed = True
        self.savedir = '/data/3DTracking/tools/gt_database'
        os.makedirs(self.savedir, exist_ok=True)
        self.save_rpn_name = os.path.join(self.savedir, '%s_gt_database_%s_%s.pkl' % (self.split, self.category_name, 'rpn'))
        if not self.packed:
            # 取出每一帧的点云数据和BB
            self.logger.info("loading PC and BB of one frame...")
            self.gt_database = []
            for key in tqdm(self.list_of_frame_anno):
                box_list = self.filtrate_objects(self.get_label(key))
                if len(box_list) == 0:
                    print('No gt object')
                    continue
                PC, _ , calib_file = self.getBBandPC(box_list[0].src)
                if PC.points.size == 3:
                    print('the Point cloud is missing')
                    continue
                calib = calibration.Calibration(calib_file)
                pts_rect = PC.points.T
                pts_rect = calib.lidar_to_rect(pts_rect)
                pts_intensity = PC.intensity.T

                gt_boxes3d = np.zeros((box_list.__len__(), 7), dtype=np.float32)
                for k, box in enumerate(box_list):
                    gt_boxes3d[k, :] = np.array([box.src["x"], box.src["y"], box.src["z"],
                                                        box.src["height"], box.src["width"], box.src["length"],
                                                        box.src["rotation_y"]])

                boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(torch.from_numpy(pts_rect),
                                                                         torch.from_numpy(gt_boxes3d))
                for k in range(boxes_pts_mask_list.__len__()):
                    pt_mask_flag = (boxes_pts_mask_list[k].numpy() == 1)
                    cur_pts = pts_rect[pt_mask_flag].astype(np.float32)
                    cur_pts_intensity = pts_intensity[pt_mask_flag].astype(np.float32)
                    sample_dict = {'sample_id': key,
                                   'cls_type': box_list[k].src["type"],
                                   'gt_box3d': gt_boxes3d[k],
                                   'points': cur_pts,
                                   'intensity': cur_pts_intensity,
                                   'box': box_list[k]}
                    self.gt_database.append(sample_dict)
            with open(self.save_rpn_name, 'wb') as f:
                pickle.dump(self.gt_database, f)
        else:
            self.gt_database = pickle.load(open(self.save_rpn_name, 'rb'))

        self.logger.info("PC and BB of one frame loaded!")

        if cfg.GT_AUG_HARD_RATIO > 0:
            easy_list, hard_list = [], []
            for k in range(self.gt_database.__len__()):
                obj = self.gt_database[k]
                if obj['points'].shape[0] > 100:
                    easy_list.append(obj)
                else:
                    hard_list.append(obj)
            self.gt_database = [easy_list, hard_list]
            self.logger.info('Loading gt_database(easy(pt_num>100): %d, hard(pt_num<=100): %d)'
                        % (len(easy_list), len(hard_list)))
        else:
            self.logger.info('Loading gt_database(%d)' % len(self.gt_database))
        if mode == 'TRAIN':
            self.preprocess_rpn_training_data()
        else:
            for sample_id in self.list_of_frame_anno:
                box_list = self.filtrate_objects(self.get_label(sample_id))
                PC, _, _ = self.getBBandPC(box_list[0].src)
                if PC.points.size == 3:
                    print('the Point cloud is missing')
                    continue
                self.sample_id_list.append(sample_id)
            self.logger.info('load testing samples from %s' % self.dataset.KITTI_Folder)
            self.logger.info('Done: total test samples %d' % len(self.sample_id_list))
            
    def preprocess_rpn_training_data(self):
        """
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        self.logger.info('Loading %s samples...' % self.mode)
        for sample_id in self.list_of_frame_anno:
            # sample_id = int(self.image_idx_list[idx])
            box_list = self.filtrate_objects(self.get_label(sample_id))
            if len(box_list) == 0:
                # self.self.logger.info('No gt classes: %10d' % sample_id)
                continue
            PC, _, _ = self.getBBandPC(box_list[0].src)
            if PC.points.size == 3:
                print('the Point cloud is missing')
                continue
            self.sample_id_list.append(sample_id)
        self.logger.info('Done: filter %s results: %d\n' % (self.mode, len(self.sample_id_list)))

    def get_label(self, index):

        scene = index[:4]
        idx = int(index[4:])
        df_scene = self.dataset.all_frame_anno[scene]
        df_frame = df_scene[df_scene["frame"] == idx]
        df_frame = df_frame.reset_index(drop=True)
        frame_anno = [anno for index, anno in df_frame.iterrows()]
        list_of_frame_BBs = [None] * len(frame_anno)
        for index in range(len(frame_anno)):
            anno = frame_anno[index]
            _, box, _ = self.getBBandPC(anno)
            list_of_frame_BBs[index] = box
        return list_of_frame_BBs

    def get_image_shape(self, scene, idx):
        return self.dataset.get_image_shape(scene, idx)


    @staticmethod
    def get_rpn_features(rpn_feature_dir, idx):
        rpn_feature_file = os.path.join(rpn_feature_dir, '%06d.npy' % idx)
        rpn_xyz_file = os.path.join(rpn_feature_dir, '%06d_xyz.npy' % idx)
        rpn_intensity_file = os.path.join(rpn_feature_dir, '%06d_intensity.npy' % idx)
        if cfg.RCNN.USE_SEG_SCORE:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_rawscore.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
            rpn_seg_score = torch.sigmoid(torch.from_numpy(rpn_seg_score)).numpy()
        else:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_seg.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
        return np.load(rpn_xyz_file), np.load(rpn_feature_file), np.load(rpn_intensity_file).reshape(-1), rpn_seg_score

    def filtrate_objects(self, box_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param box_list: list
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_box_list = []
        for box in box_list:
            if box.src["type"] not in type_whitelist:  # rm Van, 20180928
                continue
            if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(box.pos) is False):
                continue
            valid_box_list.append(box)
        return valid_box_list

    @staticmethod
    def filtrate_dc_objects(obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ['DontCare']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

    @staticmethod
    def check_pc_range(xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_id_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(index)
        else:
            raise NotImplementedError

    def get_rpn_sample(self, index):
        sample_id = self.sample_id_list[index]
        scene = sample_id[:4]
        df_scene = self.dataset.all_frame_anno[scene]
        idx = sample_id[4:]
        if int(idx) < 10000:
            img_shape = self.get_image_shape(scene, idx)
            df_frame = df_scene[df_scene["frame"] == int(idx)]
            df_frame = df_frame.reset_index(drop=True)
            frame_anno = [anno for index, anno in df_frame.iterrows()]
            anno = frame_anno[0]
            pc, _, calib_file = self.getBBandPC(anno)
            calib = calibration.Calibration(calib_file)
            pts_rect = pc.points.T
            pts_rect = calib.lidar_to_rect(pts_rect)
            pts_intensity = pc.intensity.T #SHAPE(NUM,)
        else:
            calib = self.get_calib(sample_id % 10000)
            img_shape = self.get_image_shape(sample_id % 10000)

            pts_file = os.path.join(self.aug_pts_dir, '%06d.bin' % sample_id)
            assert os.path.exists(pts_file), '%s' % pts_file
            aug_pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
            pts_rect, pts_intensity = aug_pts[:, 0:3], aug_pts[:, 3]

        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]

        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN':
            # all labels for checking overlapping
            all_gt_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
            all_gt_boxes3d = kitti_utils.objs_to_boxes3d(all_gt_obj_list)

            gt_aug_flag = False
            if np.random.rand() < cfg.GT_AUG_APPLY_PROB:
                # augment one scene
                gt_aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = \
                    self.apply_gt_aug_to_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)

        # generate inputs
        if self.mode == 'TRAIN' or self.random_select:
            if self.npoints < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if self.npoints > len(pts_rect):
                    # print(len(pts_rect))
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
        else:
            ret_pts_rect = pts_rect
            ret_pts_intensity = pts_intensity - 0.5

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        sample_info = {'sample_id': sample_id, 'random_select': self.random_select}
        gt_box_list = self.filtrate_objects(self.get_label(sample_id))

        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN' and gt_aug_flag:
            gt_box_list.extend(extra_gt_obj_list)
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_box_list)

        gt_alpha = np.zeros((gt_box_list.__len__()), dtype=np.float32)
        for k, box in enumerate(gt_box_list):
            gt_alpha[k] = box.src["alpha"]

        # data augmentation
        aug_pts_rect = ret_pts_rect.copy()
        aug_gt_boxes3d = gt_boxes3d.copy()
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            aug_pts_rect, aug_gt_boxes3d, aug_method = self.data_augmentation(aug_pts_rect, aug_gt_boxes3d, gt_alpha,
                                                                              sample_id)
            sample_info['aug_method'] = aug_method

        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = aug_pts_rect

        # generate training labels
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(aug_pts_rect, aug_gt_boxes3d)
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = aug_pts_rect
        sample_info['pts_features'] = ret_pts_features
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = aug_gt_boxes3d
        return sample_info

    @staticmethod
    def generate_rpn_training_labels(pts_rect, gt_boxes3d):
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928
                                                                    # center3d yv fg_pts_rect 维度一样?
            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # ry

        return cls_label, reg_label

    def rotate_box3d_along_y(self, box3d, rot_angle):
        old_x, old_z, ry = box3d[0], box3d[2], box3d[6]
        old_beta = np.arctan2(old_z, old_x)
        alpha = -np.sign(old_beta) * np.pi / 2 + old_beta + ry

        box3d = kitti_utils.rotate_pc_along_y(box3d.reshape(1, 7), rot_angle=rot_angle)[0]
        new_x, new_z = box3d[0], box3d[2]
        new_beta = np.arctan2(new_z, new_x)
        box3d[6] = np.sign(new_beta) * np.pi / 2 + alpha - new_beta

        return box3d

    def apply_gt_aug_to_one_scene(self, sample_id, pts_rect, pts_intensity, all_gt_boxes3d):
        """
        :param pts_rect: (N, 3)
        :param all_gt_boxex3d: (M2, 7)
        :return:
        """
        assert self.gt_database is not None
        # extra_gt_num = np.random.randint(10, 15)
        # try_times = 50
        if cfg.GT_AUG_RAND_NUM:
            extra_gt_num = np.random.randint(10, cfg.GT_EXTRA_NUM)
        else:
            extra_gt_num = cfg.GT_EXTRA_NUM
        try_times = 100
        cnt = 0
        cur_gt_boxes3d = all_gt_boxes3d.copy()
        cur_gt_boxes3d[:, 4] += 0.5  # TODO: consider different objects
        cur_gt_boxes3d[:, 5] += 0.5  # enlarge new added box to avoid too nearby boxes
        cur_gt_corners = kitti_utils.boxes3d_to_corners3d(cur_gt_boxes3d)

        extra_gt_obj_list = []
        extra_gt_boxes3d_list = []
        new_pts_list, new_pts_intensity_list = [], []
        src_pts_flag = np.ones(pts_rect.shape[0], dtype=np.int32)

        while try_times > 0:
            if cnt > extra_gt_num:
                break

            try_times -= 1
            if cfg.GT_AUG_HARD_RATIO > 0:
                p = np.random.rand()
                if p > cfg.GT_AUG_HARD_RATIO:
                    # use easy sample
                    rand_idx = np.random.randint(0, len(self.gt_database[0]))
                    new_gt_dict = self.gt_database[0][rand_idx]
                else:
                    # use hard sample
                    rand_idx = np.random.randint(0, len(self.gt_database[1]))
                    new_gt_dict = self.gt_database[1][rand_idx]
            else:
                rand_idx = np.random.randint(0, self.gt_database.__len__())
                new_gt_dict = self.gt_database[rand_idx]

            new_gt_box3d = new_gt_dict['gt_box3d'].copy()
            new_gt_points = new_gt_dict['points'].copy()
            new_gt_intensity = new_gt_dict['intensity'].copy()
            new_gt_obj = new_gt_dict['obj']
            center = new_gt_box3d[0:3]
            if cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(center) is False):
                continue

            if new_gt_points.__len__() < 5:  # too few points
                continue

            new_enlarged_box3d = new_gt_box3d.copy()
            new_enlarged_box3d[4] += 0.5
            new_enlarged_box3d[5] += 0.5  # enlarge new added box to avoid too nearby boxes

            cnt += 1
            new_corners = kitti_utils.boxes3d_to_corners3d(new_enlarged_box3d.reshape(1, 7))
            iou3d = kitti_utils.get_iou3d(new_corners, cur_gt_corners)
            valid_flag = iou3d.max() < 1e-8
            if not valid_flag:
                continue

            enlarged_box3d = new_gt_box3d.copy()
            enlarged_box3d[3] += 2  # remove the points above and below the object

            boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(
                torch.from_numpy(pts_rect), torch.from_numpy(enlarged_box3d.reshape(1, 7)))
            pt_mask_flag = (boxes_pts_mask_list[0].numpy() == 1)
            src_pts_flag[pt_mask_flag] = 0  # remove the original points which are inside the new box

            new_pts_list.append(new_gt_points)
            new_pts_intensity_list.append(new_gt_intensity)
            cur_gt_boxes3d = np.concatenate((cur_gt_boxes3d, new_enlarged_box3d.reshape(1, 7)), axis=0)
            cur_gt_corners = np.concatenate((cur_gt_corners, new_corners), axis=0)
            extra_gt_boxes3d_list.append(new_gt_box3d.reshape(1, 7))
            extra_gt_obj_list.append(new_gt_obj)

        if new_pts_list.__len__() == 0:
            return False, pts_rect, pts_intensity, None, None

        extra_gt_boxes3d = np.concatenate(extra_gt_boxes3d_list, axis=0)
        # remove original points and add new points
        pts_rect = pts_rect[src_pts_flag == 1]
        pts_intensity = pts_intensity[src_pts_flag == 1]
        new_pts_rect = np.concatenate(new_pts_list, axis=0)
        new_pts_intensity = np.concatenate(new_pts_intensity_list, axis=0)
        pts_rect = np.concatenate((pts_rect, new_pts_rect), axis=0)
        pts_intensity = np.concatenate((pts_intensity, new_pts_intensity), axis=0)

        return True, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list

    def data_augmentation(self, aug_pts_rect, aug_gt_boxes3d, gt_alpha, sample_id=None, mustaug=False, stage=1):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            aug_pts_rect = kitti_utils.rotate_pc_along_y(aug_pts_rect, rot_angle=angle)
            if stage == 1:
                # xyz change, hwl unchange
                aug_gt_boxes3d = kitti_utils.rotate_pc_along_y(aug_gt_boxes3d, rot_angle=angle)

                # calculate the ry after rotation
                x, z = aug_gt_boxes3d[:, 0], aug_gt_boxes3d[:, 2]
                beta = np.arctan2(z, x)
                new_ry = np.sign(beta) * np.pi / 2 + gt_alpha - beta
                aug_gt_boxes3d[:, 6] = new_ry  # TODO: not in [-np.pi / 2, np.pi / 2]
            elif stage == 2:
                # for debug stage-2, this implementation has little float precision difference with the above one
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0] = self.rotate_box3d_along_y(aug_gt_boxes3d[0], angle)
                aug_gt_boxes3d[1] = self.rotate_box3d_along_y(aug_gt_boxes3d[1], angle)
            else:
                raise NotImplementedError

            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect = aug_pts_rect * scale
            aug_gt_boxes3d[:, 0:6] = aug_gt_boxes3d[:, 0:6] * scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            if stage == 1:
                aug_gt_boxes3d[:, 6] = np.sign(aug_gt_boxes3d[:, 6]) * np.pi - aug_gt_boxes3d[:, 6]
            elif stage == 2:
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0, 6] = np.sign(aug_gt_boxes3d[0, 6]) * np.pi - aug_gt_boxes3d[0, 6]
                aug_gt_boxes3d[1, 6] = np.sign(aug_gt_boxes3d[1, 6]) * np.pi - aug_gt_boxes3d[1, 6]
            else:
                raise NotImplementedError

            aug_method.append('flip')

        return aug_pts_rect, aug_gt_boxes3d, aug_method

    def collate_batch(self, batch):
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue

            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)], axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict


if __name__ == '__main__':
    pass
