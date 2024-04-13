'''
the dataloder for the offline train

'''

from torch.utils.data import Dataset
from tools.data_classes import PointCloud, Box
import logging
import pickle
import numpy as np
import pandas as pd
import os
import torch
from pyquaternion import Quaternion
from tqdm import tqdm
import tools.utils as utils
from tools.utils import getModel
from tools.searchspace import KalmanFiltering

from PIL import Image
from lib.config import cfg
import lib.utils.calibration_KITTI as calibration
import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils

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

class TrainRSIAMDataset(Dataset):

    def __init__(self, model_inputsize, path, split, category_name="Car",
                 regress="GAUSSIAN", if_train_rpn=False, mode='TRAIN'):

        if category_name == 'Car':
            self.classes = ('Background', 'Car')
        elif category_name == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif category_name == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
        elif category_name == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
        else:
            assert False, "Invalid classes: %s" % category_name
        self.num_class = self.classes.__len__()

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        self.dataset = kittiDataset(path=path)
        self.model_inputsize = model_inputsize
        self.split = split
        self.sceneID = self.dataset.getSceneID(split=split)
        self.getBBandPC = self.dataset.getBBandPC

        self.category_name = category_name
        self.regress = regress

        self.list_of_tracklet_anno, _ = self.dataset.getListOfAnno(
            self.sceneID, category_name, if_train_rpn)
        self.list_of_anno = [
            anno for tracklet_anno in self.list_of_tracklet_anno
            for anno in tracklet_anno
        ]

        if 'TRAIN' in self.split.upper():
            self.samples = 'roisampled48' #'rcnnresult'
        elif 'VALID' in self.split.upper():
            self.samples = 'roisampled48'
        elif 'TEST' in self.split.upper():
            self.samples = 'rpnpred147'
        else:
            raise ModuleNotFoundError
        self.KITTI_pred = os.path.join(self.dataset.KITTI_Folder, self.samples)
        self.dict_of_predscene_anno = self.getlistofperedanno()

    def getlistofperedanno(self):
        split_dir = os.path.join('/data/3DTracking/intermediate', 'pred' + self.split + '.txt')
        list_of_scene = [x.strip() for x in open(split_dir).readlines()]
        dict_of_scene_anno = {}
        for index in range(len(list_of_scene)):
            scene = list_of_scene[index]
            label_file = os.path.join(self.KITTI_pred, scene + '.txt')
            df = pd.read_csv(
                    label_file,
                    sep=' ',
                    names=["type", "truncated", "occluded",
                        "alpha", "bbox_left", "bbox_top", "bbox_right",
                        "bbox_bottom", "height", "width", "length", "x", "y", "z",
                        "rotation_y", "score"
                    ])
            df.insert(loc=0, column="frame", value=int(scene[-6:]))
            df.insert(loc=0, column="scene", value=scene[:4])
            list_of_scene_anno = [anno for index, anno in df.iterrows()]
            dict_of_scene_anno[scene[:4] + scene[-6:]] = list_of_scene_anno

        return dict_of_scene_anno

    def isTiny(self):
        return ("TINY" in self.split.upper())

    def __getitem__(self, index):
        return self.getitem(index)

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

    def check_pc_range(self, xyz):
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


class SiameseTrain(TrainRSIAMDataset):
    def __init__(self, model_inputsize, path, split="", category_name="Car", mode=None,
                 regress="GAUSSIAN", sigma_Gaussian=1, offset_BB=0, scale_BB=1.0, logger=None):

        super().__init__(model_inputsize=model_inputsize, path=path, split=split,
                         category_name=category_name, regress=regress, mode=mode)
        #PR
        self.logger = logger

        #S3D
        self.sigma_Gaussian = sigma_Gaussian
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB
        self.packed = True
        self.savedir = '/data/3DTracking/tools/gt_database/'

        self.logger.info("preloading PC...")
        os.makedirs(self.savedir, exist_ok=True)
        self.save_PC_name = os.path.join(self.savedir, '%s_siamese_gt_database_%s_%s.pkl' % (self.split + '_' + self.mode, self.category_name, 'pc'))
        if not os.path.exists(self.save_PC_name):
            self.packed = False
        self.save_BB_name = os.path.join(self.savedir, '%s_siamese_gt_database_%s_%s.pkl' % (self.split + '_' + self.mode, self.category_name, 'bb'))

        if not self.packed:
            self.list_of_PCs = []
            self.list_of_BBs = []
            self.list_of_vaild_anno = []
            for index in tqdm(range(len(self.list_of_anno))):
                anno = self.list_of_anno[index]
                PC, box, calib_file = self.getBBandPC(anno)
                if PC.points.size == 3:
                    print('the Point cloud is missing')
                    continue
                calib = calibration.Calibration(calib_file)
                pts_rect = calib.lidar_to_rect(PC.points.T)
                pts_intensity = PC.intensity.T
                PC = PointCloud(np.vstack((pts_rect.T, pts_intensity.T)))
                new_PC = utils.cropPC(PC, box, offset=10)
                self.list_of_PCs.append(new_PC)
                self.list_of_BBs.append(box)
                self.list_of_vaild_anno.append(anno)
            self.logger.info("PC preloaded!")
            with open(self.save_PC_name, 'wb') as f:
                pickle.dump(self.list_of_PCs, f)
            self.logger.info("BB preloaded!")
            with open(self.save_BB_name, 'wb') as f:
                pickle.dump(self.list_of_BBs, f)
        else:
            self.logger.info("Packed PC preloaded!")
            self.list_of_PCs = pickle.load(open(self.save_PC_name, 'rb'))
            self.logger.info("Packed BB preloaded!")
            self.list_of_BBs = pickle.load(open(self.save_BB_name, 'rb'))

        self.logger.info("preloading Model..")
        self.save_Model_name = os.path.join(self.savedir, '%s_siamese_gt_database_%s_%s.pkl' % (self.split + '_' + self.mode, self.category_name, 'Model'))
        self.save_trackanno_name = os.path.join(self.savedir, '%s_siamese_gt_database_%s_%s.pkl' % (self.split + '_' + self.mode, self.category_name, 'trackanno'))
        self.save_anno_name = os.path.join(self.savedir, '%s_siamese_gt_database_%s_%s.pkl' % (self.split + '_' + self.mode, self.category_name, 'anno'))

        if not self.packed:
            self.model_PC = [None] * len(self.list_of_tracklet_anno)
            for i in tqdm(range(len(self.list_of_tracklet_anno))):
                list_of_anno = self.list_of_tracklet_anno[i]
                PCs = []
                BBs = []
                cnt = 0
                for anno in list_of_anno:
                    this_PC, this_BB, calib_file = self.getBBandPC(anno)
                    if this_PC.points.size == 3:
                        print('the Point cloud is missing')
                        continue
                    calib = calibration.Calibration(calib_file)
                    this_PC.points = calib.lidar_to_rect(this_PC.points.T).T
                    PCs.append(this_PC)
                    BBs.append(this_BB)
                    anno["model_idx"] = i
                    anno["relative_idx"] = cnt
                    cnt += 1

                self.model_PC[i] = getModel(PCs, BBs, offset=self.offset_BB, scale=self.scale_BB)

            with open(self.save_anno_name, 'wb') as f:
                pickle.dump(self.list_of_vaild_anno, f)
            self.logger.info("Model preloaded!")
            with open(self.save_Model_name, 'wb') as f:
                pickle.dump(self.model_PC, f)
            with open(self.save_trackanno_name, 'wb') as f:
                pickle.dump(self.list_of_tracklet_anno, f)
        else:
            self.list_of_vaild_anno = pickle.load(open(self.save_anno_name, 'rb'))
            self.list_of_tracklet_anno = pickle.load(open(self.save_trackanno_name, 'rb'))
            self.logger.info("Packed Model preloaded!")
            self.model_PC = pickle.load(open(self.save_Model_name, 'rb'))

        self.logger.info("preloading Sample..")
        self.save_sample_PC_name = os.path.join(self.savedir, ' %s_siamese_gt_database_%s_%s.pkl' % (
            self.split + '_' + self.mode, self.category_name, 'Sample_PC'))
        self.save_sample_BB_name = os.path.join(self.savedir, ' %s_siamese_gt_database_%s_%s.pkl' % (
            self.split + '_' + self.mode, self.category_name, 'Sample_BB'))
        if not self.packed:
            self.sample_PC = {}
            self.sample_BB = {}
            for key, values in self.dict_of_predscene_anno.items():
                list_sample_PC = []
                list_sample_BB = []
                for index in tqdm(range(len(values))):
                    anno = values[index]
                    sample_PC, sample_BB, calib_file = self.getBBandPC(anno)
                    list_sample_BB.append(sample_BB)
                    if sample_PC.points.size == 3:
                        print('the Point cloud is missing')
                        continue
                    if anno["scene"] + f'{anno["frame"]:06}' in self.sample_PC.keys():
                        continue
                    calib = calibration.Calibration(calib_file)
                    sample_PC.points = calib.lidar_to_rect(sample_PC.points.T).T
                    sample_PC = utils.cropAndCenterPC(sample_PC, sample_BB, offset=0, scale=1.25)
                    list_sample_PC.append(sample_PC)
                self.sample_PC[key] = list_sample_PC
                self.sample_BB[key] = list_sample_BB

            self.logger.info("Sample preloaded!")
            with open(self.save_sample_PC_name, 'wb') as f:
                pickle.dump(self.sample_PC, f)
            with open(self.save_sample_BB_name, 'wb') as f:
                pickle.dump(self.sample_BB, f)
        else:
            self.sample_PC = pickle.load(open(self.save_sample_PC_name, 'rb'))
            self.sample_BB = pickle.load(open(self.save_sample_BB_name, 'rb'))
            self.logger.info("Packed Sample preloaded!")

    def __getitem__(self, index):
        return self.getitem(index)

    def getPCandBBfromIndex(self, anno_idx):
        this_PC = self.list_of_PCs[anno_idx]
        this_BB = self.list_of_BBs[anno_idx]
        return this_PC, this_BB

    def getitem(self, index):

        this_anno = self.list_of_vaild_anno[index]
        sample_id = this_anno["scene"] + f'{this_anno["frame"]:06}'

        this_PC, this_BB = self.getPCandBBfromIndex(index)

        if sample_id not in self.sample_PC.keys():
            return self.getitem(np.random.randint(0, self.__len__()))

        sample_PCs = self.sample_PC[sample_id]
        sample_BBs = self.sample_BB[sample_id]
        sample_PCs_reg = [
            utils.regularizePC(PC, self.model_inputsize)
            for PC in sample_PCs
        ]

        sample_PCs_torch = torch.cat(sample_PCs_reg, dim=0)

        model_idx = this_anno["model_idx"]
        model_PC = self.model_PC[model_idx]
        model_PC = utils.regularizePC(model_PC, self.model_inputsize)[0]

        repeat_shape = np.ones(len(sample_PCs_torch.shape), dtype=np.int32)
        repeat_shape[0] = len(sample_PCs_torch)
        model_PC_encoded = model_PC.repeat(tuple(repeat_shape))

        if "IOU" in self.regress.upper():
            scores = [utils.getScoreIoU(this_BB, sample_BB) for sample_BB in sample_BBs]
            score = torch.cat(scores, dim=0)
        elif "HINGE" in self.regress.upper():
            scores = [utils.getScoreHingeIoU(this_BB, sample_BB) for sample_BB in sample_BBs]
            score = torch.cat(scores, dim=0)
        elif "Gaussian" in self.regress.upper():
            scores = [utils.distanceBB_Gaussian(this_BB, sample_BB) for sample_BB in sample_BBs]
            score = torch.cat(scores, dim=0)
        else:
            raise ModuleNotFoundError
        sample_info = {'sample_PC': sample_PCs_torch, 'model_PC': model_PC_encoded, 'score': score}
        return sample_info

    def __len__(self):
        return len(self.list_of_vaild_anno)


class SiameseTest(TrainRSIAMDataset):
    def __init__(self, model_inputsize, path, split="", category_name="Car", mode=None,
                 regress="GAUSSIAN", sigma_Gaussian=1, offset_BB=0, scale_BB=1.0, logger=None):

        super().__init__(model_inputsize=model_inputsize, path=path, split=split,
                         category_name=category_name, regress=regress, mode=mode)
        #PR
        self.logger = logger

        #S3D
        self.sigma_Gaussian = sigma_Gaussian
        self.offset_BB = offset_BB
        self.scale_BB = scale_BB
        self.packed = True
        self.savedir = '/data/3DTracking/tools/gt_database'

        self.logger.info("preloading Sample..")
        self.save_sample_PC_name = os.path.join(self.savedir, ' %s_siamese_gt_database_%s_%s.pkl' % (
            self.split + '_' + self.mode, self.category_name, 'Sample_PC_147'))
        if not os.path.exists(self.save_sample_PC_name):
            self.packed = False
        self.save_sample_BB_name = os.path.join(self.savedir, ' %s_siamese_gt_database_%s_%s.pkl' % (
            self.split + '_' + self.mode, self.category_name, 'Sample_BB_147'))
        if not self.packed:
            self.sample_PC = {}
            self.sample_BB = {}
            for key, values in self.dict_of_predscene_anno.items():
                list_sample_PC = []
                list_sample_BB = []
                for index in tqdm(range(len(values))):
                    anno = values[index]
                    sample_PC, sample_BB, calib_file = self.getBBandPC(anno)
                    if sample_PC.points.size == 3:
                        print('the Point cloud is missing')
                        continue
                    assert key == anno["scene"] + f'{anno["frame"]:06}'
                    calib = calibration.Calibration(calib_file)
                    sample_PC.points = calib.lidar_to_rect(sample_PC.points.T).T
                    sample_PC = utils.cropAndCenterPC(sample_PC, sample_BB, offset=0, scale=1.25)
                    list_sample_PC.append(sample_PC)
                    list_sample_BB.append(sample_BB)

                self.sample_PC[key] = list_sample_PC
                self.sample_BB[key] = list_sample_BB

            with open(self.save_sample_PC_name, 'wb') as f:
                pickle.dump(self.sample_PC, f)
            with open(self.save_sample_BB_name, 'wb') as f:
                pickle.dump(self.sample_BB, f)
        else:
            self.sample_PC = pickle.load(open(self.save_sample_PC_name, 'rb'))
            self.sample_BB = pickle.load(open(self.save_sample_BB_name, 'rb'))
            self.logger.info("Packed Sample preloaded!")

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index):
        sample_info = {}
        list_of_anno = self.list_of_tracklet_anno[index]
        list_of_Sample_PC = []
        list_of_Sample_BB = []
        list_of_PCs = []
        list_of_BBs = []
        list_of_sample_id = []
        list_of_valid_anno = []

        for anno in list_of_anno:
            PC, box, calib_file = self.getBBandPC(anno)
            if PC.points.size == 3:
                print('the Point cloud is missing')
                continue

            box_list = [box]
            gt_box_list = self.filtrate_objects(box_list)
            if len(gt_box_list) == 0:
                continue

            sample_id = anno["scene"] + f'{anno["frame"]:06}'
            if sample_id not in self.sample_PC.keys():
                continue
            else:
                sample_PC = self.sample_PC[sample_id]
            calib = calibration.Calibration(calib_file)
            pts_rect = calib.lidar_to_rect(PC.points.T)
            pts_intensity = PC.intensity.T
            PC = PointCloud(np.vstack((pts_rect.T, pts_intensity.T)))

            sample_BB = self.sample_BB[sample_id]

            list_of_sample_id.append(sample_id)
            list_of_valid_anno.append(anno)
            list_of_PCs.append(PC)
            list_of_BBs.append(box)
            list_of_Sample_PC.append(sample_PC)
            list_of_Sample_BB.append(sample_BB)

        sample_info['sample_id'] = list_of_sample_id
        sample_info['tracklet_anno'] = list_of_valid_anno
        sample_info['list_of_pc'] = list_of_PCs
        sample_info['list_of_BB'] = list_of_BBs
        sample_info['list_of_Sample_PC'] = list_of_Sample_PC
        sample_info['list_of_Sample_BB'] = list_of_Sample_BB
        return sample_info

    def __len__(self):
        return len(self.list_of_tracklet_anno)



