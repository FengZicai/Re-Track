This is the implementation of "A Novel Object Re-Track Framework for 3D Point Clouds", which is accepted by the ACM International Conference on Multimedia 2020 (ACM-MM 2020).

This work is based on the following works:

* [Leveraging Shape Completion for 3D Siamese Tracking](https://arxiv.org/pdf/1903.01784.pdf)

* [PointRCNN：3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/abs/1812.04244)

* [Efficient Tracking Proposals using 2D-3D Siamese Networks on LIDAR](https://arxiv.org/pdf/1903.10168v1.pdf)

## Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 16.04)
* Python 3.6+
* PyTorch 1.1

Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.

Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## Download KITTI Tracking dataset

Download the dataset from [KITTI Tracking](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

You will need to download the data for
[velodyne](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip), 
[calib](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip) and
[label_02](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip).


Place the 3 folders in the same parent folder as following:
```
[Parent Folder]
--> [calib]
    --> {0000-0020}.txt
--> [label_02]
    --> {0000-0020}.txt
--> [velodyne]
    --> [0000-0020] folders with velodynes .bin files
```
For convenience,you can use a soft link to link data to this project.
```angular2
ln -s /media/fengzicai/fzc/KITTI_TRACKING/training/  data/
```
 


## Train
* this code is based on [pointrcnn](https://github.com/sshaoshuai/PointRCNN) and [ShapeCompletion3DTracking](https://github.com/SilvioGiancola/ShapeCompletion3DTracking). The training of RPN stage is the same as the stage of PointRCNN, please refer to [pointrcnn](https://github.com/sshaoshuai/PointRCNN)

   
### Training of Siamese stage


```
python train_siamese.py --dataset KITTI --cfg_file /data/3DTracking/tools/cfgs/default.yaml --batch_size 1 --train_mode siamese --epochs 70 --ckpt_save_interval 2 --DenseAutoEncoder
```


## eval

Test the RPN model, and the generated predictions will be used for training the Siamese stage.

```angular2
python eval_siamese.py
--cfg_file
cfgs/default.yaml
--eval_mode
rpn
--dataset_path=/media/fengzicai/fzc/KITTI_TRACKING/training
--rpn_ckpt
/media/fengzicai/fzc/PointRCNN/output/rpn/default/ckpt/checkpoint_epoch_200.pth
--batch_size
16
--save_rpn_feature
--set
TEST.RPN_POST_NMS_TOP_N
147
```


Used to test and evaluate Siamese, calculate Success, Precision


```angular2
python eval_siamese.py
--dataset 
KITTI 
--cfg_file 
/data/3DTracking/tools/cfgs/default.yaml 
--eval_mode 
siameseoffline 
--dataset_path=/data/3DTracking/data/training 
--siam_ckpt /data/3DTracking/models/checkpoint_epoch_14.pth 
--DenseAutoEncoder 
--GPU 0 
```


## Citation

If you find the code useful in your research, please consider citing our [paper](https://drive.google.com/file/d/1D1igPaerMVD-hV6W6OjwiCZ0560iN-2l/view):

```
@inproceedings{feng2020novel,
  title={A novel object re-track framework for 3D point clouds},
  author={Feng, Tuo and Jiao, Licheng and Zhu, Hao and Sun, Long},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={3118--3126},
  year={2020}
}
```




