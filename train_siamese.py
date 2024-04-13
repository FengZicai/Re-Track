import tools._init_path
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import argparse
import logging
from functools import partial

from lib.net.point_rcnn import PointRCNN
import lib.net.train_functions as train_functions
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.config import cfg, cfg_from_file, save_config_to_file
import tools.train_utils.train_utils as train_utils
from tools.train_utils.fastai_optim import OptimWrapper
from tools.train_utils import learning_schedules_fastai as lsf


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--dataset', type=str, default='KITTI', help='ARGO or KITTI')
parser.add_argument('--cfg_file', type=str, default='cfgs/default.yaml', help='specify the config for training')
parser.add_argument("--train_mode", type=str, default='rpn', required=True, help="specify the training mode")
parser.add_argument("--batch_size", type=int, default=16, required=True, help="batch size for training")
parser.add_argument("--epochs", type=int, default=200, required=True, help="Number of epochs to train for")

parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
parser.add_argument("--ckpt_save_interval", type=int, default=5, help="number of training epochs")
parser.add_argument('--output_dir', type=str, default=None, help='specify an output directory if needed')
parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
parser.add_argument('--GPU', required=False, type=int, default=-1, help='ID of the GPU to use')
parser.add_argument("--ckpt", type=str, default=None, help="continue training from this checkpoint")
parser.add_argument("--gt_database", type=str, default='gt_database/train_gt_database_3level_Car.pkl',
                    help='generated gt database for augmentation')
parser.add_argument('--train_with_eval', action='store_true', default=False, help='whether to train with evaluation')

# 跟踪输入的参数
parser.add_argument('--regress', required=False, type=str, default='IoU', help='how to regress (IoU/Gaussian)')
parser.add_argument('--sigma_Gaussian', required=False, type=float, default=1, help='Gaussian distance variation sigma for regression in training')
parser.add_argument('--offset_BB', required=False, type=float, default=0, help='offset around the BB in meters')
parser.add_argument('--scale_BB', required=False, type=float, default=1.25, help='scale of the BB before cropping')
parser.add_argument('--inputsize', required=False, type=float, default=2048, help='inputsize of the model')
parser.add_argument('--bneck_size', required=False, type=int, default=128, help='Size of the bottleneck')
parser.add_argument("--chkpt_file", type=str, default=None,
                    help="specify the well-trained S3D checkpoint")
parser.add_argument("--finetune", type=str, default=None, help="specify the well-trained S3D checkpoint")
parser.add_argument('--lambda_completion', required=False, type=float, default=1e-6, help='lambda ratio for completion loss')
parser.add_argument('--DenseAutoEncoder', action='store_true', default=False, help='use proposed model')
args = parser.parse_args()

if 'ARGO' in args.dataset.upper():
    from argo.rpn_train_dataset import Train_RPN  # ARGO
    from argo.siamese_offline_dataset import SiameseTrain
else:
    from tools.rpn_train_dataset import Train_RPN  # ARGO
    from tools.siamese_offline_dataset import SiameseTrain

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def create_dataloader(logger):
    if 'ARGO' in args.dataset.upper():
        DATA_PATH = '/data/Argo/argoverse-tracking/train_kitti'  # ARGO
    else:
        DATA_PATH = os.path.join('/data/3DTracking/data', 'training')


    # create dataloader
    if args.train_mode == 'rpn':
        train_set = Train_RPN(path=DATA_PATH, split=cfg.TRAIN.SPLIT, category_name=cfg.CLASSES,
                              npoints=cfg.RPN.NUM_POINTS, mode='TRAIN', logger=logger)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True,
                                  num_workers=args.workers, shuffle=True, collate_fn=train_set.collate_batch,
                                  drop_last=True)
        if args.train_with_eval:  # whether to train with evaluation
            Validation_set = Train_RPN(path=DATA_PATH, npoints=cfg.RPN.NUM_POINTS, split=cfg.TRAIN.VAL_SPLIT,
                                 mode='EVAL', logger=logger, category_name=cfg.CLASSES)  #未修改
            Validation_loader = DataLoader(Validation_set, batch_size=1, shuffle=True, pin_memory=True,
                                     num_workers=args.workers, collate_fn=Validation_set.collate_batch)
        else:
            Validation_loader = None
    elif args.train_mode == 'siamese':
        train_set = SiameseTrain(model_inputsize=args.inputsize, path=DATA_PATH,
                                 split=cfg.TRAIN.SPLIT, category_name=cfg.CLASSES, mode='TRAIN',
                                 regress=args.regress, sigma_Gaussian=args.sigma_Gaussian,
                                 offset_BB=args.offset_BB, scale_BB=args.scale_BB, logger=logger)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True)
        if args.train_with_eval:
            Validation_set = SiameseTrain(model_inputsize=args.inputsize, path=DATA_PATH,
                                          split=cfg.TRAIN.VAL_SPLIT, category_name=cfg.CLASSES, mode='EVAL',
                                          regress=args.regress, sigma_Gaussian=args.sigma_Gaussian,
                                          offset_BB=args.offset_BB, scale_BB=args.scale_BB, logger=logger)
            Validation_loader = DataLoader(Validation_set, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers, pin_memory=True)
        else:
            Validation_loader = None
    else:
        raise NotImplementedError

    return train_loader, Validation_loader


def create_optimizer(model):

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                              momentum=cfg.TRAIN.MOMENTUM)
    elif cfg.TRAIN.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=cfg.TRAIN.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )

        # fix rpn: do this since we use costomized optimizer.step
        if cfg.RPN.ENABLED and cfg.RPN.FIXED:
            for param in model.rpn.parameters():
                param.requires_grad = False
    else:
        raise NotImplementedError

    return optimizer


def create_scheduler(optimizer, total_steps, last_epoch):
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.LR_DECAY
        return max(cur_decay, cfg.TRAIN.LR_CLIP / cfg.TRAIN.LR)

    def bnm_lmbd(cur_epoch):
        cur_decay = 1
        for decay_step in cfg.TRAIN.BN_DECAY_STEP_LIST:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * cfg.TRAIN.BN_DECAY
        return max(cfg.TRAIN.BN_MOMENTUM * cur_decay, cfg.TRAIN.BNM_CLIP)

    if cfg.TRAIN.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = lsf.OneCycle(
            optimizer, total_steps, cfg.TRAIN.LR, list(cfg.TRAIN.MOMS), cfg.TRAIN.DIV_FACTOR, cfg.TRAIN.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    bnm_scheduler = train_utils.BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return lr_scheduler, bnm_scheduler


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


if __name__ == "__main__":
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.TAG = os.path.splitext(os.path.basename(args.cfg_file))[0]

    if args.train_mode == 'rpn':
        cfg.RPN.ENABLED = True
        cfg.RCNN.ENABLED = False
        root_result_dir = os.path.join('../', 'output', 'rpn', cfg.TAG)
    elif args.train_mode == 'siamese':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = False
        cfg.RPN.FIXED = False
        root_result_dir = os.path.join('../', 'output', 'siamese', cfg.TAG)
    elif args.train_mode == 'joint':
        cfg.RCNN.ENABLED = True
        cfg.RPN.ENABLED = True
        cfg.RPN.FIXED = True
        root_result_dir = os.path.join('../', 'output', 'joint', cfg.TAG)
    else:
        raise NotImplementedError

    if args.output_dir is not None:
        root_result_dir = args.output_dir
    os.makedirs(root_result_dir, exist_ok=True)

    log_file = os.path.join(root_result_dir, 'log_train.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')

    # log to file
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
        # logger.info('CUDA_VISIBLE_DEVICES=%s' % os.environ["CUDA_DEVICE_ORDER"])



    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))

    save_config_to_file(cfg, logger=logger)

    # copy important files to backup
    backup_dir = os.path.join(root_result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp *.py %s/' % backup_dir)
    os.system('cp ../tools/*.py %s/' % backup_dir)
    os.system('cp ../lib/net/*.py %s/' % backup_dir)
    os.system('cp ../lib/datasets/kitti_rcnn_dataset.py %s/' % backup_dir)

    # tensorboard log
    tb_log = SummaryWriter(log_dir=os.path.join(root_result_dir, 'tensorboard'))

    # create dataloader & network & optimizer
    train_loader, Validation_loader = create_dataloader(logger)
    model = PointRCNN(num_classes=train_loader.dataset.num_class, use_xyz=True, mode='TRAIN',
                      bneck_size=args.bneck_size, DenseAutoEncoder=args.DenseAutoEncoder)

    optimizer = create_optimizer(model)


    model.cuda()

    if args.mgpus:
        model = nn.DataParallel(model)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        load_part_ckpt(model, args.ckpt, logger)


    lr_scheduler, bnm_scheduler = create_scheduler(optimizer, total_steps=len(train_loader) * args.epochs,
                                                   last_epoch=last_epoch)

    if cfg.TRAIN.LR_WARMUP and cfg.TRAIN.OPTIMIZER != 'adam_onecycle':
        lr_warmup_scheduler = train_utils.CosineWarmupLR(optimizer, T_max=cfg.TRAIN.WARMUP_EPOCH * len(train_loader), eta_min=cfg.TRAIN.WARMUP_MIN)
    else:
        lr_warmup_scheduler = None

    # start training
    logger.info('**********************Start training**********************')
    ckpt_dir = os.path.join(root_result_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer = train_utils.Trainer(model, train_functions.model_joint_fn_decorator(), optimizer, ckpt_dir=ckpt_dir,
                                  lr_scheduler=lr_scheduler, bnm_scheduler=bnm_scheduler, model_fn_eval=train_functions.model_joint_fn_decorator(),
                                  tb_log=tb_log, eval_frequency=1, lr_warmup_scheduler=lr_warmup_scheduler, warmup_epoch=cfg.TRAIN.WARMUP_EPOCH,
                                  grad_norm_clip=cfg.TRAIN.GRAD_NORM_CLIP, lambda_completion=args.lambda_completion)

    trainer.train(it, start_epoch, args.epochs, train_loader, Validation_loader,
                  ckpt_save_interval=args.ckpt_save_interval,
                  lr_scheduler_each_iter=(cfg.TRAIN.OPTIMIZER == 'adam_onecycle'))

    logger.info('**********************End training**********************')
