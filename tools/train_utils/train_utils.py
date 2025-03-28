import logging
import os
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

import torch.optim.lr_scheduler as lr_sched
import math
from lib.config import cfg
from tools.PCLosses import ChamferLoss
from tools.metrics import AverageMeter, Success, Precision, Accuracy_Completeness
from tools.metrics import estimateOverlap, estimateAccuracy
import tqdm
import time

logging.getLogger(__name__).addHandler(logging.StreamHandler())
cur_logger = logging.getLogger(__name__)


def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model=None, optimizer=None, filename='checkpoint', logger=cur_logger):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return it, epoch


def load_part_ckpt(model, filename, logger=cur_logger, total_keys=-1):
    if os.path.isfile(filename):
        logger.info("==> Loading part model from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
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


class Trainer(object):
    def __init__(self, model, model_fn, optimizer, ckpt_dir, lr_scheduler, bnm_scheduler,
                 model_fn_eval, tb_log, eval_frequency=1, lr_warmup_scheduler=None, warmup_epoch=-1,
                 grad_norm_clip=1.0, lambda_completion=0):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler, self.model_fn_eval = \
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler, model_fn_eval

        self.lambda_completion = lambda_completion
        self.ckpt_dir = ckpt_dir
        self.eval_frequency = eval_frequency
        self.tb_log = tb_log
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.warmup_epoch = warmup_epoch
        self.grad_norm_clip = grad_norm_clip
        self.regress = "IOU"

    def _train_it(self, batch, lambda_completion, ckpt_dir):
        self.model.train()

        self.optimizer.zero_grad()
        loss, tb_dict, disp_dict = self.model_fn(self.model, batch, lambda_completion, ckpt_dir)

        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss.item(), tb_dict, disp_dict

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = count = 0.0

        # eval one epoch
        for i, data in tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val'):
            self.optimizer.zero_grad()

            loss, tb_dict, disp_dict = self.model_fn_eval(self.model, data)

            total_loss += loss.item()
            count += 1
            for k, v in tb_dict.items():
                eval_dict[k] = eval_dict.get(k, 0) + v

        # statistics this epoch
        for k, v in eval_dict.items():
            eval_dict[k] = eval_dict[k] / max(count, 1)

        cur_performance = 0
        if 'recalled_cnt' in eval_dict:
            eval_dict['recall'] = eval_dict['recalled_cnt'] / max(eval_dict['gt_cnt'], 1)
            cur_performance = eval_dict['recall']
        elif 'iou' in eval_dict:
            cur_performance = eval_dict['iou']

        return total_loss / count, eval_dict, cur_performance

    def validate(self, val_loader, model, criterion_tracking,  criterion_completion, epoch, lambda_completion=0):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_tracking = AverageMeter()
        loss_completion = AverageMeter()
        losses = AverageMeter()
        model.eval()
        with tqdm.tqdm(enumerate(val_loader), total=len(val_loader), ncols=120) as t:
            with torch.no_grad():
                end = time.time()
                for i, batch in t:
                    # measure data loading time
                    data_time.update(time.time() - end)
                    sample_PC, model_PC, score = batch['sample_PC'], batch['model_PC'], batch['score']
                    sample_PC = sample_PC.view(-1, 3, 2048).cuda()
                    model_PC = model_PC.view(-1, 3, 2048).cuda()
                    target = score.float().cuda(non_blocking=True).view(-1)

                    input_data = {'sample_PC': sample_PC, 'model_PC': model_PC, 'target': target}

                    ret_dict = model(input_data)
                    output, Y_AE, target, model_PC = ret_dict['Sim'], ret_dict['Y_AE'], ret_dict['target'], ret_dict['model_PC']

                    if lambda_completion < 1:
                        loss1 = criterion_tracking(output, target)
                    else:
                        loss1 = torch.tensor([0]).float().cuda()

                    if lambda_completion != 0:
                        loss2 = criterion_completion(Y_AE, model_PC)
                    else:
                        loss2 = torch.tensor([0]).float().cuda()
                    siamese_loss = loss1 + lambda_completion * loss2

                    # measure accuracy and record loss
                    loss_tracking.update(loss1.item(), sample_PC.size(0))
                    loss_completion.update(loss2.item(), sample_PC.size(0))
                    losses.update(siamese_loss.item(), sample_PC.size(0))


                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    t.set_description(f'Valid {epoch}: '
                                      f'Time {batch_time.avg:.3f}s '
                                      f'(it:{batch_time.val:.3f}s) '
                                      f'Data:{data_time.avg:.3f}s '
                                      f'(it:{data_time.val:.3f}s) '
                                      f'Loss {losses.avg:.4f} '
                                      f'(tr:{loss_tracking.avg:.4f}, '
                                      f'comp:{loss_completion.avg:.0f})')
        return losses.avg

    def train(self, start_it, start_epoch, n_epochs, train_loader, Validation_loader=None, ckpt_save_interval=5,
              lr_scheduler_each_iter=False):
        eval_frequency = self.eval_frequency if self.eval_frequency > 0 else 1

        it = start_it
        with tqdm.trange(start_epoch, n_epochs, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:

            # keep the best model
            best_loss = 9e99
            os.makedirs(self.ckpt_dir, exist_ok=True)

            for epoch in tbar:
                if self.lr_scheduler is not None and self.warmup_epoch <= epoch and (not lr_scheduler_each_iter):
                    self.lr_scheduler.step(epoch)

                if self.bnm_scheduler is not None:
                    self.bnm_scheduler.step(it)
                    self.tb_log.add_scalar('bn_momentum', self.bnm_scheduler.lmbd(epoch), it)

                # train one epoch
                for cur_it, batch in enumerate(train_loader):
                    if lr_scheduler_each_iter:
                        self.lr_scheduler.step(it)
                        cur_lr = float(self.optimizer.lr)
                        self.tb_log.add_scalar('learning_rate', cur_lr, it)
                    else:
                        if self.lr_warmup_scheduler is not None and epoch < self.warmup_epoch:
                            self.lr_warmup_scheduler.step(it)
                            cur_lr = self.lr_warmup_scheduler.get_lr()[0]
                        else:
                            cur_lr = self.lr_scheduler.get_lr()[0]

                    loss, tb_dict, disp_dict = self._train_it(batch, self.lambda_completion, self.ckpt_dir)
                    it += 1

                    disp_dict.update({'loss': loss, 'lr': cur_lr})

                    # log to console and tensorboard
                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.set_postfix(disp_dict)
                    tbar.refresh()

                    if self.tb_log is not None:
                        self.tb_log.add_scalar('train_loss', loss, it)
                        self.tb_log.add_scalar('learning_rate', cur_lr, it)
                        for key, val in tb_dict.items():
                            self.tb_log.add_scalar('train_' + key, val, it)

                # save trained model
                trained_epoch = epoch + 1
                if trained_epoch % ckpt_save_interval == 0:
                    ckpt_name = os.path.join(self.ckpt_dir, 'checkpoint_epoch_%d' % trained_epoch)
                    save_checkpoint(
                        checkpoint_state(self.model, self.optimizer, trained_epoch, it), filename=ckpt_name,
                    )

                # eval one epoch
                if (epoch % eval_frequency) == 0:
                    pbar.close()
                    if Validation_loader is not None:
                        if cfg.RPN.ENABLED and not cfg.RCNN.ENABLED and not cfg.RPN.FIXED:
                            with torch.set_grad_enabled(False):
                                    val_loss, eval_dict, cur_performance = self.eval_epoch(Validation_loader)

                            if self.tb_log is not None:
                                self.tb_log.add_scalar('val_loss', val_loss, it)
                                for key, val in eval_dict.items():
                                    self.tb_log.add_scalar('val_' + key, val, it)
                        elif not cfg.RPN.ENABLED and cfg.RCNN.ENABLED:
                            # define criterions
                            criterion_tracking = torch.nn.MSELoss()
                            criterion_completion = ChamferLoss()
                            loss_validation = self.validate(Validation_loader, self.model,
                                                       criterion_tracking, criterion_completion, epoch,
                                                       lambda_completion=self.lambda_completion)
                            is_better = loss_validation < best_loss
                            best_loss = min(loss_validation, best_loss)
                            state = {'model_state': self.model.state_dict(), 'best_loss': best_loss}
                            best_model_path = os.path.join(self.ckpt_dir, f'{best_loss:.4f}_best_model.pth')
                            if is_better:
                                torch.save(state, best_model_path)
                pbar.close()
                pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train')
                pbar.set_postfix(dict(total_it=it))

        return None
