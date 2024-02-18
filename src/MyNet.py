#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import cv2
from utils import *
import os
import json
import networks
import pytorch_ssim
import MyLoss

# 记录一个最优值
best_psnr = 0
best_epoch = 0
class EENet(object):
    """Implementation of UHDFour from Li et al. (2023)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.trainable = trainable
        self._compile()

    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print(' UHDFour from Li et al. (2023)')

        # Model (3x3=9 channels for Monte Carlo since it uses 3 HDR buffers)
        from MyNet_LOL import InteractNet as UHD_Net
        self.model = UHD_Net()
        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=2,
                                                                                 # T_mult=2)  # CosineAnnealingLR

        # CUDA support
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.L1 = self.L1.cuda()
                self.L2 = self.L2.cuda()
        self.model = torch.nn.DataParallel(self.model)

    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()

    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""

        # Create directory for model checkpoints, if nonexistent
        if first:
            ckpt_dir_name = f'{datetime.now():{self.p.dataset_name}-%m%d-%H%M}'
            if self.p.ckpt_overwrite:
                ckpt_dir_name = self.p.dataset_name

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)
            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)

        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/UHDFour-{}.pt'.format(self.ckpt_dir, self.p.dataset_name)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/UHDFour-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/UHDFour-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""
        # import pdb;pdb.set_trace()
        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        #声明使用全局变量
        global best_psnr
        global best_epoch
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)
        if( valid_psnr > best_psnr):
            best_psnr = valid_psnr
            best_epoch = epoch + 1
        print('best_psnr = ', best_psnr, 'best_epoch = ', best_epoch)
        # Decrease learning rate if plateau
        #self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            plot_per_epoch(self.ckpt_dir, 'Valid loss', stats['valid_loss'], 'L1_Loss')
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')

    @torch.no_grad()
    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()

        for batch_idx, (source, target1, lowlight_name) in enumerate(valid_loader):

            source = source.cuda()
            target1 = target1.cuda()

            # import pdb;pdb.set_trace()
            final_result = self.model(source)
            # Update loss
            loss = self.L1(final_result, target1)
            loss_meter.update(loss.item())

            # Compute PSRN
            for i in range(1):
                # import pdb;pdb.set_trace()
                final_result = final_result.cpu()
                target1 = target1.cpu()
                psnr_meter.update(psnr(final_result[i], target1[i]).item())

        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg

        return valid_loss, valid_time, psnr_avg

    def train(self, train_loader, valid_loader):
        """Trains UHDNet on training set."""

        self.model.train(True)
        if self.p.ckpt_load_path is not None:
            self.model.load_state_dict(torch.load(self.p.ckpt_load_path), strict=False)
            print('The pretrain model is loaded.')
        self._print_params()
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': []}

        # load VGG19 function
        VGG = networks.VGG19(init_weights='/home/mip1/yyf/vgg19.pth', feature_mode=True)
        VGG.cuda()
        VGG.eval()
        self.L_color = MyLoss.L_color()
        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            index = 0
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # import pdb;pdb.set_trace()
                final_result = self.model(source)


                loss_l1 = 5 * F.smooth_l1_loss(final_result, target)
                result_feature = VGG(final_result)
                target_feature = VGG(target)
                loss_per = 0.001 * self.L2(result_feature, target_feature)
                loss_ssim = 0.2 * (1 - pytorch_ssim.ssim(final_result, target))
                loss_color = torch.mean(self.L_color(final_result))
                loss_final = loss_l1 + loss_ssim + loss_per + loss_color

                loss_meter.update(loss_final.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss_final.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(),0.1) ###########new added
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                    if index % 20 == 0:
                        show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                    train_loss_meter.update(loss_meter.avg)
                    loss_meter.reset()
                    time_meter.reset()
                    # if batch_idx==10:
                #    break
                if index % 20 == 0:
                    print("total", ":", loss_final.item(), "loss_l1", ":", loss_l1.item(), "loss_ssim", ":",
                          loss_ssim.item())
                index = index+1

            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()
            # import pdb
            # pdb.set_trace()
        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))




