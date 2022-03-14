from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data.data_provider import Provider
from data.provider_valid import Provider_valid
from model.model_interp import IFNet
# from loss.loss_ssim import MS_SSIM
from loss.loss_adversarial import NLayerDiscriminator, PixelDiscriminator, Discriminator, Discriminator2
from loss.loss_adversarial import weights_init_normal, ReplayBuffer
from utils.psnr_ssim import compute_psnr

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    if cfg.TRAIN.is_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider_valid(cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    model = IFNet(kernel_size=cfg.TRAIN.kernel_size).to(device)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    target_real = Variable(Tensor(cfg.TRAIN.batch_size, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(cfg.TRAIN.batch_size, 1).fill_(0.0), requires_grad=False)
    fake_buffer = ReplayBuffer()
    model.train()
    PAD = cfg.TRAIN.pad
    TPAD = cfg.TEST.pad
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    device1 = torch.device('cuda:0')

    # net_D = NLayerDiscriminator(input_nc=1).to(device1)
    # net_D = PixelDiscriminator(input_nc=1).to(device1)
    # net_D = Discriminator(input_nc=1).to(device1)
    net_D = Discriminator2(input_nc=1, ngf=cfg.TRAIN.ngf).to(device1)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('Discriminator built on %d GPUs ... ' % cuda_count, flush=True)
            net_D = nn.DataParallel(net_D)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('Discriminator built on a single GPU ... ', flush=True)
    
    net_D.apply(weights_init_normal)
    optimizer_ad = optim.Adam(net_D.parameters(), lr=cfg.TRAIN.ad_lr, betas=(cfg.TRAIN.ad_beta1, 0.999))

    L1Loss = nn.L1Loss()
    criterion_GAN = torch.nn.MSELoss()
    while iters <= cfg.TRAIN.total_iters:
        # train
        iters += 1
        t1 = time.time()
        input, target = train_provider.next()
        
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            if cfg.TRAIN.D_optimizer:
                for param_group in optimizer_ad.param_groups:
                    param_group['lr'] = current_lr
        
        # optimize generator network
        # for param in net_D.parameters():
        #     param.requires_grad = False
        optimizer.zero_grad()
        input = F.pad(input, (PAD, PAD, PAD, PAD))
        pred = model(input)
        pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
        D_pred = net_D(pred)

        loss1 = L1Loss(pred, target) * cfg.TRAIN.lambda_1
        loss2 = criterion_GAN(D_pred, target_real) * cfg.TRAIN.lambda_D
        loss = loss1 + loss2
        loss.backward()
        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        # optimize ad network
        # for param in net_D.parameters():
        #     param.requires_grad = True
        optimizer_ad.zero_grad()
        # Real loss
        pred_real = net_D(target)
        loss_D_real = criterion_GAN(pred_real, target_real)
        # fake loss
        # pred = fake_buffer.push_and_pop(pred)
        pred_fake = net_D(pred.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)
        # Total loss
        loss_ad = (loss_D_real + loss_D_fake)*0.5
        loss_ad.backward()
        optimizer_ad.step()

        sum_loss += loss.item()
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            logging.info('step %d, loss = %.4f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
            writer.add_scalar('loss/L1Loss', loss1, iters)
            writer.add_scalar('loss/adLoss_G', loss2, iters)
            writer.add_scalar('loss/adLoss_D', loss_ad, iters)
            writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0
        
        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            input = F.pad(input, (-PAD, -PAD, -PAD, -PAD))
            input0 = ((np.squeeze(input[0].data.cpu().numpy())) * 255).astype(np.uint8)
            input1 = input0[0]
            input2 = input0[3]
            # target = (np.squeeze(target[0].data.cpu().numpy()) * 255).astype(np.uint8)
            target = ((np.squeeze(target[0].data.cpu().numpy())) * 255).astype(np.uint8)
            pred = np.squeeze(pred[0].data.cpu().numpy())
            pred[pred>1] = 1; pred[pred<0] = 0
            pred = (pred * 255).astype(np.uint8)
            im_cat1 = np.concatenate([input1, input2], axis=1)
            im_cat2 = np.concatenate([pred, target], axis=1)
            im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
            Image.fromarray(im_cat).save(os.path.join(cfg.cache_path, '%06d.png' % iters))
        
        # valid
        if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.eval()
            running_loss = 0.0
            dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
            for k, data in enumerate(dataloader, 0):
                inputs, gt = data
                inputs = inputs.to(device)
                # gt = gt.to(device)
                inputs = F.pad(inputs, (TPAD, TPAD, TPAD, TPAD))
                with torch.no_grad():
                    pred = model(inputs)
                pred = F.pad(pred, (-TPAD, -TPAD, -TPAD, -TPAD))
                pred = np.squeeze(pred.data.cpu().numpy())
                gt = np.squeeze(gt.data.cpu().numpy())
                pred[pred>1] = 1; pred[pred<0] = 0
                _, psnr = compute_psnr(pred, gt)
                loss = psnr
                running_loss += loss
                if k == 0:
                    pred = (pred * 255).astype(np.uint8)
                    gt = (gt * 255).astype(np.uint8)
                    im_cat = np.concatenate([pred, gt], axis=1)
                    Image.fromarray(im_cat).save(os.path.join(cfg.valid_path, str(iters).zfill(6)+'.png'))
            epoch_loss = running_loss / len(valid_provider)
            print('model-%d, valid-psnr=%.6f' % (iters, epoch_loss))
            writer.add_scalar('psnr', epoch_loss, iters)
            f_valid_txt.write('model-%d, valid-psnr=%.6f' % (iters, epoch_loss))
            f_valid_txt.write('\n')
            f_valid_txt.flush()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters))
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='standard', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(cfg.TRAIN.g_beta1, 0.999))
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')