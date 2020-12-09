import os
import sys
import numpy as np
import argparse
import time
import random
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from model.utils.config import cfg, cfg_from_file
from model.utils.net_utils import clip_gradient
from model.fpn.resnet import resnet
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from attack import PGD
from convert import convert_model


def init_seeds(seed=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

    parser.add_argument('--r', dest='resume', default=False, action="store_true", help='resume checkpoint or not')
    parser.add_argument('--dataset', type=str, default='pascal_voc_0712', dest='dataset', help='training dataset')
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file',default='cfgs/res50.yml', type=str)
    parser.add_argument('--weights', type=str, default='weights/voc_pretrained.npy')
    parser.add_argument('--bs', type=int, default=8, help='batch_size')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--save_dir', type=str, default='weights/', dest='save_dir',help='directory to save models')

    parser.add_argument('--adv', default=False, action="store_true")
    parser.add_argument('--iw', type=float, default=0.5, help='imgs_weight')
    parser.add_argument('--step_size', type=float, default=0.01, help='attack step size')
    parser.add_argument('--num_steps', type=int, default=1, help='attack num_steps')
    parser.add_argument('--kdfa', default=False, action="store_true", help='Knowledge-Distilled Feature Alignment')
    parser.add_argument('--ssfa', default=False, action="store_true", help='Self-Supervised Feature Alignment')
    parser.add_argument('--beta', type=float, default=1., help='weight for kdfa')
    parser.add_argument('--gamma', type=float, default=1., help='weight for ssfa')
    parser.add_argument('--kdfa_weights', type=str, default='weights/voc_pretrained.npy')

    # config optimization
    parser.add_argument('--o', type=str, default="sgd", dest='optimizer', help='training optimizer', )
    parser.add_argument('--lr', type=float, default=0.001, dest='lr', help='starting learning rate')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':

    args = parse_args()
    print(args)

    path = args.save_dir
    if not os.path.exists(path):
        os.makedirs(path)
    wdir = path + os.sep  # weights dir
    last = wdir + 'last.pt'

    weights = args.weights
    epochs = args.epochs
    batch_size = args.bs
    lr = args.lr

    # adv
    adv = args.adv
    imgs_weight = args.iw
    num_steps = args.num_steps
    step_size = args.step_size
    kdfa = args.kdfa
    ssfa = args.ssfa
    beta = args.beta
    gamma = args.gamma
    kdfa_weights = args.kdfa_weights

    tb_writer = SummaryWriter()

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    init_seeds(cfg.RNG_SEED)

    cfg_from_file(args.cfg_file)
    cfg.TRAIN.USE_FLIPPED = False
    cfg.POOLING_MODE = 'align'

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))

    model = resnet(imdb.classes, 50, pretrained=False, class_agnostic=False)

    train_size = len(roidb)
    sampler_batch = sampler(train_size, batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                             imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          sampler=sampler_batch, num_workers=min([os.cpu_count(), batch_size, 16]))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    print("load checkpoint %s" % (weights))
    start_epoch = 1
    if weights.endswith('.pt'):
        checkpoint = torch.load(weights)
        checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if
                               model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(checkpoint['model'], strict=True)
        if args.resume:
            if (checkpoint['optimizer'] is not None):
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
    elif weights.endswith('.npy'):
        checkpoint = np.load(weights, allow_pickle=True).item()
        model_dict = {k: torch.from_numpy(checkpoint[k]) for k in checkpoint.keys() if
                      model.state_dict()[k].numel() == torch.from_numpy(checkpoint[k]).numel()}
        model.load_state_dict(model_dict, strict=True)
    model.cuda().train()
    del checkpoint
    print('load model successfully!')

    iters_per_epoch = int(train_size / batch_size)

    if adv:
        model_adv = PGD(model)
        if kdfa:
            model_t = resnet(imdb.classes, 50, pretrained=False, class_agnostic=False)
            if kdfa_weights.endswith('.pt'):
                checkpoint = torch.load(kdfa_weights)
                checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if
                                       model_t.state_dict()[k].numel() == v.numel()}
                model_t.load_state_dict(checkpoint['model'], strict=True)
            elif kdfa_weights.endswith('.npy'):
                checkpoint = np.load(kdfa_weights, allow_pickle=True).item()
                model_dict = {k: torch.from_numpy(checkpoint[k]) for k in checkpoint.keys() if
                              model_t.state_dict()[k].numel() == torch.from_numpy(checkpoint[k]).numel()}
                model_t.load_state_dict(model_dict, strict=True)
            model_t.cuda().eval()
            for param_k in model_t.parameters():
                param_k.requires_grad = False


    for epoch in range(start_epoch, epochs+1):
        if epoch == int(epochs*2/3)+1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print('param_group lr anealing: ', param_group['lr'])

        print(('\n' + '%10s' * 7) % (
            'Epoch', 'gpu_mem', 'clean', 'adv', 'kd', 'ss', 'total'))
        # setorching to train mode
        model.train()
        t0 = time.time()

        data_iter = iter(dataloader)
        mloss = torch.zeros(5).cuda()  # mean losses,'clean', 'adv', 'kd', 'ss', 'total'
        loss_kd = torch.zeros(1).cuda()
        loss_ss = torch.zeros(1).cuda()
        loss_clean = torch.zeros(1).cuda()
        loss_adv = torch.zeros(1).cuda()
        loss = torch.zeros(1).cuda()
        pbar = trange(iters_per_epoch)
        for step in pbar:
            data = next(data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
            if adv:
                im_adv, loss_clean, ssfa_out = model_adv.adv_sample_train(im_data, im_info, gt_boxes, num_boxes,
                                                                               step_size=step_size,
                                                                               num_steps=num_steps, all_bp=True, sf=imgs_weight)
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, fa_out = model(im_adv, im_info, gt_boxes, num_boxes, fa=True)
                fa_out_norm = F.normalize(fa_out, dim=1)
                if kdfa:
                    kdfa_out = model_t(im_data, im_info, gt_boxes, num_boxes, fa=True, only_fa=True)
                    kdfa_out_norm = F.normalize(kdfa_out, dim=1)
                    kd_sim = torch.einsum('nc,nc->n', [fa_out_norm, kdfa_out_norm])
                    kd_sim.data.clamp_(-1., 1.)
                    loss_kd = (1. - kd_sim).mean().view(-1) * beta
                if ssfa:
                    ssfa_out_norm = F.normalize(ssfa_out, dim=1)
                    ss_sim = torch.einsum('nc,nc->n', [fa_out_norm, ssfa_out_norm])
                    ss_sim.data.clamp_(-1., 1.)
                    loss_ss = (1 - ss_sim).mean().view(-1) * gamma

                loss_adv = ((rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()) * (1-imgs_weight)).view(-1)

            else:
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = model(im_data, im_info, gt_boxes, num_boxes)

                loss_adv = (rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox).view(-1)

            loss_items = torch.cat((loss_clean, loss_adv, loss_kd, loss_ss,
                                    (loss_clean + loss_adv + loss_kd + loss_ss))).detach()
            if torch.isnan(loss_adv):
                print('WARNING: non-finite loss, ending training ', loss_adv)
                break
            loss = loss_adv + loss_kd + loss_ss
            mloss = (mloss * step + loss_items) / (step + 1)
            loss.backward()

            clip_gradient(model, 10.)

            optimizer.step()
            optimizer.zero_grad()


            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            script = ('%10s' * 2 + '%10.3g' * 5) % (
                '%g/%g' % (epoch, epochs), '%.3gG' % mem, *mloss)
            pbar.set_description(script)
            # print(script)

        x = list(mloss.cpu().numpy())  # + list(results) + list([thre]) + list([prune_ratio]) + list([par_prune_ratio])
        titles = ['Loss_clean', 'Loss_adv', 'Loss_kd', 'Loss_ss', 'Train_loss']
        for xi, title in zip(x, titles):
            tb_writer.add_scalar(title, xi, epoch)

        chkpt = {'epoch': epoch,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()
                 }
        torch.save(chkpt, last)
        if epoch > 0 and ((epoch) % 5 == 0):
            torch.save(chkpt, wdir + 'backup%g.pt' % epoch)
            if epoch == epochs:
                convert_model(data=args.dataset, load_name=wdir + 'backup%g.pt' % epoch)
        time_consume = '%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
        print(time_consume)

    torch.cuda.empty_cache()

