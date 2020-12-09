import os
import sys
import numpy as np
import argparse
import pprint
import shutil
import time
import random

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from PIL import Image

from model.utils.config import cfg, cfg_from_file, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from bbox_nms import multiclass_nms

from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections
from model.fpn.resnet import resnet
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from attack import *
from misc_functions import *

xrange = range  # Python 3

def init_seeds(seed=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def test_adv(step_size = 0.01,
             num_steps = 0,
             dataset='coco',
             batch_size = 1,
             weights = 'weights/voc_pretrained.npy',
             save = False,
             grad_cam = False):
    if save:
        p_t1 = 'detect_adv_normal'
        if not os.path.exists(p_t1):
            os.makedirs(p_t1)

    cfg_file = 'cfgs/res50.yml'
    cfg_from_file(cfg_file)
    cfg.POOLING_MODE = 'align'
    cfg.TRAIN.USE_FLIPPED = False
    init_seeds(cfg.RNG_SEED)

    if dataset == "pascal_voc":
        imdb_name = "voc_2007_trainval"
        imdbval_name = "voc_2007_test"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif dataset == "pascal_voc_0712":
        imdb_name = "voc_2007_trainval+voc_2012_trainval"
        imdbval_name = "voc_2007_test"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif dataset == "coco":
        imdb_name = "coco_2014_train+coco_2014_valminusminival"
        imdbval_name = "coco_2014_minival"
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif dataset == "imagenet":
        imdb_name = "imagenet_train"
        imdbval_name = "imagenet_val"
        set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif dataset == "vg":
        imdb_name = "vg_150-50-50_minitrain"
        imdbval_name = "vg_150-50-50_minival"
        set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    imdb, roidb, ratio_list, ratio_index = combined_roidb(imdbval_name, training=False)
    imdb.competition_mode(on=True)
    print('{:d} roidb entries'.format(len(roidb)))

    model = resnet(imdb.classes, 50, pretrained=False, class_agnostic=False)
    print("load checkpoint %s" % (weights))
    if weights.endswith('.pt'):
        checkpoint = torch.load(weights)
        checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if
                               model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(checkpoint['model'], strict=True)
    elif weights.endswith('.npy'):
        checkpoint = np.load(weights, allow_pickle=True).item()
        model_dict = {k: torch.from_numpy(checkpoint[k]) for k in checkpoint.keys() if
                      model.state_dict()[k].numel() == torch.from_numpy(checkpoint[k]).numel()}
        model.load_state_dict(model_dict, strict=True)
    # load_state_dict(fpn.state_dict(), checkpoint['state_dict'])
    model.cuda().eval()
    del checkpoint
    print('load model successfully!')
    if not grad_cam:
        for param in model.parameters():
            param.requires_grad = False
    model_adv = PGD(model)

    max_per_image = 100
    vis = False
    thresh = 0.001
    iou_thre = 0.5
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(imdb.classes))]

    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    save_name = 'v1'
    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    im_data = torch.FloatTensor(1).cuda()
    im_info = torch.FloatTensor(1).cuda()
    num_boxes = torch.LongTensor(1).cuda()
    gt_boxes = torch.FloatTensor(1).cuda()

    for i in range(num_images):

        data = next(data_iter)

        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        if vis or save:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)

        with torch.enable_grad():
            if num_steps*step_size > 0:
                im_adv = model_adv.adv_sample_infer(im_data, im_info, gt_boxes, num_boxes, step_size, num_steps=num_steps)
            else:
                im_adv = im_data
            if save:
                file_name = imdb.image_path_at(i).split('/')[-1]

            if grad_cam:
                model.eval()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, conv_output = model(im_adv, im_info, gt_boxes, num_boxes, grad_cam=True)

                one_hot_output = torch.zeros_like(cls_prob)
                one_hot_output[0][:, 0:-1] = 1

                model.zero_grad()
                cls_prob.backward(gradient=one_hot_output, retain_graph=True)
                guided_gradients = model.gradients.cpu().data.numpy()[0]
                target = conv_output.cpu().data.numpy()[0]
                ws = np.mean(guided_gradients, axis=(1, 2))  # take averages for each gradient
                # create empty numpy array for cam
                cam = np.ones(target.shape[1:], dtype=np.float32)
                # multiply each weight with its conv output and then, sum
                for l, w in enumerate(ws):
                    cam += w * target[l, :, :]
                cam = np.maximum(cam, 0)
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # normalize between 0-1
                cam = np.uint8(cam * 255)  # scale between 0-255 to visualize
                im_rgb = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
                cam = np.uint8(Image.fromarray(cam).resize((im_rgb.shape[1],
                                                            im_rgb.shape[0]), Image.ANTIALIAS)) / 255
                original_image = Image.fromarray(im_rgb)
                save_class_activation_images(original_image, cam, file_name)


        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = model(im_adv, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(batch_size, -1, 4 * (imdb.num_classes - 1))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        scores = scores.squeeze()
        pred_boxes /= data[1][0][2].item()
        pred_boxes = pred_boxes.squeeze()
        nms_cfg = {'type': 'nms', 'iou_threshold': iou_thre}
        det_bboxes, det_labels = multiclass_nms(pred_boxes, scores, thresh, nms_cfg, max_per_image)
        keep = det_bboxes[:, 4] > thresh
        det_bboxes = det_bboxes[keep]
        det_labels = det_labels[keep]

        for j in xrange(0, imdb.num_classes - 1):
            inds = torch.nonzero(det_labels == j, as_tuple=False).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_dets = det_bboxes[inds]
                if vis or save:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), color=colors[int(j)])
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(0, imdb.num_classes - 1)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(0, imdb.num_classes - 1):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        if save:
            cv2.imwrite(os.path.join(p_t1, file_name.replace('jpg', 'png')), im2show)
        elif vis:
            cv2.imwrite('result.png', im2show)
        if i % 200 == 0:
            print(i, 'waiting.....')

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    mAP = imdb.evaluate_detections(all_boxes, output_dir)
    return mAP

