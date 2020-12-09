import os
import sys
import numpy as np
import argparse
import shutil
import time
import random
import glob
import cv2
from PIL import Image
from tqdm import trange
import xml.etree.ElementTree as ET
import scipy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from bbox_nms import multiclass_nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.fpn.resnet import resnet
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from attack import PGD
from misc_functions import *

def init_seeds(seed=2):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def prep_im_for_blob(im, pixel_means, pixel_stds, target_size, max_size):

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im /= pixel_stds
    # im = im[:, :, ::-1]
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def load_pascal_annotation(filename, classes):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    num_classes = len(classes)
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    ishards = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        diffc = obj.find('difficult')
        difficult = 0 if diffc == None else int(diffc.text)
        ishards[ix] = difficult
        class_to_ind = dict(zip(classes, range(num_classes)))
        cls = class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return boxes, gt_classes



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Faster-rcnn-fpn network')
    parser.add_argument('--data', type=str, default='pascal_voc_0712', help='dataset')
    parser.add_argument('--source', type=str, default='samples', help='the diretory of test image')
    parser.add_argument('--weights', type=str, default='weights/voc/adv_nsr_v1/backup15_v2.npy')
    args = parser.parse_args()
    load_name = args.weights
    data = args.data
    source = args.source

    cfg_from_file('cfgs/res50.yml')
    cfg.TRAIN.USE_FLIPPED = False
    cfg.POOLING_MODE = 'align'
    init_seeds(cfg.RNG_SEED)

    if data == 'coco':
        classes = load_classes('data/coco.names')
        classes = tuple(classes + ['__background__'])
    elif 'voc' in data:
        classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor', '__background__')
    model = resnet(classes, 50, pretrained=False, class_agnostic=False)
    print("load checkpoint %s" % (load_name))
    if load_name.endswith('.pt'):
        checkpoint = torch.load(load_name)
        checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if
                               model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(checkpoint['model'], strict=True)
    elif load_name.endswith('.npy'):
        checkpoint = np.load(load_name, allow_pickle=True).item()
        model_dict = {k: torch.from_numpy(checkpoint[k]) for k in checkpoint.keys() if
                      model.state_dict()[k].numel() == torch.from_numpy(checkpoint[k]).numel()}
        model.load_state_dict(model_dict, strict=True)
    model.cuda().eval()
    model_adv = PGD(model)

    files = sorted(glob.glob('%s/*.jpg' % source))
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    for img_name_index in trange(len(files)):
        img_path = files[img_name_index]
        img_name = img_path.split('/')[-1]

        original_image = Image.open(img_path)
        im = np.asarray(original_image)
        img_rgb = im.copy()
        im2show = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        target_size = cfg.TRAIN.SCALES[0]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.PIXEL_STDS, target_size,
                                        cfg.TRAIN.MAX_SIZE)

        anotation_path = img_path.replace('jpg', 'xml')
        boxes, gt_classes = load_pascal_annotation(anotation_path, classes)
        gt_boxes = np.empty((len(boxes), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = boxes * im_scale
        gt_boxes[:, 4] = gt_classes
        gt_boxes = torch.from_numpy(gt_boxes).unsqueeze(0).cuda().float()

        im_data = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).cuda().float()
        im_info = torch.from_numpy(np.array([[im.shape[0], im.shape[1], im_scale]]).astype(np.float64)).cuda()
        num_boxes = torch.tensor([gt_boxes.shape[1]]).float().cuda()

        im_adv = model_adv.adv_sample_infer(im_data, im_info, gt_boxes, num_boxes, step_size=0.03,
                                                      num_steps=1)

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
        cam = np.uint8(Image.fromarray(cam).resize((img_rgb.shape[1],
                                                    img_rgb.shape[0]), Image.ANTIALIAS)) / 255
        save_class_activation_images(original_image, cam, img_name)

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = model(im_adv, im_info, gt_boxes, num_boxes)
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4 * (len(classes) - 1))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        scores = scores.squeeze()
        pred_boxes /= im_scale
        pred_boxes = pred_boxes.squeeze()
        nms_cfg = {'type': 'nms', 'iou_threshold': 0.5}
        thresh = 0.001
        det_bboxes, det_labels = multiclass_nms(pred_boxes, scores, thresh, nms_cfg, 100)
        keep = det_bboxes[:, 4] > thresh
        det_bboxes = det_bboxes[keep]
        det_labels = det_labels[keep]

        for j in range(0, len(classes) - 1):
            inds = torch.nonzero(det_labels == j, as_tuple=False).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_dets = det_bboxes[inds]
                im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), color=colors[int(j)])
        if not os.path.exists('results/detection'):
            os.makedirs('results/detection')
        cv2.imwrite(os.path.join('results/detection', img_name.replace('jpg', 'png')), im2show)
    print("GradCAM completed")
