
import os
import sys
import numpy as np
import argparse
import pprint
import shutil


import torch
import torch.nn as nn

from model.fpn.resnet import resnet

xrange = range  # Python 3


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def convert_model(data='voc', load_name='weights/voc/voc_pretrained.npy'):
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


    if load_name.endswith('.pt'):
        params = {}
        m_dict = model.state_dict()
        for key in m_dict.keys():
            params[key] = m_dict[key].detach().cpu().numpy()
        np.save(load_name.replace('pt', 'npy'), params)
    elif load_name.endswith('.npy'):
        chkpt = {'model': model.module.state_dict() if type(
            model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                 }
        torch.save(chkpt, load_name.replace('npy', 'pt'))
    print('save successfully')


if __name__ == '__main__':
    convert_model()









