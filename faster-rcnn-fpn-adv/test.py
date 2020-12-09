import os
import glob
import torch
import argparse
import numpy as np
from attack_test import test_adv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Faster-rcnn-fpn network')
    parser.add_argument('--dataset', type=str, default='pascal_voc_0712', help='dataset')
    parser.add_argument('--weights', type=str, default='weights/voc_pretrained.npy')
    parser.add_argument('--step_size', type=float, default=0.03, help='attack step size')
    parser.add_argument('--num_steps', type=int, default=1, help='attack num_steps')
    args = parser.parse_args()
    weights = args.weights
    dataset = args.dataset
    num_steps = args.num_steps
    step_size = args.step_size

    with torch.no_grad():
        mAP = test_adv(step_size=step_size, num_steps=0, dataset=dataset, batch_size=1, weights=weights)
        mAP = np.round(np.array(mAP), decimals=3)
        mAP_adv = test_adv(step_size=step_size, num_steps=num_steps, dataset=dataset, batch_size=1,
                           weights=weights)
        mAP_adv = np.round(mAP_adv, decimals=3)

    s = "model:{},map_adv:{},map:{},step_size:{},num_steps:{}".format(
        weights, mAP_adv, mAP, step_size, num_steps)

    print(s)










