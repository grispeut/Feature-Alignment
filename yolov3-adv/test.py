import os
import argparse
import glob
import torch
import numpy as np
from attack_test import test_adv



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test YOLO-V3 network')
    parser.add_argument('--data', type=str, default='data/coco.data', help='dataset')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file')
    parser.add_argument('--bs', type=int, default=16, help='batch_size')
    parser.add_argument('--weights', type=str, default='weights/coco_pretrained.weights')
    parser.add_argument('--step_size', type=float, default=0.03, help='attack step size')
    parser.add_argument('--num_steps', type=int, default=1, help='attack num_steps')
    args = parser.parse_args()
    cfg = args.cfg
    data = args.data
    batch_size = args.bs
    weights = args.weights
    num_steps = args.num_steps
    step_size = args.step_size

    with torch.no_grad():
        mAP = test_adv(cfg=cfg, data=data, weights=weights, batch_size=batch_size, step_size=step_size,
                           num_steps=0)
        mAP = np.round(mAP, decimals=3)

        mAP_adv = test_adv(cfg=cfg, data=data, weights=weights, batch_size=batch_size, step_size=step_size, num_steps=num_steps)
        mAP_adv = np.round(mAP_adv, decimals=3)

    s = "model:{},map_adv:{},map:{},step_size:{},num_steps:{}".format(
        weights, mAP_adv, mAP, step_size, num_steps)

    print(s)











