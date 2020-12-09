import cv2
import os
import argparse
import torch
import numpy as np
from attack import PGD

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def test_adv(cfg,
         data,
         weights=None,
         batch_size=4,
         step_size=0.01,
         num_steps=3,
         test_type = 'test',
         iou_thres=0.5,
         nms_thres=0.5,
         conf_thres=0.001,
         img_size=416,
         ):

    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    if test_type == 'valid':
        test_path = data['valid']  # path to test images
    elif test_type == 'test':
        test_path = data['test']
    print('test_path:', test_path)

    # Initialize model
    model = Darknet(cfg, img_size).cuda().eval()
    if weights.endswith('.pt'):  # pytorch format
        chkpt = torch.load(weights)
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(chkpt['model'], strict=True)
        del chkpt
    elif weights.endswith('.weights'):  # darknet format
        print('inherit model weights')
        if 'yolo' in weights:
            model.load_darknet_weights(weights)
            print(' inherit model weights from yolo pretrained weights')
        else:
            load_darknet_weights(model, weights)
            print(' do not inherit model weights from yolo pretrained weights')

    model_adv = PGD(model)


    dataset = LoadImagesAndLabels(test_path,
                                  img_size=img_size,
                                  batch_size=batch_size,
                                  augment=False,
                                  hyp=None,  # augmentation hyperparameters
                                  rect=True,  # rectangular training
                                  image_weights=False,
                                  cache_labels=False,
                                  cache_images=False)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=min([os.cpu_count(), batch_size, 16]),
                                             shuffle=False,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)


    # Run inference
    seen = 0
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.cuda()
        targets = targets.cuda()

        if num_steps * step_size > 0:
            input_img = model_adv.adv_sample_infer(imgs, targets, step_size=step_size, num_steps=num_steps)
        else:
            input_img = imgs

        _, _, height, width = imgs.shape  # batch size, channels, height, width
        # Run model
        inf_out, train_out = model(input_img)  # inference and training outputs
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls
        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue


            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                seen += 1

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

        # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    return map











