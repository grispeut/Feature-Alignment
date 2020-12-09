import numpy as np
import cv2
import torch
import torch.nn as nn
from model.utils.config import cfg


class PGD(nn.Module):
    def __init__(self, basic_model):
        super(PGD, self).__init__()
        self.basic_model = basic_model

        self.pixel_means = torch.from_numpy(cfg.PIXEL_MEANS).cuda().view(1, -1).float()
        self.pixel_stds = torch.from_numpy(cfg.PIXEL_STDS).cuda().view(1, -1).float()

        self.epsilon = 0.03 * 255


    def adv_sample_infer(self, im_data, im_info, gt_boxes, num_boxes, step_size, num_steps=1):
        im_adv = im_data.clone().detach()

        x = im_adv.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, self.pixel_stds)
        x = torch.add(x, self.pixel_means)
        x_raw = x.clone()
        x.requires_grad_()

        step_size_t = step_size * 255
        self.basic_model.train()
        for step in range(num_steps):
            im = torch.sub(x, self.pixel_means)
            im = torch.div(im, self.pixel_stds)
            im = im.permute(0, 3, 1, 2).contiguous()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = self.basic_model(im, im_info, gt_boxes, num_boxes)

            loss_adv = (rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox).view(-1)

            x_grad = torch.autograd.grad(loss_adv, [x], retain_graph=False)[0]

            eta = torch.sign(x_grad) * step_size_t
            x.data = x.data + eta
            x.data = torch.min(torch.max(x.data, x_raw - self.epsilon), x_raw + self.epsilon)
            x.data.clamp_(0, 255)

            im_adv = torch.sub(x, self.pixel_means)
            im_adv = torch.div(im_adv, self.pixel_stds)
            im_adv = im_adv.permute(0, 3, 1, 2).contiguous()

        self.basic_model.eval()
        return im_adv.detach()

    def adv_sample_train(self, im_data, im_info, gt_boxes, num_boxes, step_size, num_steps=1, all_bp=False, sf=0.5):
        im_adv = im_data.clone().detach()

        x = im_adv.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, self.pixel_stds)
        x = torch.add(x, self.pixel_means)
        x_raw = x.clone()
        x.requires_grad_()

        step_size_t = step_size * 255
        loss_clean = torch.zeros(1).cuda()
        ssfa_out = torch.zeros(1).cuda()
        for step in range(num_steps):
            im = torch.sub(x, self.pixel_means)
            im = torch.div(im, self.pixel_stds)
            im = im.permute(0, 3, 1, 2).contiguous()

            if step == 0 and all_bp:
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, ssfa_out = self.basic_model(im, im_info, gt_boxes, num_boxes, fa=True)
                loss_adv = (rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox).view(-1)
                loss_adv *= sf
                loss_clean = loss_adv.detach()
                loss_adv.backward()
                x_grad = x.grad

            else:
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = self.basic_model(im, im_info, gt_boxes, num_boxes)
                loss_adv = (rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox).view(-1)
                x_grad = torch.autograd.grad(loss_adv, [x], retain_graph=False)[0]

            eta = torch.sign(x_grad) * step_size_t
            x.data = x.data + eta
            x.data = torch.min(torch.max(x.data, x_raw - self.epsilon), x_raw + self.epsilon)
            x.data.clamp_(0, 255)

            im_adv = torch.sub(x, self.pixel_means)
            im_adv = torch.div(im_adv, self.pixel_stds)
            im_adv = im_adv.permute(0, 3, 1, 2).contiguous()

        return im_adv.detach(), loss_clean, ssfa_out.detach()

