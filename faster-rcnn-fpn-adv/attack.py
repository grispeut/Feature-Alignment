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
    
    def adv_sample_train_tod(self, im_data, im_info, gt_boxes, num_boxes, step_size, num_steps=1, all_bp=False, sf=0.5):
        im_adv = im_data.clone().detach()

        x = im_adv.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(x, self.pixel_stds)
        x = torch.add(x, self.pixel_means)
        x_raw = x.clone()
        x.requires_grad_()
        step_size_t = step_size * 255
        loss_clean = torch.zeros(1).cuda()
        for i in range(num_steps):
            im = torch.sub(x, self.pixel_means)
            im = torch.div(im, self.pixel_stds)
            im = im.permute(0, 3, 1, 2).contiguous()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = self.basic_model(im, im_info, gt_boxes, num_boxes, fa=False)

            loss_loc = (rpn_loss_box + RCNN_loss_bbox).view(-1)
            loss_cls = (rpn_loss_cls + RCNN_loss_cls).view(-1)
            x_loc_grad = torch.autograd.grad(loss_loc, [x], retain_graph=True)[0]
            if all_bp and i == 0:
                x_cls_grad = torch.autograd.grad(loss_cls, [x], retain_graph=True)[0]
                loss_clean = loss_loc + loss_cls
                if sf > 0:
                    loss_clean = loss_clean * sf
                loss_clean.backward(retain_graph=False)
            else:
                x_cls_grad = torch.autograd.grad(loss_cls, [x], retain_graph=False)[0]
            x_loc_eta = step_size_t * torch.sign(x_loc_grad)
            x_cls_eta = step_size_t * torch.sign(x_cls_grad)

            x_loc = x.clone().detach()
            x_loc.data = x_loc.data + x_loc_eta
            x_loc.data = torch.min(torch.max(x_loc.data, x_raw - self.epsilon), x_raw + self.epsilon)
            x_loc.data = torch.clamp(x_loc.data, 0, 255)
            im_adv_loc = torch.sub(x_loc, self.pixel_means)
            im_adv_loc = torch.div(im_adv_loc, self.pixel_stds)
            im_adv_loc = im_adv_loc.permute(0, 3, 1, 2).contiguous()

            x_cls = x.clone().detach()
            x_cls.data = x_cls.data + x_cls_eta
            x_cls.data = torch.min(torch.max(x_cls.data, x_raw - self.epsilon), x_raw + self.epsilon)
            x_cls.data = torch.clamp(x_cls.data, 0, 255)
            im_adv_cls = torch.sub(x_cls, self.pixel_means)
            im_adv_cls = torch.div(im_adv_cls, self.pixel_stds)
            im_adv_cls = im_adv_cls.permute(0, 3, 1, 2).contiguous()

            with torch.no_grad():
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = self.basic_model(im_adv_loc, im_info, gt_boxes, num_boxes, fa=False)
                loss_img_loc = (rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox).view(-1)

                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = self.basic_model(im_adv_cls, im_info, gt_boxes, num_boxes, fa=False)
                loss_img_cls = (rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox).view(-1)

                im_adv = im_adv_loc if loss_img_loc > loss_img_cls else im_adv_cls

                if not i == num_steps - 1:
                    x_adv = im_adv.permute(0, 2, 3, 1).contiguous()
                    x_adv = torch.mul(x_adv, self.pixel_stds)
                    x_adv = torch.add(x_adv, self.pixel_means)
                    x.data = x_adv.data

        return im_adv.detach(), loss_clean.detach()

