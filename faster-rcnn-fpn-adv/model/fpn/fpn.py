import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck
from torch.autograd.gradcheck import gradgradcheck
from torch.autograd import Variable
import numpy as np

from model.utils.config import cfg
from model.rpn.rpn_fpn import _RPN_FPN


from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import time
import pdb

from torchvision.ops import RoIAlign


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, size, mode):
        # x = F.interpolate(x, size=size, mode=mode, align_corners=False)
        x = F.interpolate(x, size=size, mode=mode)
        return x

class _FPN(nn.Module):
    """ FPN """
    def __init__(self, classes, class_agnostic):
        super(_FPN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        self.maxpool2d = nn.MaxPool2d(1, stride=2)

        self._init_modules()

        # define rpn
        self.RCNN_rpn = _RPN_FPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        roi_layer = {'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 0}
        featmap_strides = [4,8,16,32]
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        # self.RCNN_roi_crop = _RoICrop()

        self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(1024, 4 * (self.n_classes-1))

        self.RCNN_top = nn.Sequential(
            nn.Linear(256 * cfg.POOLING_SIZE * cfg.POOLING_SIZE, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True)
        )


        self.upsample = Upsample()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def build_roi_layers(self, layer_cfg, featmap_strides):
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (dict): Dictionary to construct and config RoI layer
                operation. Options are modules under ``mmcv/ops`` such as
                ``RoIAlign``.
            featmap_strides (int): The stride of input feature map w.r.t to the
                original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature
                map.
        """

        cfg = layer_cfg.copy()

        roi_layers = nn.ModuleList([RoIAlign((cfg['output_size'], cfg['output_size']), 1.0 / s,
                 cfg['sampling_ratio'], aligned=True) for s in featmap_strides])

        return roi_layers

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_toplayer, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_smooth3, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer2, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_latlayer3, 0, 0.01, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return self.upsample(x, size=(H,W), mode='nearest') + y

    def _PyramidRoI_Feat(self, feat_maps, rois, im_info):
        ''' roi pool on pyramid feature maps'''
        # do roi pooling based on predicted rois
        img_area = im_info[0][0] * im_info[0][1]
        scale = torch.sqrt(
            (rois.data[:, 3] - rois.data[:, 1]) * (rois.data[:, 4] - rois.data[:, 2]))
        roi_level = torch.floor(torch.log2(scale / 56 + 1e-6))
        roi_level = roi_level.clamp(min=0, max=4 - 1).long()

        if cfg.POOLING_MODE == 'align':
            roi_pool_feats = []
            box_to_levels = []
            roi_feats = feat_maps[0].new_zeros(
                rois.size(0), 256, cfg.POOLING_SIZE, cfg.POOLING_SIZE)
            for i, l in enumerate(range(0, 4)):
                # if (roi_level == l).sum() == 0:
                #     continue
                if (roi_level == l).sum() == 0:
                    continue
                idx_l = (roi_level == l).nonzero().squeeze().view(-1)
                box_to_levels.append(idx_l)
                scale = feat_maps[i].size(2) / im_info[0][0]
                # feat = self.RCNN_roi_align(feat_maps[i], rois[idx_l], scale)
                feat = self.roi_layers[i](feat_maps[i], rois[idx_l,:])
                roi_feats[idx_l] = feat
                # roi_pool_feats.append(feat)
            # roi_pool_feat = torch.cat(roi_pool_feats)
            # box_to_level = torch.cat(box_to_levels)
            # idx_sorted, order = torch.sort(box_to_level)
            # roi_pool_feat = roi_pool_feat[order]
            roi_pool_feat = roi_feats

        else:
            raise NotImplementedError

            
        return roi_pool_feat

    def forward(self, im_data, im_info, gt_boxes, num_boxes, fa=False, only_fa=False, grad_cam=False):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # Bottom-up
        c1 = self.RCNN_layer0(im_data)
        c2 = self.RCNN_layer1(c1)
        c3 = self.RCNN_layer2(c2)
        c4 = self.RCNN_layer3(c3)
        c5 = self.RCNN_layer4(c4)
        if fa:
            avg_out = self.global_avg_pool(c5)
            avg_out = avg_out.view(avg_out.size(0), -1)
            if only_fa:
                return avg_out
        if grad_cam:
            c5.register_hook(self.save_gradient)
            conv_output = c5

        # Top-down
        p5 = self.RCNN_toplayer(c5)
        p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
        p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
        p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
        p5 = self.RCNN_roi_feat_ds(p5)
        p4 = self.RCNN_smooth1(p4)
        p3 = self.RCNN_smooth2(p3)
        p2 = self.RCNN_smooth3(p2)

        p6 = self.maxpool2d(p5)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        rois, rpn_loss_cls, rpn_loss_bbox, rpn_loss_cls_neg = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois = rois.view(-1, 5)
            rois_label = rois_label.view(-1).long()
            gt_assign = gt_assign.view(-1).long()
            pos_id = (rois_label<(self.n_classes-1)).nonzero().squeeze()
            gt_assign_pos = gt_assign[pos_id]
            rois_label_pos = rois_label[pos_id]
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois_label = Variable(rois_label)
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))


        else:
            ## NOTE: additionally, normalize proposals to range [0, 1],
            #        this is necessary so that the following roi pooling
            #        is correct on different feature maps
            # rois[:, :, 1::2] /= im_info[0][1]
            # rois[:, :, 2::2] /= im_info[0][0]

            rois_label = None
            gt_assign = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = torch.zeros(1).cuda()
            rpn_loss_bbox = torch.zeros(1).cuda()
            rois = rois.view(-1, 5)
            pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
            rois_label_pos_ids = pos_id
            rois_pos = Variable(rois[pos_id])
            rois = Variable(rois)

        # pooling features based on rois, output 14x14 map
        roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(roi_pool_feat)
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)
        RCNN_loss_cls = torch.zeros(1).cuda()
        RCNN_loss_bbox = torch.zeros(1).cuda()
        # BCE = nn.BCEWithLogitsLoss()
        if self.training:
            bbox_pred = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            length = rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)
            length = (length) % (self.n_classes - 1)
            bbox_pred = torch.gather(bbox_pred, 1, length)
            bbox_pred = bbox_pred.squeeze(1)

            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        rois = rois.view(batch_size, -1, rois.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

        # if self.training:
        #     rois_label = rois_label.view(batch_size, -1)

        if grad_cam:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, conv_output

        if fa:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, avg_out
        else:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label