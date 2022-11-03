#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:22:14 2020

@author: amax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.fpn import FPN50
from torch.autograd import Variable
from torch.autograd import Function
from tools.encoder import DataEncoder
import sys,os
import numpy as np
sys.path.append(os.getcwd()+'/../../')
from tools.bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace
from tools.utils import change_box_order,convert_polyons_into_angle,convert_polyons_into_angle_cuda,convert_polyons_into_angle_upgrade
from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.modeling.roi_heads.rroi_heads import build_roi_heads
import random
import time
BN_MOMENTUM = 0.1
class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        if bn_type=="test_bn":
            return nn.Sequential(
                nn.BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                BatchNorm2d(num_features, **kwargs),
                nn.ReLU()
            )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context
class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output
class Binary_a(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        a = torch.ones_like(input)
        b = torch.zeros_like(input)
        output = torch.where(input>=0.5,a,b)
        return output
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        #*******************ste*********************
        grad_input = grad_output.clone()
        # print(grad_input)
        #****************saturate_ste***************
        grad_input[input.ge(1)] = 0
        grad_input[input.le(0)] = 0
        # print(grad_input)
        '''
        #******************soft_ste*****************
        size = input.size()
        zeros = torch.zeros(size).cuda()
        grad = torch.max(zeros, 1 - torch.abs(input))
        #print(grad)
        grad_input = grad_output * grad
        '''
        return grad_input
class TCM(nn.Module):
    def __init__(self,size=256):
        super(TCM, self).__init__()
        self.TOP_DOWN_PYRAMID_SIZE=256
        self.saliency_map_size=2
        self.size=size
        self.conv1=nn.Conv2d(self.size,self.TOP_DOWN_PYRAMID_SIZE,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(self.TOP_DOWN_PYRAMID_SIZE,self.TOP_DOWN_PYRAMID_SIZE,kernel_size=3,stride=1,padding=1)
        self.conv3=nn.Conv2d(self.TOP_DOWN_PYRAMID_SIZE,self.saliency_map_size,kernel_size=1,stride=1,padding=0)
        self.m=nn.Softmax(dim=1)
    def forward(self, x):

        Conv1=self.conv1(x)
        Conv2=self.conv2(Conv1)
        Conv3=self.conv3(Conv2)
        text_prob = self.m(Conv3)
        active = text_prob[:,1,:,:].unsqueeze(1).exp()#add channel torch.Tensor([1.0]).cuda().exp()-
        #active_zero=text_prob[:,0,:,:].unsqueeze(1).exp()#add channel
        broadcast =active.expand(x.shape)
        mult = torch.mul(Conv1,broadcast)
        #global_text_seg=nn.functional.interpolate(Conv3, size=(self.image_shape, self.image_shape), scale_factor=None, mode='bilinear', align_corners=None)
        global_text_seg=active#torch.cat([active_zero,active],dim=1)
        output = torch.add(Conv1, mult)
        return output,global_text_seg
class panopticFPN_OCR(nn.Module):
    def __init__(self,num_classes=2,bn_type=None):
        super(panopticFPN_OCR, self).__init__()
        self.num_classes=num_classes
        self.m = nn.Softmax(dim=1)
        self.bn_type=bn_type
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.semantic_branch = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.concatconv_branch=nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(512, 512)
        self.gn2 = nn.GroupNorm(256, 256)
        if bn_type=="test_bn":
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(2048, 512,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=relu_inplace),
            )
        else:
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(2048, 512,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(512),
                nn.ReLU(inplace=relu_inplace),
            )
        self.ocr_gather_head = SpatialGather_Module(2)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.bn_type,
                                                 )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        p2,p3,p4,p5,p6,p7=x
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        s5_add_s4=F.relu(self.gn1(self.concatconv_branch(s5+s4)))
        s4_add_s3 = F.relu(self.gn1(self.concatconv_branch(s5_add_s4 + s3)))
        s3_add_s2 = F.relu(self.gn1(self.concatconv_branch(s4_add_s3 + s2)))
        out_aux = self.conv3(s3_add_s2)

        feats = torch.cat([s2, s3, s4, s5], 1)
        out_aux_seg = []
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        text_prob=self.m(out)
        attention=text_prob[:,1,:,:].unsqueeze(1).exp()
        #panoptic_feature=self._upsample(out, 4 * h, 4 * w)
        return out_aux_seg,attention
class Parallel_mask_OCR(nn.Module):
    def __init__(self, num_classes=2, bn_type=None):
        super(Parallel_mask_OCR, self).__init__()
        self.num_classes = num_classes
        self.bn_type=bn_type
        self.P3_1 = self._make_layer(256, 256, 3, 1, 1)
        self.P3_2 = self._make_layer(256, 256, 1, 1, 0)
        self.P3_3 = self._make_layer(256, 256, 1, 1, 0)

        self.P4_1 = self._make_layer(256, 256, 3, 2, 1)
        self.P4_2 = self._make_layer(256, 256, 3, 1, 1)
        self.P4_3 = self._make_layer(256, 256, 1, 1, 0)

        self.P5_1 = self._make_layer_2(256, 256, 3, 2, 1)
        self.P5_2 = self._make_layer(256, 256, 3, 2, 1)
        self.P5_3 = self._make_layer(256, 256, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.m = nn.Softmax(dim=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0)
        if bn_type == "test_bn":
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(1024, 512,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=relu_inplace),
            )
        else:
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(1024, 512,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(512),
                nn.ReLU(inplace=relu_inplace),
            )
        self.ocr_gather_head = SpatialGather_Module(2)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.bn_type,
                                                 )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
    def _upsample(self, x,scale):
        _,_,h,w=x.size()
        return F.interpolate(x, size=(h*scale, w*scale), mode='bilinear', align_corners=True)
    def _make_layer_2(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
            nn.Conv2d(num_channels_pre_layer,num_channels_cur_layer,kerner_size,stride,padding,bias=False),
            nn.BatchNorm2d(num_channels_cur_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels_cur_layer, num_channels_cur_layer, kerner_size, stride, padding, bias=False),
            nn.BatchNorm2d(num_channels_cur_layer),
        ))
        return nn.Sequential(*transition_layers)
    def _make_layer(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
                nn.Conv2d(num_channels_pre_layer,num_channels_cur_layer,kerner_size,stride,padding,bias=False),
                nn.BatchNorm2d(num_channels_cur_layer),
                nn.ReLU(inplace=True),
                ))
        return nn.Sequential(*transition_layers)
    def forward(self, x):
        p2,p3,p4,p5,_,_=x
        p3_1 = self.P3_1(p3)
        p3_2 = self._upsample(self.P3_2(p4), scale=2)
        p3_3 = self._upsample(self.P3_3(p5), scale=4)
        p4_1 = self.P4_1(p3)
        p4_2 = self.P4_2(p4)
        p4_3 = self._upsample(self.P4_3(p5), scale=2)
        p5_1 = self.P5_1(p3)
        p5_2 = self.P5_2(p4)
        p5_3 = self.P5_3(p5)
        s3 = p3_1 + p3_2 + p3_3
        s4 = p4_1 + p4_2 + p4_3
        s5 = p5_1 + p5_2 + p5_3
        x0_h, x0_w = p2.size(2), p2.size(3)
        s3 = F.upsample(s3, size=(x0_h, x0_w), mode='bilinear')
        s4 = F.upsample(s4, size=(x0_h, x0_w), mode='bilinear')
        s5 = F.upsample(s5, size=(x0_h, x0_w), mode='bilinear')
        feats = torch.cat([p2, s3, s4, s5], 1)
        out_aux=self.conv3(self.conv2(p2))
        out_aux_seg = []
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        text_prob = self.m(out)
        attention = text_prob[:, 1, :, :].unsqueeze(1).exp()
        return out_aux_seg, attention
class OCR(nn.Module):
    def __init__(self,num_classes=2,bn_type=None):
        super(OCR, self).__init__()
        self.num_classes=num_classes
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.m = nn.Softmax(dim=1)
        if bn_type == "test_bn":
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(1024, 512,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=relu_inplace),
            )
        else:
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(1024, 512,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(512),
                nn.ReLU(inplace=relu_inplace),
            )
        self.ocr_gather_head = SpatialGather_Module(2)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.bn_type,
                                                 )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        p2,p3,p4,p5,_,_=x
        x0_h, x0_w = p2.size(2), p2.size(3)
        s3 = F.upsample(p3, size=(x0_h, x0_w), mode='bilinear')
        s4 = F.upsample(p4, size=(x0_h, x0_w), mode='bilinear')
        s5 = F.upsample(p5, size=(x0_h, x0_w), mode='bilinear')
        feats = torch.cat([p2, s3, s4, s5], 1)
        out_aux = self.conv3(self.conv2(p2))
        out_aux_seg = []
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        text_prob = self.m(out)
        attention = text_prob[:, 1, :, :].unsqueeze(1).exp()
        return out_aux_seg, attention

class Parallel_mask_OCR_attention(nn.Module):
    def __init__(self, num_classes=2, bn_type=None):
        super(Parallel_mask_OCR_attention, self).__init__()
        ####CA Module
        self.in_channels = 256*3
        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)
        ####
        self.num_classes = num_classes
        self.bn_type=bn_type
        self.P3_1 = self._make_layer(256, 256, 3, 1, 1)
        self.P3_2 = self._make_layer(256, 256, 1, 1, 0)
        self.P3_3 = self._make_layer(256, 256, 1, 1, 0)

        self.P4_1 = self._make_layer(256, 256, 3, 2, 1)
        self.P4_2 = self._make_layer(256, 256, 3, 1, 1)
        self.P4_3 = self._make_layer(256, 256, 1, 1, 0)

        self.P5_1 = self._make_layer_2(256, 256, 3, 2, 1)
        self.P5_2 = self._make_layer(256, 256, 3, 2, 1)
        self.P5_3 = self._make_layer(256, 256, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.m = nn.Softmax(dim=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.cls_head = nn.Conv2d(
            256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        """
        if bn_type == "test_bn":
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(1024, 512,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=relu_inplace),
            )
        else:
            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(1024, 512,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(512),
                nn.ReLU(inplace=relu_inplace),
            )
        self.ocr_gather_head = SpatialGather_Module(2)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=self.bn_type,
                                                 )
        self.cls_head = nn.Conv2d(
            512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        """
    def ChannelwiseAttention(self,input_):
        n_b, n_c, h, w = input_.size()
        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))
        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()
        return feats
    def _upsample(self, x,scale):
        _,_,h,w=x.size()
        return F.interpolate(x, size=(h*scale, w*scale), mode='bilinear', align_corners=True)
    def _make_layer_2(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
            nn.Conv2d(num_channels_pre_layer,num_channels_cur_layer,kerner_size,stride,padding,bias=False),
            nn.BatchNorm2d(num_channels_cur_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels_cur_layer, num_channels_cur_layer, kerner_size, stride, padding, bias=False),
            nn.BatchNorm2d(num_channels_cur_layer),
        ))
        return nn.Sequential(*transition_layers)
    def _make_layer(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
                nn.Conv2d(num_channels_pre_layer,num_channels_cur_layer,kerner_size,stride,padding,bias=False),
                nn.BatchNorm2d(num_channels_cur_layer),
                nn.ReLU(inplace=True),
                ))
        return nn.Sequential(*transition_layers)
    def forward(self, x):
        p2,p3,p4,p5,_,_=x
        p3_1 = self.P3_1(p3)
        p3_2 = self._upsample(self.P3_2(p4), scale=2)
        p3_3 = self._upsample(self.P3_3(p5), scale=4)
        p4_1 = self.P4_1(p3)
        p4_2 = self.P4_2(p4)
        p4_3 = self._upsample(self.P4_3(p5), scale=2)
        p5_1 = self.P5_1(p3)
        p5_2 = self.P5_2(p4)
        p5_3 = self.P5_3(p5)
        s3 = p3_1 + p3_2 + p3_3
        s4 = p4_1 + p4_2 + p4_3
        s5 = p5_1 + p5_2 + p5_3
        x0_h, x0_w = p2.size(2), p2.size(3)
        s3 = F.upsample(s3, size=(x0_h, x0_w), mode='bilinear')
        s4 = F.upsample(s4, size=(x0_h, x0_w), mode='bilinear')
        s5 = F.upsample(s5, size=(x0_h, x0_w), mode='bilinear')
        Upsampled_low_features=torch.cat([s3, s4, s5], 1)
        low_features_ca = self.ChannelwiseAttention(Upsampled_low_features)
        low_features = torch.mul(Upsampled_low_features, low_features_ca)
        high_features=p2.mean(1).exp().unsqueeze(1).expand(p2.shape)*p2
        feats = torch.cat([high_features, low_features], 1)
        out_aux=self.conv3(self.conv2(p2))
        out_aux_seg = []
        feats = self.conv3x3(feats)

        # compute contrast feature
        #feats = self.conv3x3_ocr(feats)
        #context = self.ocr_gather_head(feats, out_aux)
        #feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)
        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        text_prob = self.m(out)
        attention = text_prob[:, 1, :, :].unsqueeze(1).exp()
        return out_aux_seg, attention
class RFN(nn.Module):
    def __init__(self,num_classes=1,input_size=768,bn_type=None,cfg=None,encode=None):
        super(RFN, self).__init__()
        self.num_anchors = 8 # vertical offset -> *2
        self.num_classes = num_classes
        self.fpn = FPN50()
        self.encoder = encode
        self.loc_head = self._make_head(self.num_anchors*8)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)
        self.bn_type=bn_type
        #self.mask=panopticFPN_OCR(bn_type=self.bn_type)
        #self.mask=Parallel_mask_OCR(bn_type=self.bn_type)##bn_type buqizuoyongl SyncBatchnorm--> Batchnorm2d
        self.mask=Parallel_mask_OCR_attention(bn_type=self.bn_type)##bn_type buqizuoyongl SyncBatchnorm--> Batchnorm2d
        #self.mask=TCM()
        self.roi_heads = build_roi_heads(cfg)
        self.size_4 = input_size // 4
        self.size_8 = input_size // 8
        self.size_16 = input_size // 16
        self.size_32 = input_size // 32
        self.size_64 = input_size // 64
        self.size_128 = input_size // 128
        self.filter_size = [0, self.size_4 * self.size_4 * self.num_anchors,
                            self.size_4 * self.size_4 * self.num_anchors + self.size_8 * self.size_8 * self.num_anchors, \
                            self.size_4 * self.size_4 * self.num_anchors + self.size_8 * self.size_8 * self.num_anchors + self.size_16 * self.size_16 * self.num_anchors, \
                            self.size_4 * self.size_4 * self.num_anchors + self.size_8 * self.size_8 * self.num_anchors + self.size_16 * self.size_16 * self.num_anchors + self.size_32 * self.size_32 * self.num_anchors,
                            self.size_4 * self.size_4 * self.num_anchors + self.size_8 * self.size_8 * self.num_anchors + self.size_16 * self.size_16 * self.num_anchors + self.size_32 * self.size_32 * self.num_anchors + self.size_64 * self.size_64 * self.num_anchors, \
                            self.size_4 * self.size_4 * self.num_anchors + self.size_8 * self.size_8 * self.num_anchors + self.size_16 * self.size_16 * self.num_anchors + self.size_32 * self.size_32 * self.num_anchors + self.size_64 * self.size_64 * self.num_anchors + self.size_128 * self.size_128 * self.num_anchors]
        self.m_youhuasudu = nn.Softmax(dim=0)
        self.GT_BOX_MARGIN=1.2
        # self.conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        # self.conv4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        inputs,target_polyons=x
        fms = self.fpn(inputs)
        feature_roi=[fms[0],fms[1],fms[2],fms[3]]
        #feature_roi = [feature_roi[0], feature_roi[1], feature_roi[2], feature_roi[3]]
        gts_preds, mask_fliter_feature=self.mask(fms)
        loc_preds = []
        cls_preds = []
        for index,fm in enumerate(fms): #for all FPN features
            #feature_roi.append(fm)
            # print('fm.shape={:}'.format(fm.shape))
            if index>0:
                mask_fliter_feature_fm=nn.functional.interpolate(mask_fliter_feature, size=(fm.shape[2], fm.shape[3]), scale_factor=None, mode='bilinear', align_corners=None)
                broadcast = mask_fliter_feature_fm.expand(fm.shape)
            else:
                broadcast=mask_fliter_feature.expand(fm.shape)
            mult = torch.mul(fm, broadcast)
            out = torch.add(fm, mult)
            loc_pred = self.loc_head(out)
            cls_pred = self.cls_head(out)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(inputs.size(0),-1,8)                 # [N,H*W*num_anchors, 8]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(inputs.size(0),-1,self.num_classes)  # [N,H*W*num_anchors, num_classes]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        loc_preds_all=torch.cat(loc_preds,1)
        cls_preds_all=torch.cat(cls_preds,1)
        ########################################################################################
        ########################################################################################
        '''refine bbox location'''
        proposal_num=1000
        proposals = []
        for i in range(inputs.shape[0]):
            '''optimize get mask==1 location index'''
            gts_mask_interpolate = gts_preds[1][i]
            gts_bin_mask_ceshi = (self.m_youhuasudu(gts_mask_interpolate)[1, :, :] > 0.5).int()
            mask_index = (gts_bin_mask_ceshi.view(-1) == 1).nonzero().squeeze().view(-1).cpu().numpy()
            X = mask_index // gts_bin_mask_ceshi.shape[0]
            Y = mask_index % gts_bin_mask_ceshi.shape[1]
            cls_index_ = []
            for num in range(6):  # stage==6
                scale = pow(2, num)
                W = (X / scale).astype(np.int) * (self.size_4 / scale) + (Y / scale).astype(np.int)
                mask_stage_index = np.unique(W.astype(np.int))
                try:
                    mask_stage_index=torch.Tensor(mask_stage_index)
                    match_index1 = self.filter_size[num] + torch.cat(((mask_stage_index*8+0).reshape(-1,1),
                                                                      (mask_stage_index*8+1).reshape(-1,1),
                                                                      (mask_stage_index*8+2).reshape(-1,1),
                                                                      (mask_stage_index*8+3).reshape(-1,1),
                                                                      (mask_stage_index*8+4).reshape(-1,1),
                                                                      (mask_stage_index*8+5).reshape(-1,1),
                                                                      (mask_stage_index*8+6).reshape(-1,1),
                                                                      (mask_stage_index*8+7).reshape(-1,1)),1)
                    a=match_index1[np.arange(0,match_index1.shape[0]), cls_preds_all[i][match_index1.long()].argmax(1).cpu().reshape(-1)]
                    cls_index_.extend(list(a.numpy().astype(np.int)))
                except:
                    a=3
            cls_index_i=cls_index_
            pred_i_score=cls_preds_all[i].sigmoid()
            if len(cls_index_i) >= proposal_num:
                '''fliter too many points ,save proposal_num points'''
                geo_score = pred_i_score[cls_index_i]
                score_rank = geo_score.view(-1).sort(dim=0, descending=True)
                geo_score = score_rank[0][:proposal_num]
                cls_index_i = np.array(cls_index_i)[score_rank[1][:proposal_num].cpu()]
            elif len(cls_index_i) == 0:
                ''' some mask==1 is zero, take random sample method'''
                list_ = list(np.arange(cls_preds_all[0].shape[0]))
                slice_ = random.sample(list_, proposal_num)
                cls_index_i = slice_
                geo_score = pred_i_score[cls_index_i]
            elif len(cls_index_i) > 0 and len(cls_index_i) < proposal_num:
                geo_score = pred_i_score[cls_index_i]
            '''
            if len(cls_index_i) > 0 and len(cls_index_i) < 500:
                leftover=500-len(cls_index_i)
                left_set=set(np.arange(cls_preds_all[0].shape[0]))-set(cls_index_i)
                complementary=pred_i_score[list(left_set)].view(-1).sort(dim=0, descending=True)[1][:leftover]
                cls_index_i.extend(complementary.cpu().numpy().astype(np.int))
                geo_score = pred_i_score[cls_index_i]
                cls_index_i = np.array(cls_index_i)
            '''

            '''get proposal_num points bboxes location'''
            geo_map_loc = self.encoder.anchor_quad_boxes[cls_index_i].cuda() + loc_preds_all[i, cls_index_i,
                                                                          :].contiguous().view(-1, 8) * \
                          self.encoder.anchor_rect_boxes[:, 2:4][cls_index_i].cuda().view(-1, 2).repeat(1, 4)
            geo_map_loc = torch.clamp(geo_map_loc, 0, inputs.shape[2])
            bboxes = convert_polyons_into_angle_upgrade(geo_map_loc)
            bboxes = torch.Tensor(bboxes).cuda()
            """
            Clear some wrong bounding boxes
            Height==0 or Width==0
            """
            bboxes_length=bboxes.shape[0]
            nonzero=bboxes[:, 2:4].cpu().reshape(-1).nonzero().numpy()[:, 0]
            if nonzero.shape[0]<bboxes_length*2:
                zero_index=set(np.arange(0, bboxes_length*2)).difference(set(nonzero))
                zero_index=np.array(list(zero_index)) // 2
                bboxes=bboxes[list(set(np.arange(0,bboxes_length)).difference(zero_index))]
                # print("{:} is zero!".format(bboxes_length*2-nonzero.shape[0]))

            bboxes[:,2:4] *= self.GT_BOX_MARGIN
            boxlist = RBoxList(bboxes, (inputs.shape[2], inputs.shape[3]))
            boxlist.add_field("objectness", geo_score.view(-1) / geo_score.max())
            proposals.append(boxlist)

        if self.bn_type == None:
            for target_polyon in target_polyons:
                target_polyon.bbox = target_polyon.bbox.cuda()
                target_polyon.extra_fields['difficult'] = target_polyon.extra_fields['difficult'].cuda()
                target_polyon.extra_fields['labels'] = target_polyon.extra_fields['labels'].cuda()

        x, result, detector_losses, recur_proposals = self.roi_heads(feature_roi, proposals, target_polyons)
        ########################################################################################
        return loc_preds_all,cls_preds_all,gts_preds,detector_losses,recur_proposals

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=(3, 5), stride=1, padding=(1, 2)))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.fpn.modules():
            if isinstance(layer, nn.BatchNorm2d):
                #print(layer)
                layer.eval()
        """
        for layer in self.fpn.modules():
            if isinstance(layer, nn.BatchNorm2d):
                #print(layer)
                layer.eval()
        for layer in self.roi_heads.modules():
            if isinstance(layer, nn.BatchNorm2d):
                #print(layer)
                layer.eval()
        """


