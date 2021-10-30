from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils import one_hot_embedding, one_hot_v3
from torch.autograd import Variable
import cv2
import numpy as np
from shapely.geometry import Polygon
from utils import check_and_validate_polys
def intersection(g, p):
    g=np.asarray(g)
    p=np.asarray(p)
    g = Polygon(g)
    p = Polygon(p)
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        probs=F.softmax(logits,dim=1)
        ###foreground
        num = targets.size(0)
        smooth = 1
        
        fore_probs =probs[:,1,:,:] #F.sigmoid(logits[:,1,:,:])
        m1 = fore_probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = ((2. * intersection.sum(1)) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        fore_score = 1 - score.sum() / num
        ###background
        back_probs = probs[:,0,:,:]#F.sigmoid(logits[:,0,:,:])
        m1 = back_probs.view(num, -1)
        m2 = 1-targets.view(num, -1)
        intersection = (m1 * m2)

        score = ((2. * intersection.sum(1)) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        back_score = 1 - score.sum() / num
        #print("fore_score:{:}".format(fore_score))
        #print("back_score:{:}".format(back_score))
        return fore_score+back_score
class Innovation_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Innovation_Loss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        x2 = targets.view(num, -1)
        delta = 0.01 * x2.sum(1)

        probs = F.softmax(logits, dim=1)[:, 1, :, :]
        Fn = ((probs - targets) >= 0.5).float() * probs
        Fn = Fn.view(num, -1).sum(1)
        Fp = ((targets - probs) >= 0.5).float() * (1-probs)
        Fp = Fp.view(num, -1).sum(1)
        Fp = Fp/(x2.sum(1)+1)
        Fn = (Fn - delta) * (Fn - delta > 0).float() / (x2.sum(1) + 1)
        hinge_loss = Fp+Fn

        # score=torch.clamp(limit_max,0,1)+torch.clamp(limit_min,0,1)
        if torch.isnan(Fn.mean().cpu()):
            print("debug")

        score = torch.clamp(hinge_loss, 0, 1)

        score = score.sum() / num
        # print("limit_min:{:}  limit_max:{:}".format(limit_min.mean(),limit_max.mean()))
        # print("limit_max:{:}".format(limit_max.mean()))
        return score,Fn.mean(),Fp.mean()
class EdgeSaliencyLoss(nn.Module):
    def __init__(self, alpha_sal=0.7):
        super(EdgeSaliencyLoss, self).__init__()

        self.alpha_sal = alpha_sal

        self.laplacian_kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], dtype=torch.float, requires_grad=False)
        self.laplacian_kernel = self.laplacian_kernel.view((1, 1, 3, 3))  # Shape format of weight for convolution

    @staticmethod
    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (1 - target) * torch.log(
            1 - input_ + eps)
        return torch.mean(wbce_loss)

    def forward(self, y_pred, y_gt):
        edge=self.laplacian_kernel.cuda()
        # Generate edge maps
        y_gt_edges = F.relu(torch.tanh(F.conv2d(y_gt, edge, padding=(1, 1))))
        y_pred_edges = F.relu(torch.tanh(F.conv2d(y_pred, edge, padding=(1, 1))))

        # sal_loss = F.binary_cross_entropy(input=y_pred, target=y_gt)
        sal_loss = self.weighted_bce(input_=y_pred, target=y_gt, weight_0=1.0, weight_1=1.12)
        edge_loss = F.binary_cross_entropy(input=y_pred_edges, target=y_gt_edges)

        total_loss = self.alpha_sal * sal_loss + (1 - self.alpha_sal) * edge_loss
        return total_loss
class FocalLoss(nn.Module):
    def __init__(self,loss_seg=False):
        super(FocalLoss, self).__init__()
        self.num_classes = 1
        self.loss_seg=loss_seg
        self.gts_loss=torch.nn.BCEWithLogitsLoss()#torch.nn.BCELoss()
        self.m=nn.Sigmoid()
        self.Dice=SoftDiceLoss()
        self.Innovation=Innovation_Loss()
        #self.EdgeSaliencyLoss=EdgeSaliencyLoss()

    def weighted_bce(input_, target, weight_0=1.0, weight_1=1.0, eps=1e-15):
        wbce_loss = -weight_1 * target * torch.log(input_ + eps) - weight_0 * (1 - target) * torch.log(
            1 - input_ + eps)
        return torch.mean(wbce_loss)
    def Diceloss(self,global_text_segs,gts_masks):
        global_mask=nn.functional.interpolate(gts_masks.unsqueeze(0), size=(global_text_segs.shape[2], global_text_segs.shape[3]), scale_factor=None, mode='bilinear', align_corners=None)
        score=self.Dice(global_text_segs,global_mask.squeeze())
        return score
    def Cover_iou_Loss(self,global_text_segs,gts_masks):
        global_mask=nn.functional.interpolate(gts_masks.unsqueeze(0), size=(global_text_segs.shape[2], global_text_segs.shape[3]), scale_factor=None, mode='bilinear', align_corners=None)
        score=self.Innovation(global_text_segs,global_mask.squeeze())
        return score
    def cross_entropy(self,global_text_segs,gts_masks):
        global_mask = nn.functional.interpolate(gts_masks.unsqueeze(0), size=(global_text_segs.shape[2], global_text_segs.shape[3]),scale_factor=None, mode='bilinear',align_corners=None)

        input_global_masks = global_mask.view([-1]).long()
        masks = one_hot_embedding(input_global_masks, 2)
        """ use BCELoss
        sigmoid_prep_mask = self.m(global_text_segs)
        pred_masks = sigmoid_prep_mask.permute(0, 2, 3, 1).contiguous().view([-1, 2])
        ###use pos_weight
        pos_weight=torch.Tensor([1.0,(masks.size()-masks.sum())/masks.sum()])
        loss = self.weighted_bce(pred_masks, masks.cuda(),weight_0=pos_weight[0], weight_1=pos_weight[1])
        """
        """use unbalance BCEwithLogitloss
        pos_weight=torch.Tensor([1.0,(masks.size()-masks.sum())/masks.sum()])
        """
        pred_masks = global_text_segs.permute(0, 2, 3, 1).contiguous().view([-1, 2])
        assert pred_masks.shape[0] == masks.shape[0]
        loss = self.gts_loss(pred_masks, masks.cuda())
        return loss
    def build_global_mask_loss1(self,global_text_segs,gts_masks):
        """
        global_text_segs=gts_preds#(6,4,2,192,192)
        gts_masks=gts_masks_mults
        
        """
        losses = torch.Tensor([0]).cuda()
        
        for index in range(len(global_text_segs)):
            global_mask=nn.functional.interpolate(gts_masks.unsqueeze(0), size=(global_text_segs[index].shape[2], global_text_segs[index].shape[3]), scale_factor=None, mode='bilinear', align_corners=None) 
            input_global_masks = global_mask.view([-1]).long()
            masks=one_hot_embedding(input_global_masks,2)
            sigmoid_prep_mask=self.m(global_text_segs[index])
            pred_masks = sigmoid_prep_mask.permute(0,2,3,1).contiguous().view([-1,2])
            assert pred_masks.shape[0]==masks.shape[0]
            if input_global_masks.size()[0]>0:
                loss=self.gts_loss(pred_masks,masks.cuda())
            else:
                loss=torch.Tensor([0.0])[0]
            losses+=loss
        return losses/(index+1)
    def multi_scales_focal_loss(self,global_text_segs,gts_masks):
        loss=0.0
        for index in range(len(global_text_segs)):
            global_mask=nn.functional.interpolate(gts_masks.unsqueeze(0), size=(global_text_segs[index].shape[2], global_text_segs[index].shape[3]), scale_factor=None, mode='bilinear', align_corners=None) 
            loss+=self.focal_loss(global_text_segs[index],global_mask.squeeze().long())
        return loss/(index+1)
    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        gamma = 2
        alpha = torch.Tensor([0.25,1.75])
        if x.dim()>2:
            x = x.view(x.size(0),x.size(1),-1)  # N,C,H,W => N,C,H*W
            x = x.transpose(1,2)    # N,C,H*W => N,H*W,C
            x = x.contiguous().view(-1,x.size(2))   # N,H*W,C => N*H*W,C
        target = y.view(-1,1)

        logpt = F.log_softmax(x)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if alpha is not None:
            if alpha.type()!=x.data.type():
                alpha = alpha.type_as(x.data)
            at = alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**gamma * logpt
        return loss.sum()

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        t = one_hot_embedding(y.data.cpu(), self.num_classes+1)
        t = t[:,1:]
        t = Variable(t).cuda()
        
        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets,global_text_segs,gts_masks,iteration):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 8].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.float().sum()
        
        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,8]
        masked_loc_preds = loc_preds[mask].view(-1,8)      # [#pos,8]
        masked_loc_targets = loc_targets[mask].view(-1,8)  # [#pos,8]
        
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        #loc_loss *= 0.5  # TextBoxes++ has 8-loc offset
        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])


        aux_mask=global_text_segs[0]
        seg_mask=global_text_segs[1]
        if self.loss_seg:
            r = 0.1
            if iteration > 8000:
                gts_dice_loss = (self.Diceloss(aux_mask, gts_masks) + self.Diceloss(seg_mask, gts_masks)) / 2
                gts_loss1,fn1,fp1 = self.Cover_iou_Loss(aux_mask, gts_masks)
                gts_loss2,fn2,fp2 = self.Cover_iou_Loss(seg_mask, gts_masks)
                gts_loss=(gts_loss1+gts_loss2)/2
                fn=fn1+fn2
                fp=fp1+fp2
                Mixed_loss = (gts_dice_loss + (-1 * gts_dice_loss * r).exp() * gts_loss)
            else:
                gts_dice_loss = (self.Diceloss(aux_mask, gts_masks) + self.Diceloss(seg_mask, gts_masks)) / 2
                cel_loss = (self.cross_entropy(aux_mask, gts_masks) + self.cross_entropy(seg_mask, gts_masks)) / 2
                Mixed_loss = cel_loss + gts_dice_loss
                fp=0.0
                fn=0.0
        else:
            gts_dice_loss = (self.Diceloss(aux_mask, gts_masks) + self.Diceloss(seg_mask, gts_masks)) / 2
            cel_loss = (self.cross_entropy(aux_mask, gts_masks) + self.cross_entropy(seg_mask, gts_masks)) / 2
            Mixed_loss = cel_loss + gts_dice_loss
            fp = 0.0
            fn = 0.0
        return torch.clamp(loc_loss/num_pos,0,5), cls_loss/num_pos, Mixed_loss,fn,fp

    
    
def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


class OHEM_loss(nn.Module):
    def __init__(self):
        super(OHEM_loss, self).__init__()
        self.num_classes = 2
        self.negpos_ratio = 3
        self.gts_loss=torch.nn.BCEWithLogitsLoss()#torch.nn.CrossEntropyLoss()
    def build_global_mask_loss(self,global_text_segs,gts_masks):
        """
        Compute gloable text segmentation mask loss
        input_global_masks: [batch_size, height, width]
        pred_global_masks: dict{p2, p3, p4, p5}
                           batch_size, height, width, 2]
        """
        losses = []
        # One hot encoding: [batch_size * height * width, 2]
        '''
        gts_masks=torch.randn((2,1,768,768))
        gts_masks=torch.ByteTensor(gts_masks>0)
        input_global_masks = gts_masks.long().view([-1])
        label=np.array(input_global_masks)
        num_labels = label.shape[0]
        index_offset = np.arange(num_labels) * 2
        labels_onehot = np.zeros((num_labels,2))
        labels_onehot.flat[index_offset + label.ravel()] = 1  
        '''
        input_global_masks = gts_masks.view([-1,1]).long()
        input_global_masks = torch.zeros(input_global_masks.shape[0],self.num_classes).scatter_(1,input_global_masks.cpu(),1)

        for i in range(len(global_text_segs)):
            # [batch_size * h * w, 2] 
            pred_masks = global_text_segs[i].view([-1,self.num_classes])
            assert pred_masks.shape[0]==input_global_masks.shape[0]
            if input_global_masks.size()[0]>0:
                loss=self.gts_loss(pred_masks,input_global_masks.cuda())
            else:
                loss=torch.Tensor([0.0])[0]
            losses.append(loss)
        losses = torch.Tensor(losses)
        loss = torch.mean(losses)
        return loss
    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets,global_text_segs,gts_masks):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 8].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        cls_targets = cls_targets.clamp(0, 1)   #remove ignore (-1)
        pos = cls_targets > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,8]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        masked_loc_preds = loc_preds[pos_idx].view(-1, 8)
        masked_loc_targets = loc_targets[pos_idx].view(-1, 8)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        # Compute max conf across batch for hard negative mining
        num = loc_preds.size(0)
        batch_conf = cls_preds.view(-1, self.num_classes)
        
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, cls_targets.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(cls_preds)
        neg_idx = neg.unsqueeze(2).expand_as(cls_preds)
        
        conf_p = cls_preds[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = cls_targets[(pos+neg).gt(0)]
        cls_loss = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.float().sum()
        loc_loss /= N
        cls_loss /= N
        gts_loss=self.build_global_mask_loss(global_text_segs,gts_masks)
        return loc_loss, cls_loss,gts_loss

    
def Debug():
    loc_preds = torch.randn((2, 6, 8))
    loc_targets = torch.randn((2, 6, 8))
    
    cls_preds = torch.randn((2, 6, 2))
    cls_targets = torch.randint(0, 2, (2, 6)).type(torch.LongTensor)
    
    gts_masks=torch.randn((2,1,768,768))
    gts_masks=torch.ByteTensor(gts_masks>0).long()
    
    #print(cls_targets.data)
    gts_preds=list()
    ohem = OHEM_loss()
    a,b,c=ohem.forward(loc_preds, loc_targets, cls_preds, cls_targets,gts_preds,gts_masks)
    return a,b,c
#a,b,c=Debug()