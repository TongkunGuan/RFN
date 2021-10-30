from __future__ import print_function

import time
import os
import sys
import argparse
import numpy as np
import cv2
from subprocess import Popen, PIPE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils

from tensorboardX import SummaryWriter
sys.path.append(os.getcwd()+'/tools/')
from augmentations import Augmentation_traininig, Resize
from loss import FocalLoss, OHEM_loss
from model import RFN
from datagen import ListDataset
from encoder import DataEncoder
from maskrcnn_benchmark.config import cfg
from multi_image_test_ocr import test_online
import os
import warnings
from maskrcnn_benchmark.structures.bounding_box import RBoxList
from utils import change_box_order,convert_polyons_into_angle,convert_polyons_into_angle_cuda,convert_angle_into_polygons
warnings.filterwarnings("ignore")
device=0
os.environ["CUDA_VISIBLE_DEVICES"] = "{:}".format(device)

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def adjust_learning_rate(cur_lr, optimizer, gamma, step):
    lr = cur_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

parser = argparse.ArgumentParser(description='PyTorch RFN Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--input_size', default=768, type=int, help='Input size for training')
parser.add_argument('--batch_size', default=11, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=11, type=int, help='Number of workers used in dataloading')
parser.add_argument('--resume', default='/media/amax/guantongkun/Textdet_comparative_experiments/RFN++SV1K/ckpt_1100.pth', type=str,help='resume from checkpoint')  # '/home/amax/GTK/improve_ocr/Pytorch/weights/multi_step1/build_global_mask_loss_v4/ckpt_90000.pth'
parser.add_argument('--dataset', default='USTB-SV1K', type=str, help='select training dataset')
parser.add_argument('--multi_scale', default=False, type=str2bool, help='Use multi-scale training')
parser.add_argument('--focal_loss', default=True, type=str2bool, help='Use Focal loss or OHEM loss')
parser.add_argument('--logdir', default='./Final_log/', type=str, help='Tensorboard log dir')
parser.add_argument('--max_iter', default=1200000, type=int, help='Number of training iterations')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_interval', default=100, type=int, help='Location to save checkpoint models')
parser.add_argument('--save_folder', default='/media/amax/guantongkun/Textdet_comparative_experiments/RFN++SV1K/', help='Location to save checkpoint models')
parser.add_argument('--evaluation', default=True, type=str2bool, help='Evaulation during training')
parser.add_argument('--eval_step', default=100, type=int, help='Evauation step')
parser.add_argument('--eval_device', default=1, type=int, help='GPU device for evaluation')
parser.add_argument('--seed', default=5, type=int, help='random seed')
parser.add_argument('--summary_iter', default=100, type=int, help='write summary')
parser.add_argument('--training_visualization_iter', default=100, type=int, help='draw training image')
parser.add_argument('--config_file', default='./configs/R_50_C4_1x_train.yaml', type=str, help='default parameters')
parser.add_argument('--eval_dir', default='./eval_dir/', type=str, help='evaluation dir')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.logdir):
    os.mkdir(args.logdir)
if not os.path.exists(args.eval_dir):
    os.mkdir(args.eval_dir)
if args.training_visualization_iter and not os.path.exists(args.eval_dir+'training_visualization/'):
    os.mkdir(args.eval_dir+'training_visualization/')

# set random seed
if args.seed > 0:
    import random
    print('Seeding with', args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# Data
if args.dataset in ["MSC2020_build","S-MSC2020"]:
    mean = (0.525, 0.519, 0.510)
    var = (0.279, 0.278, 0.281)
    Augmentation_traininig_method=Augmentation_traininig(size=args.input_size,mean=mean,var=var)
elif args.dataset in ["MSRA-TD500","ICDAR2013","ICDAR2017MLT","USTB-SV1K"]:
    mean = (0.485, 0.456, 0.406)
    var = (0.229, 0.224, 0.225)
    Augmentation_traininig_method=Augmentation_traininig(size=args.input_size,mean=mean,var=var)
elif args.dataset in ["SynthText"]:
    mean =(0.465, 0.453, 0.416)
    var = (0.295, 0.282, 0.302)
    Augmentation_traininig_method=Augmentation_traininig(size=args.input_size,mean=mean,var=var)

#load dataset
encoder = DataEncoder(cls_thresh=0.35,nms_thresh=0.1,input_size=args.input_size)
print('==> Preparing data..')
trainset = ListDataset(root="data/USTB-SV1K/", dataset=args.dataset, train=True,
                       transform=Augmentation_traininig_method, input_size=args.input_size, multi_scale=args.multi_scale,encoder=encoder)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,shuffle=True, collate_fn=trainset.collate_fn)

# set model (focal_loss vs OHEM_CE loss)
if args.focal_loss:
    imagenet_pretrain = 'weights/retinanet_se50_with_mask.pth'
    if args.dataset=="MSC2020_build":
        criterion = FocalLoss(loss_seg=True)
    else:
        criterion = FocalLoss(loss_seg=False)
    num_classes = 1
else:
    imagenet_pretrain = 'weights/retinanet_se50_OHEM.pth'
    criterion = OHEM_loss()
    num_classes = 2

# Training Detail option
if args.dataset in ["SynthText"]:
    stepvalues = (5, 10, 15)
elif args.dataset in ["MSC2020_build"]:
    stepvalues = (56,100,122,140)
else:
    stepvalues = None
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch
iteration = 0
cur_lr = args.lr
step_index = 0
new_epoch = 0

# Model
cfg.merge_from_file(args.config_file)
cfg.freeze()
net = RFN(num_classes,input_size=args.input_size,bn_type=None,cfg=cfg,encode=encoder)
net.load_state_dict(torch.load(imagenet_pretrain))

print("input_size : ", args.input_size)
print("stepvalues : ", stepvalues)
print("start_epoch : ", start_epoch)
print("iteration : ", iteration)
print("cur_lr : ", cur_lr)
print("step_index : ", step_index)
print("num_gpus : ", torch.cuda.device_count())

if args.resume:
    print('==> Resuming from checkpoint..', args.resume)
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    cur_lr = checkpoint['lr']
    step_index = checkpoint['step_index']
    #optimizer.load_state_dict(state["optimizer"])

net.cuda()
net.train()
net.freeze_bn()  # you must freeze batchnorm
optimizer = optim.SGD(net.parameters(), lr=cur_lr, momentum=0.9, weight_decay=1e-4)
writer = SummaryWriter(log_dir=args.logdir)
t0 = time.time()

# Training
for epoch in range(start_epoch, 10000):
    if iteration > args.max_iter:
        break
    # initial parameters
    gts_loss_batch=[]
    cls_loss_batch = []
    loc_loss_batch = []
    loss_batch = []
    all_idx = []
    FN = []
    FP = []
    roi_loss_classifier_batch = []
    roi_loss_box_reg_batch = []
    detector_losses = {"loss_classifier": 0.0, "loss_box_reg": 0.0}
    recur_proposals = None

    """
    nature scene text dataset: we adopt the following formula to adjust the learning rate
    cur_lr = adjust_learning_rate(args.lr, optimizer, 0.998, epoch)
    MPSC SynthMPSC SynthText dataset: we set a stepwise adjustment of the learning rate 
    """
    if args.dataset in ["MSRA-TD500","ICDAR2013","ICDAR2017MLT","USTB-SV1K"]:
        cur_lr = adjust_learning_rate(args.lr, optimizer, 0.95, epoch)
    else:
        if epoch in stepvalues:
            flag = new_epoch == epoch
            if not flag:
                step_index += 1
                cur_lr = adjust_learning_rate(cur_lr, optimizer, args.gamma, step_index)
            new_epoch = epoch

    t0 = time.time()
    for inputs, loc_targets, cls_targets, gts_masks,target_polyons in trainloader:
        inputs = Variable(inputs.cuda())  # (batch,3,size,size)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        gts_masks = Variable(gts_masks.cuda())

        optimizer.zero_grad()
        loc_preds,cls_preds,gts_preds,detector_losses,recur_proposals = net((inputs,target_polyons))

        loss_classifier=detector_losses['loss_classifier']
        loss_box_reg=detector_losses['loss_box_reg']
        loc_loss, cls_loss,gts_loss,fn,fp = criterion(loc_preds, loc_targets, cls_preds, cls_targets,gts_preds,gts_masks,iteration)
        loss = loc_loss + cls_loss+gts_loss+loss_box_reg+loss_classifier
        if torch.isnan(loss) or torch.isinf(loss):
            del target_polyons,inputs,loc_targets,cls_targets,gts_masks,loc_preds, cls_preds, gts_preds, detector_losses, \
                recur_proposals,loss_classifier,loss_box_reg,loc_loss, cls_loss,gts_loss,fn,fp,loss
            print("Dirty data in the current step! Break!")
            break
        else:
            loss.backward()
            optimizer.step()
        loc_loss_batch.append(loc_loss)
        cls_loss_batch.append(cls_loss)
        gts_loss_batch.append(gts_loss)
        roi_loss_box_reg_batch.append(loss_box_reg)
        roi_loss_classifier_batch.append(loss_classifier)
        loss_batch.append(loss)
        FN.append(fn)
        FP.append(fp)

        if iteration % args.summary_iter == 0:
            if loss.cpu() > torch.Tensor([100]):
                print('Abnormal loss, program suspension!')
                break
            t1 = time.time()
            print('iter ' + repr(iteration) + ' (epoch ' + repr(epoch) + ') || loss: %.4f || l loc_loss: %.4f || l cls_loss: %.4f || l gts_loss: %.4f ||  l loss_box_reg: %.4f || l loss_classifier: %.4f (Time : %.1f)'\
                  % (torch.Tensor(loss_batch).mean(), torch.Tensor(loc_loss_batch).mean(), torch.Tensor(cls_loss_batch).mean(), torch.Tensor(gts_loss_batch).mean(), torch.Tensor(roi_loss_box_reg_batch).mean(),torch.Tensor(roi_loss_classifier_batch).mean(), (t1 - t0)))
            t0 = time.time()
            writer.add_scalar('loc_loss', torch.Tensor(loc_loss_batch).mean(), iteration)
            writer.add_scalar('cls_loss', torch.Tensor(cls_loss_batch).mean(), iteration)
            writer.add_scalar('gts_loss', torch.Tensor(gts_loss_batch).mean(), iteration)
            writer.add_scalar('roi_loss_box_reg', torch.Tensor(roi_loss_box_reg_batch).mean(), iteration)
            writer.add_scalar('roi_loss_classifier', torch.Tensor(roi_loss_classifier_batch).mean(), iteration)
            writer.add_scalar('loss', torch.Tensor(loss_batch).mean(), iteration)
            writer.add_scalar('learning_rate', cur_lr, iteration)
            writer.add_scalar('FN', torch.Tensor(FN).mean(), iteration)
            writer.add_scalar('FP', torch.Tensor(FP).mean(), iteration)
            gts_loss_batch = []
            cls_loss_batch = []
            loc_loss_batch = []
            roi_loss_classifier_batch = []
            roi_loss_box_reg_batch = []
            loss_batch = []
            FN = []
            FP = []

        if iteration % args.training_visualization_iter == 0:
            # show inference image in local file system
            infer_img = inputs[0].permute(1, 2, 0)
            infer_img *= torch.Tensor(var).cuda()
            infer_img += torch.Tensor(mean).cuda()
            infer_img *= 255.
            infer_img = torch.clamp(infer_img, 0, 255, out=None)
            # infer_img = infer_img.astype(np.uint8)
            h, w, _ = infer_img.shape
            # infer_mask=gts_masks[0].cpu().numpy()
            boxes, labels, scores = encoder.decode(loc_preds[0], cls_preds[0],(w, h))
            boxes = boxes.reshape(-1, 4, 2).astype(np.int32)
            img = cv2.polylines(infer_img.cpu().numpy(), boxes, True, (255, 0, 0), 4)
            # writer.add_image('prep_result', np.transpose(img,(2,0,1)), iteration)
            if recur_proposals != None:
                bboxes_np, _ = encoder.refine_score(recur_proposals[0], 0.35, 0.1, gts_preds,
                                                    cfg.MODEL.RRPN.GT_BOX_MARGIN, args.input_size,0.3)
                bboxes_np = bboxes_np.reshape(-1, 4, 2).astype(np.int32)
                refine_img = cv2.polylines(infer_img.cpu().numpy(), bboxes_np, True, (255, 0, 0), 4)
                # writer.add_image('refine_result', np.transpose(refine_img, (2, 0, 1)), iteration)

            gt_box=convert_angle_into_polygons(target_polyons[0].bbox.data.cpu())
            gt_box = gt_box.reshape(-1, 4, 2).astype(np.int32)
            Gt_img=cv2.polylines(infer_img.cpu().numpy(), gt_box, True, (255, 0, 0), 4)
            # x2 = gts_preds[0][0, 1, :, :].sigmoid()
            x3 = gts_preds[1][0, 1, :, :].sigmoid()
            x3 = x3.cpu().detach().numpy()
            x4 = x3 / x3.max() * 255
            # x4 = torch.cat([x2,x3],1)
            # img_grid = vutils.make_grid(x4, normalize=True, scale_each=True, nrow=1)
            # writer.add_image('gt_prep_mask', img_grid, iteration)
            cv2.imwrite('./eval_dir/training_visualization/{:}_img.jpg'.format(iteration), img)
            cv2.imwrite('./eval_dir/training_visualization/{:}_refine_img.jpg'.format(iteration), refine_img)
            cv2.imwrite('./eval_dir/training_visualization/{:}_gt.jpg'.format(iteration), Gt_img)
            cv2.imwrite('./eval_dir/training_visualization/{:}_pred.jpg'.format(iteration), x4)

        if iteration % args.save_interval == 0:
            print('Saving state, iter : ', iteration)
            state = {
                'net': net.state_dict(),
                "optimizer": optimizer.state_dict(),
                'iteration': iteration,
                'epoch': epoch,
                'lr': cur_lr,
                'step_index': step_index
            }
            model_file = args.save_folder + 'ckpt_' + repr(iteration) + '.pth'
            torch.save(state, model_file)

        if args.evaluation and iteration % args.eval_step == 0 and iteration>0:
            del target_polyons, inputs, loc_targets, cls_targets, gts_masks, loc_preds, cls_preds, gts_preds, detector_losses, \
                recur_proposals, loss_classifier, loss_box_reg, loc_loss, cls_loss, gts_loss, fn, fp, loss
            torch.cuda.empty_cache()
            Flag = 0
            try:
                scorestring0= test_online(eval_device=device, weight_path=model_file, \
                                                           output_path='MASK_result.zip', cls_thresh=0.3,
                                                           nms_thresh=0.1,save_img=False,show_mask=False)
                hmean = float(str(scorestring0).strip().split(":")[3].split(",")[0].split("}")[0].strip())
                precise = float(str(scorestring0).strip().split(":")[1].split(",")[0].split("}")[0].strip())
                recall = float(str(scorestring0).strip().split(":")[2].split(",")[0].split("}")[0].strip())
                writer.add_scalar('hmean', hmean, iteration)
                writer.add_scalar('precise', precise, iteration)
                writer.add_scalar('recall', recall, iteration)
            except Exception as e:
                print("exception happened in evaluation ", e)

        iteration += 1
        if iteration > args.max_iter:
            break





