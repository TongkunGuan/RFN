import os
import numpy as np
import math
import argparse
import zipfile
import cv2
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from tools.augmentations import Augmentation_inference
from tools.model import RFN
from tools.encoder import DataEncoder
from PIL import Image, ImageDraw
from tools.utils import check_and_validate_polys, generate_global_input_images_mask,box_iou_xyxy
from tools.transform import resize
from tensorboardX import SummaryWriter
from maskrcnn_benchmark.config import cfg


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def get_data(img_dir,image_list,index,batchsize,size,build_method):
    inputs = torch.zeros(batchsize, 3, size, size)
    Height = []
    Width = []
    img_name=[]
    current_imgdir=image_list[index*batchsize:(index+1)*batchsize]
    for n,_img in enumerate(current_imgdir):
        img = cv2.imread(img_dir + _img)
        height, width, _ = img.shape
        Height.append(height)
        Width.append(width)
        img_name.append(_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, _, _ = build_method(img)
        inputs[n]=x
    return inputs,Height,Width,img_name
def test_online(eval_device, weight_path, output_path, cls_thresh, nms_thresh, save_img,show_mask):
    path = './data/'
    path1 = './'
    os.environ["CUDA_VISIBLE_DEVICES"] = "{:}".format(eval_device)
    print('PyTorch RFN test_online!')
    input_size = 768
    tune_from = weight_path
    output_zip = output_path
    encoder = DataEncoder(cls_thresh, nms_thresh,input_size)
    batchsize=5
    dataset = 'USTB-SV1K'
    if dataset == "MSC2020_build":
        mean = (0.525, 0.519, 0.510)
        var = (0.279, 0.278, 0.281)
        Augmentation_inference_method = Augmentation_inference(size=input_size, mean=mean, var=var)
    else:
        mean = (0.485, 0.456, 0.406)
        var = (0.229, 0.224, 0.225)
        Augmentation_inference_method = Augmentation_inference(size=input_size, mean=mean, var=var)

    """adding OCR module will update SyncBatchNorm --> BatchNorm"""
    config_file = './configs/R_50_C4_1x_train.yaml'
    cfg.merge_from_file(config_file)
    cfg.freeze()
    net = RFN(1, input_size=input_size,bn_type="test_bn", cfg=cfg, encode=encoder)
    net = net.cuda()

    # load checkpoint
    checkpoint = torch.load(tune_from)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    # test image path
    if dataset in ["MSRA-TD500"]:
        img_dir = path + "MSRA-TD500/test/"
    elif dataset in ["ICDAR2013"]:
        img_dir = path +"/ICDAR2013/Challenge2_Test_Task12_Images/"
    elif dataset in ["ICDAR2017MLT"]:
        img_dir = path + "/ICDAR2017MLT/ICDAR2017MLT_validation/"
    elif dataset in ["USTB-SV1K"]:
        img_dir = path + "USTB-SV1K/testing/"
    val_list = [im for im in os.listdir(img_dir) if "jpg" in im]
    print('dataset_length:{:}'.format(len(val_list)))

    # save results dir & zip
    eval_dir = path1 + "eval_dir/"
    output_zip_list = [output_zip[:-4] + '0', output_zip[:-4] + '1', output_zip[:-4] + '2', output_zip[:-4] + '3', output_zip[:-4] + '4']
    for output_zip in output_zip_list:
        if not os.path.exists(eval_dir+output_zip):
            os.mkdir(eval_dir+output_zip)
    if not os.path.exists(eval_dir + 'MASK_result_image'):
        os.mkdir(eval_dir + 'MASK_result_image')

    # test for each image
    flag=True
    for index in range(math.ceil(len(val_list)//batchsize)):
        print("infer : %d / %d" % (index, math.ceil(len(val_list)//batchsize)), end='\r')
        input,Height,Width,img_name=get_data(img_dir, val_list, index, batchsize, input_size, Augmentation_inference_method)
        x=Variable(input)
        x = x.cuda()
        batch_loc_preds, batch_cls_preds, batch_gts_preds, _, batch_recur_proposals = net((x, None))
        for single_index in range(batchsize):
            ###reserve gts[1] result
            loc_preds, cls_preds, gts_preds, recur_proposals = batch_loc_preds[single_index], batch_cls_preds[single_index], batch_gts_preds[1][single_index], batch_recur_proposals[single_index]
            width=Width[single_index]
            height=Height[single_index]
            _img=img_name[single_index]

            for i, cls_thresh in enumerate([0.2,0.3,0.4,0.5,0.6]):
                if index==0 and flag and single_index==0:
                    result_zip=[]
                    result_zip_0 = zipfile.ZipFile(eval_dir + output_zip_list[0]+'.zip', 'w')
                    result_zip_1 = zipfile.ZipFile(eval_dir + output_zip_list[1]+'.zip', 'w')
                    result_zip_2 = zipfile.ZipFile(eval_dir + output_zip_list[2]+'.zip', 'w')
                    result_zip_3 = zipfile.ZipFile(eval_dir + output_zip_list[3]+'.zip', 'w')
                    result_zip_4 = zipfile.ZipFile(eval_dir + output_zip_list[4]+'.zip', 'w')
                    result_zip.extend([result_zip_0,result_zip_1,result_zip_2,result_zip_3,result_zip_4])
                    flag=False
                save_file = "res_img_%s.txt" % (_img[:-4])
                f = open(eval_dir+output_zip_list[i] + "/res_img_%s.txt" % (_img[:-4]), "w")
                refine_quad_bboxes, refine_quad_scores = encoder.refine_score(recur_proposals, cls_thresh,nms_thresh,gts_preds, cfg.MODEL.RRPN.GT_BOX_MARGIN,input_size,0.2)
                quad_boxes = refine_quad_bboxes
                if quad_boxes.shape[0] is 0:
                    f.close()
                    result_zip[i].write(eval_dir+output_zip_list[i] + "/" + save_file, save_file, compress_type=zipfile.ZIP_DEFLATED)
                    # os.remove(output_zip_list[i] + "/res_%s.txt" % (_img[:-4]))
                    continue
                quad_boxes /= input_size
                quad_boxes *= ([[width, height]] * 4)
                quad_boxes = quad_boxes.astype(np.int32)
                quad_boxes = check_and_validate_polys(quad_boxes)
                _quad = []
                for quad in quad_boxes:
                    if dataset in ["MSC2020_build",'MSRA-TD500','ICDAR2017MLT','USTB-SV1K']:
                        [x0, y0], [x1, y1], [x2, y2], [x3, y3] = quad
                        f.write("%d,%d,%d,%d,%d,%d,%d,%d\n" % (x0, y0, x1, y1, x2, y2, x3, y3))
                    elif dataset in ["ICDAR2013"]:
                        xmin = np.min(quad[:, 0])
                        ymin = np.min(quad[:, 1])
                        xmax = np.max(quad[:, 0])
                        ymax = np.max(quad[:, 1])
                        f.write("%d,%d,%d,%d\n" % (xmin, ymin, xmax, ymax))
                f.close()
                result_zip[i].write(eval_dir+output_zip_list[i] + "/" + save_file, save_file, compress_type=zipfile.ZIP_DEFLATED)
                # os.remove(output_zip_list[i] + "/res_%s.txt" % (_img[:-4]))

                if save_img and cls_thresh==0.1:
                    # draw predict points
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = np.array(img, dtype=np.uint8)
                    # draw gt points Challenge2_Training_Task1_GT
                    gt_anno = open("../../../../../ICDAR2017MLT/ICDAR2017MLT_validation_GT/gt_%s.txt" % (_img[:-4]), "r")
                    gt_anno = gt_anno.readlines()
                    for label in gt_anno:
                        _x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, _,txt = label.split(",")[:10]
                        color = (0, 255, 0)
                        if "###" in txt:
                            color = (0, 255, 255)
                        _x0 = int(_x0)
                        gt_point = np.array([_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3], dtype=np.int32)
                        gt_point = gt_point.reshape(-1, 4, 2)
                        img = cv2.polylines(img, [gt_point], True, color, 4)
                    quad_boxes =quad_boxes.reshape(-1, 4, 2)
                    img = cv2.polylines(img, quad_boxes, True, (0,0,255), 4)
                    """
                    ICDAR2013 dataset
                    """
                    # for label in gt_anno:
                    #     _xmin, _ymin, _xmax, _ymax = label.split(",")[:4]
                    #     img = cv2.rectangle(img, (int(_xmin), int(_ymin)), (int(_xmax), int(_ymax)), (0, 255, 0), 4)
                    # for quad in quad_boxes:
                    #     xmin = np.min(quad[:, 0])
                    #     ymin = np.min(quad[:, 1])
                    #     xmax = np.max(quad[:, 0])
                    #     ymax = np.max(quad[:, 1])
                    #     img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 4)

                    save_img_dir = "./eval_dir/Baseline_predict_image/%s"
                    cv2.imwrite(save_img_dir % (_img), img)
                    if show_mask:
                        save_img_dir = "./eval_dir/Baseline_predict_image/%s_mask.jpg"
                        x3 = gts_preds[1][0, 1, :, :].sigmoid()
                        x3 = x3.cpu().detach().numpy()
                        x4 = x3 / x3.max() * 255
                        cv2.imwrite(save_img_dir % (_img[:-4]), x4)
        del input,x,batch_loc_preds, batch_cls_preds, batch_gts_preds, _, batch_recur_proposals,loc_preds, cls_preds, gts_preds, recur_proposals
    for result_zip_i in result_zip:
        result_zip_i.close()
    del net
    torch.cuda.empty_cache()
    if dataset in ["ICDAR2013","ICDAR2017MLT"]:
        import subprocess
        # gt_path="/home/amax/GTK/Modify_Textboxes++/GT_MSC_Test.zip"
        gt_path = "/home/amax/GTK/ICDAR2017MLT/ICDAR2017MLT_validation_GT1.zip"
        max_scorestring0 = ''
        max_hmean0 = 0.0
        for output_zip in output_zip_list:
            print('_'*100)
            query0 = "python %s../eval/ICDAR2017/script.py -g=%s -s=%s" % (path1, gt_path, eval_dir + output_zip)
            scorestring0 = subprocess.check_output(query0, shell=True)
            hmean0 = float(str(scorestring0).strip().split(":")[3].split(",")[0].split("}")[0].strip())
            precise0 = float(str(scorestring0).strip().split(":")[1].split(",")[0].split("}")[0].strip())
            recall0 = float(str(scorestring0).strip().split(":")[2].split(",")[0].split("}")[0].strip())
            print("ICDAR2013:test_hmean for cls_threshold: {:.4f}  precise:{:.4f}   recall:{:.4f}".format(hmean0,precise0, recall0))
            if hmean0 > max_hmean0:
                max_hmean0 = hmean0
                max_scorestring0 = scorestring0
        print(max_scorestring0)
        return max_scorestring0
    elif dataset in ["USTB-SV1K","MSRA-TD500"]:
        sys.path.append(os.getcwd() + '/eval/')
        from MSRA.eval_MSRA import get_sv1k_result
        gt_root = "./data/USTB-SV1K/SV1K_GT/"
        hmean_max=0
        for i in range(5):
            pred_root = eval_dir + output_path[:-4] + '{:}'.format(i)
            scorestring,hmean = get_sv1k_result(gt_root,pred_root)
            print(scorestring)
            if hmean>hmean_max:
                max_scorestring = scorestring
                hmean_max = hmean
        return max_scorestring
# path='/media/amax/guantongkun/Textdet_comparative_experiments/RFFNET++ICDAR2017MLT/7500/ckpt_7500.pth'
# test_online("1", path, "Mask_result.zip", 0.3, 0.1, save_img=False,show_mask=False)
# for pth in sorted(os.listdir(path)):
#     print(pth)
#     test_online("1", path+pth, "Mask_result.zip", 0.3, 0.1, False,False)
