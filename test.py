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
from tqdm import tqdm
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

parser = argparse.ArgumentParser(description='PyTorch RFN Testing')
parser.add_argument('--input_size', default=768, type=int, help='Input size for testing')
parser.add_argument('--batchsize', default=1, type=int, help='Batch size for testing')
parser.add_argument('--model_path', default='weights/ckpt_30000.pth', type=str,help='resume from checkpoint')
parser.add_argument('--dataset', default='MPSC', type=str, help='select testing dataset')
parser.add_argument('--eval_device', default=1, type=int, help='GPU device for evaluation')
parser.add_argument('--config_file', default='./configs/R_50_C4_1x_train.yaml', type=str, help='default parameters')
parser.add_argument('--output_zip', default='RFN.zip', type=str, help='out dir')
parser.add_argument('--cls_thresh', default=0.35, type=float, help='classification threshold')
parser.add_argument('--nms_thresh', default=0.1, type=float)
parser.add_argument('--lamda', default=0.35, type=float)
parser.add_argument('--eval', action='store_true', help='eval testset')
parser.add_argument('--test', action='store_true', help='test image')
parser.add_argument('--save_img', action='store_true', help='store predicition')
parser.add_argument('--show_mask', action='store_true', help='store mask')
args = parser.parse_args()

if args.test:
    ###Configuration parameter
    path = './data/' ### dataset path
    os.environ["CUDA_VISIBLE_DEVICES"] = "{:}".format(args.eval_device)
    encoder = DataEncoder(args.cls_thresh, args.nms_thresh,args.input_size)
    if args.dataset == "MPSC":
        mean = (0.525, 0.519, 0.510)
        var = (0.279, 0.278, 0.281)
        Augmentation_inference_method = Augmentation_inference(size=args.input_size, mean=mean, var=var)
    else:
        mean = (0.485, 0.456, 0.406)
        var = (0.229, 0.224, 0.225)
        Augmentation_inference_method = Augmentation_inference(size=args.input_size, mean=mean, var=var)

    # test image path
    if args.dataset in ["ICDAR2015"]:
        img_dir = path + "ICDAR2015/ch4_test_images/"
    elif args.dataset in ["ICDAR2013"]:
        img_dir = path +"/ICDAR2013/Challenge2_Test_Task12_Images/"
    elif args.dataset in ["ICDAR2017MLT"]:
        img_dir = path + "/ICDAR2017MLT/ICDAR2017MLT_validation/"
    elif args.dataset in ["MPSC"]:
        img_dir = path + "/MPSC/MSC_Dataset/test/"
    val_list = [im for im in os.listdir(img_dir) if "jpg" in im]
    print('dataset_length:{:}'.format(len(val_list)))

    # save results dir & zip
    if not os.path.exists(args.output_zip):
        os.mkdir(args.output_zip)
    eval_dir = "./eval_dir/"
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    if not os.path.exists(eval_dir + 'Baseline_predict_image'):
        os.mkdir(eval_dir + 'Baseline_predict_image')
    result_zip = zipfile.ZipFile(eval_dir + args.output_zip, 'w')

    #adding OCR module will update SyncBatchNorm --> BatchNorm
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    net = RFN(1, input_size=args.input_size,bn_type="test_bn", cfg=cfg, encode=encoder)
    net = net.cuda()

    # load checkpoint
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    # test for each image or batchsize>1
    for index in tqdm(range(math.ceil(len(val_list)//args.batchsize))):
        print("infer : %d / %d" % (index, math.ceil(len(val_list)//args.batchsize)), end='\r')
        input,Height,Width,img_name=get_data(img_dir, val_list, index, args.batchsize, args.input_size, Augmentation_inference_method)
        x=Variable(input)
        x = x.cuda()
        batch_loc_preds, batch_cls_preds, batch_gts_preds, _, batch_recur_proposals = net((x, None))
        for single_index in range(args.batchsize):
            ###reserve gts[1] result
            loc_preds, cls_preds, gts_preds, recur_proposals = batch_loc_preds[single_index], batch_cls_preds[single_index], batch_gts_preds[1][single_index], batch_recur_proposals[single_index]
            width=Width[single_index]
            height=Height[single_index]
            _img=img_name[single_index]
            save_file = "res_%s.txt" % (_img[5:-4])
            f = open(args.output_zip + "/res_%s.txt" % (_img[5:-4]), "w")
            refine_quad_bboxes, refine_quad_scores = encoder.refine_score(recur_proposals, args.cls_thresh,args.nms_thresh,gts_preds, cfg.MODEL.RRPN.GT_BOX_MARGIN,args.input_size,args.lamda)
            quad_boxes = refine_quad_bboxes
            if quad_boxes.shape[0] is 0:
                f.close()
                result_zip.write(args.output_zip + "/" + save_file, save_file, compress_type=zipfile.ZIP_DEFLATED)
                os.remove(args.output_zip + "/res_%s.txt" % (_img[5:-4]))
                continue
            quad_boxes /= args.input_size
            quad_boxes *= ([[width, height]] * 4)
            quad_boxes = quad_boxes.astype(np.int32)
            quad_boxes = check_and_validate_polys(quad_boxes)
            _quad = []
            for quad in quad_boxes:
                if args.dataset in ["MPSC",'MSRA-TD500','ICDAR2017MLT']:
                    [x0, y0], [x1, y1], [x2, y2], [x3, y3] = quad
                    f.write("%d,%d,%d,%d,%d,%d,%d,%d\n" % (x0, y0, x1, y1, x2, y2, x3, y3))
                elif args.dataset in ["ICDAR2013"]:
                    xmin = np.min(quad[:, 0])
                    ymin = np.min(quad[:, 1])
                    xmax = np.max(quad[:, 0])
                    ymax = np.max(quad[:, 1])
                    f.write("%d,%d,%d,%d\n" % (xmin, ymin, xmax, ymax))
            f.close()
            result_zip.write(args.output_zip + "/" + save_file, save_file, compress_type=zipfile.ZIP_DEFLATED)
            os.remove(args.output_zip + "/res_%s.txt" % (_img[5:-4]))

            if args.save_img:
                img = cv2.imread(img_dir + _img)
                # draw predict points
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = np.array(img, dtype=np.uint8)
                # draw gt points
                gt_anno = open(path+"/%s/test/gt_%s.txt" % (args.dataset,_img[5:-4]), "r")
                gt_anno = gt_anno.readlines()
                """
                you need modify some codes for different dataset!
                """
                if args.dataset in ["MPSC",'MSRA-TD500','ICDAR2017MLT','USTB-SV1K']:
                    for label in gt_anno:
                        _x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3,txt = label.split(",")[:10]
                        color = (0, 255, 0)
                        if "###" in txt:
                            color = (0, 255, 255)
                        _x0 = int(_x0)
                        gt_point = np.array([_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3], dtype=np.int32)
                        gt_point = gt_point.reshape(-1, 4, 2)
                        img = cv2.polylines(img, [gt_point], True, color, 4)
                    quad_boxes =quad_boxes.reshape(-1, 4, 2)
                    img = cv2.polylines(img, quad_boxes, True, (0,0,255), 4)
                elif args.dataset in ["ICDAR2013"]:
                    for label in gt_anno:
                        _xmin, _ymin, _xmax, _ymax = label.split(",")[:4]
                        img = cv2.rectangle(img, (int(_xmin), int(_ymin)), (int(_xmax), int(_ymax)), (0, 255, 0), 4)
                    for quad in quad_boxes:
                        xmin = np.min(quad[:, 0])
                        ymin = np.min(quad[:, 1])
                        xmax = np.max(quad[:, 0])
                        ymax = np.max(quad[:, 1])
                        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 4)
                save_img_dir = "./eval_dir/Baseline_predict_image/%s"
                cv2.imwrite(save_img_dir % (_img), img)
                if args.show_mask:
                    save_img_dir = "./eval_dir/Baseline_predict_image/%s_mask.jpg"
                    x3 = gts_preds[1][0, 1, :, :].sigmoid()
                    x3 = x3.cpu().detach().numpy()
                    x4 = x3 / x3.max() * 255
                    cv2.imwrite(save_img_dir % (_img[5:-4]), x4)
        del input,x,batch_loc_preds, batch_cls_preds, batch_gts_preds, _, batch_recur_proposals,loc_preds, cls_preds, gts_preds, recur_proposals
    result_zip.close()
    del net
    torch.cuda.empty_cache()

if args.eval:
    sys.path.append(os.getcwd() + '/eval/')
    import subprocess
    if args.dataset=='MPSC':
        gt_path="./data/MPSC/test.zip"
    elif args.dataset=='ICDAR2017-MLT':
        gt_path = "./data/ICDAR2017MLT/ICDAR2017MLT_validation_GT1.zip"
    elif args.dataset=='ICDAR2013':
        gt_path = "./data/ICDAR2013/Challenge2_Test_Task1_GT.zip"
    elif args.dataset=='ICDAR2013':
        gt_path = "./data/MSRA-TD500/gt.zip"
    eval_dir = './eval_dir/'
    print('_'*100)
    print('start evaluation!')
    if args.dataset in ['MPSC','INDAR2017-MLT','ICDAR2013']:
        query0 = "python ./eval/%s/script.py -g=%s -s=%s" % (args.dataset, gt_path, eval_dir + args.output_zip)
        scorestring0 = subprocess.check_output(query0, shell=True)
        hmean0 = float(str(scorestring0).strip().split(":")[3].split(",")[0].split("}")[0].strip())
        precise0 = float(str(scorestring0).strip().split(":")[1].split(",")[0].split("}")[0].strip())
        recall0 = float(str(scorestring0).strip().split(":")[2].split(",")[0].split("}")[0].strip())
        print("test_hmean: {:.4f}  precise:{:.4f}   recall:{:.4f}".format(hmean0,precise0, recall0))
    elif args.dataset in ["MSRA-TD500"]:
        from eval.MSRA import script
        pred_root=eval_dir + args.output_zip
        script.get_msra_result(gt_path, pred_root)
    else:
        from MSRA.eval_MSRA import get_sv1k_result
        gt_root = "./data/USTB-SV1K/SV1K_GT/"
        hmean_max = 0
        for i in range(5):
            pred_root = eval_dir + output_path[:-4] + '{:}'.format(i)
            scorestring, hmean = get_sv1k_result(gt_root, pred_root)
            print(scorestring)
            if hmean > hmean_max:
                max_scorestring = scorestring
                hmean_max = hmean
        print(max_scorestring)
