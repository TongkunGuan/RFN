import os
import sys
import glob
import math
import shutil
import numpy as np
import os.path as osp
from tqdm import tqdm
import file_util
import cv2

def rotate(theta, x, y):
    rotatex = math.cos(theta) * x - math.sin(theta) * y
    rotatey = math.cos(theta) * y + math.sin(theta) * x
    return rotatex, rotatey


def xy_rorate(theta, x, y, cx, cy):
    r_x, r_y = rotate(theta, x - cx, y - cy)
    return cx + r_x, cy + r_y


def rec_rotate(x, y, w, h, t):
    cx = x + w / 2
    cy = y + h / 2
    x1, y1 = xy_rorate(t, x, y, cx, cy)
    x2, y2 = xy_rorate(t, x + w, y, cx, cy)
    x3, y3 = xy_rorate(t, x, y + h, cx, cy)
    x4, y4 = xy_rorate(t, x + w, y + h, cx, cy)
    return x1, y1, x3, y3, x4, y4, x2, y2


def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j * 2] = corners[idx * 2]
            sorted[i, j * 2 + 1] = corners[idx * 2 + 1]
    return sorted

def get_gt(path):
    lines = file_util.read_file(path).split('\n')
    bboxes = []
    tags = []
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ' ')
        gt = line.split(' ')

        w_ = np.float(gt[4])
        h_ = np.float(gt[5])
        x1 = np.float(gt[2]) + w_ / 2.0
        y1 = np.float(gt[3]) + h_ / 2.0
        theta = np.float(gt[6]) / math.pi * 180

        bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
        bbox = bbox.reshape(-1)

        bboxes.append(bbox)
        tags.append(np.int(gt[1]))
    return np.array(bboxes), tags
def convert_label(gt_path, dst_path):
    lines = file_util.read_file(gt_path).split('\n')
    savestr = ''
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ' ')
        gt = line.split(' ')

        w_ = np.float(gt[4])
        h_ = np.float(gt[5])
        x1 = np.float(gt[2]) + w_ / 2.0
        y1 = np.float(gt[3]) + h_ / 2.0
        theta = np.float(gt[6]) / math.pi * 180
        diffcult=int(gt[1])
        txt=gt[7]
        if diffcult==1:
            print(gt_path)
        bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
        quads = bbox.reshape(-1)
        char_quads = ','.join([str(round(x)) for x in np.array(quads)[np.newaxis, :].squeeze()])
        text = ',text\n' if diffcult == 0 else ',###\n'
        char_quads += text
        savestr += char_quads
    savef = open(dst_path, 'w')
    savef.write(savestr)
    savef.close()
    #
    # f = open(gt_path, 'r')
    # savestr = ''
    # for line in f:
    #     _, diffcult, lx, ly, w, h, theta = [eval(x) for x in line.strip().split(' ')]
    #     quads = rec_rotate(lx, ly, w, h, theta)
    #     char_quads = ','.join([str(round(x)) for x in sort_corners(np.array(quads)[np.newaxis, :]).squeeze()])
    #     text = ',text\n' if diffcult == 0 else ',###\n'
    #     char_quads += text
    #     savestr += char_quads
    # savef = open(dst_path, 'w')
    # savef.write(savestr)
    # savef.close()


def convert_gt_to_txt(src_dir, eval_dir):
    label_paths = glob.glob(os.path.join(src_dir, '*.gt'))
    #print(label_paths)
    label_txt_path = osp.join(eval_dir, 'train')
    if os.path.exists(label_txt_path):
        shutil.rmtree(label_txt_path)
    os.mkdir(label_txt_path)
    
    pbar = tqdm(label_paths)
    all_ignore_num=0
    gt_num=0
    for label in pbar:
        f=open(label,'r').readlines()
        for line in f:
            if line.split(' ')[1] =="1":
                all_ignore_num+=1
            else:
                gt_num+=1
        im_name = os.path.split(label)[1].strip('.gt')
        pbar.set_description("gt.zip of MSRA_TD500 is generated in {}".format(eval_dir))
        label_txt = osp.join(label_txt_path, 'gt_img_' + im_name + '.txt')
        convert_label(label, label_txt)
    gt_zip = os.path.join(eval_dir, 'gt.zip')
    os.system('zip -j {} {}'.format(gt_zip, label_txt_path + '/*'))
    #shutil.rmtree(label_txt_path)
    print((all_ignore_num,gt_num))
def get_text_area_ratios():
    gt_path = "E:/USTB-SV1K_V1/USTB-SV1K_V1/train/"
    img_path= "E:/USTB-SV1K_V1/USTB-SV1K_V1/training/"
    area_ratio=[]
    for path in os.listdir(img_path):
        try:
            img=cv2.imread(img_path+path)
            H,W = img.shape[0],img.shape[1]
            area=H*W
            f=open(gt_path + "gt_img_%s.txt" % (path[:-4]), "r",encoding='utf-8')
            gt_anno = f.readlines()
            for label in gt_anno:
                txt = label.split(',')[-1]
                if '###' in txt:
                    continue
                else:
                    x0, y0, x1, y1, x2, y2, x3, y3=map(float,label.split(',')[:8])
                    _quad=np.array([x0, y0, x1, y1, x2, y2, x3, y3]).reshape(-4,2)
                    area_ratio.append(plg.Polygon(_quad).area()/area)
            f.close()
        except:
            print(path)
    
    sort_size=[]
    for i in range(16,768,16):
        sort_size.append((i/768)**2)
    sort_size=np.array(sort_size)
    res=[0 for i in range(len(sort_size))]
    for x in area_ratio:
        k=np.argmin(np.abs(x-sort_size))
        res[k] += 1
    ratio_dict=dict(zip(np.sqrt(sort_size)*768,res))
    sorted(ratio_dict.items(), key=lambda d:d[1],reverse = True)
    return ratio_dict
def polygons_to_rboxes(valid_polygons):
    #a=tf.constant([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])\n",
    #n_polygons=valid_polygons.get_shape().as_list()[0]\n",
    center_X=(valid_polygons[:,0]+valid_polygons[:,2]+valid_polygons[:,4]+valid_polygons[:,6])/4##dtype=tf.float64\n",
    center_Y=(valid_polygons[:,1]+valid_polygons[:,3]+valid_polygons[:,5]+valid_polygons[:,7])/4
    #dist_tltr = point_distance(x_tl, y_tl, x_tr, y_tr)\n",
    dist_tltr = np.sqrt(np.square(valid_polygons[:,2]-valid_polygons[:,0])+np.square(valid_polygons[:,3]-valid_polygons[:,1]))
    dist_blbr = np.sqrt(np.square(valid_polygons[:,6]-valid_polygons[:,4])+np.square(valid_polygons[:,7]-valid_polygons[:,5]))
    width= (dist_tltr + dist_blbr) / 2
    eps = 1e-6
    dx_top=valid_polygons[:,2]-valid_polygons[:,0]
    dy_top=valid_polygons[:,3]-valid_polygons[:,1]

    dx_bot=valid_polygons[:,4]-valid_polygons[:,6]
    dy_bot=valid_polygons[:,5]-valid_polygons[:,7]
    divider_top = dist_tltr + eps
    divider_bot = dist_blbr + eps
    dist_top = np.abs(center_X * dy_top - center_Y * dx_top + valid_polygons[:,2] * valid_polygons[:,1] - valid_polygons[:,3] * valid_polygons[:,0]) / divider_top
    dist_bot = np.abs(center_X * dy_bot - center_Y * dx_bot + valid_polygons[:,4] * valid_polygons[:,7] - valid_polygons[:,5] * valid_polygons[:,6]) / divider_bot
    helight=dist_top+dist_bot
    theta1 = np.arctan2(valid_polygons[:,3] - valid_polygons[:,1], valid_polygons[:,2] - valid_polygons[:,0])
    theta2 = np.arctan2(valid_polygons[:,5] - valid_polygons[:,7], valid_polygons[:,4] - valid_polygons[:,6])
    theta = ((theta1 + theta2) / 2.)
    rbboxs=np.stack([center_X,center_Y,width,helight,theta],1)
    return rbboxs
def compute_ratio_txt():
    path='E:/USTB-SV1K_V1/USTB-SV1K_V1/train/*.txt'
    all_not_care=0
    all_care=0
    all_polygon=[]
    for fname in glob.glob(path):
        f=open(fname,'r',encoding='utf-8')
        for line in f.readlines():
            line=line.strip()
            line=line.split(',')
            polygon=list(map(float,line[:8]))
            text=line[-1]
            if text =="###":
                all_not_care+=1
            else:
                all_care+=1
            all_polygon.append(polygon)
    all_polygon=np.array(all_polygon)
    print(all_polygon.shape)
    rboxes=polygons_to_rboxes(all_polygon)
    w,h=rboxes[:,2],rboxes[:,3]
    ratios=w/h
    return ratios
def compute_ratio_gt():
    path='E:/USTB-SV1K_V1/USTB-SV1K_V1/testing/*.gt'
    all_not_care=0
    all_care=0
    ratios=[]
    for fname in glob.glob(path):
        f=open(fname,'r',encoding='utf-8')
        for line in f.readlines():
            line=line.strip()
            line=line.split(' ')
            w=float(line[4])
            h=float(line[5])+1e-6
            text=line[-1]
            if text =="###":
                all_not_care+=1
            else:
                all_care+=1
            ratios.append(w/h)
    ratios=np.array(ratios)
    sort_size=[]
    for i in range(10,0,-1):
        sort_size.append(1/i)
    for i in range(2,20):
        sort_size.append(i)
    sort_size=np.array(sort_size)
    res=[0 for i in range(len(sort_size))]
    for x in ratios:
        k=np.argmin(np.abs(x-sort_size))
        res[k] += 1
    #for i in range(len(res)):
    #    print((sort_size[i],res[i]))
    ratio_dict=dict(zip(sort_size,res))
    sorted(ratio_dict.items(), key=lambda d:d[1],reverse = True)
    return ratios
def compute_IC17_13():
    gt_path = "E:/ICDAR2017MLT//ICDAR2017MLT_validation_GT/"
    img_path = "E:/ICDAR2017MLT//ICDAR2017MLT_validation/"
    area_ratio = []
    for path in os.listdir(img_path):
        try:
            img = cv2.imread(img_path + path)
            H, W = img.shape[0], img.shape[1]
            area = H * W
            f = open(gt_path + "gt_%s.txt" % (path[:-4]), "r", encoding='utf-8')
            gt_anno = f.readlines()
            for label in gt_anno:
                txt = label.split(',')[-1]
                if '###' in txt:
                    continue
                else:
                    x0, y0, x1, y1, x2, y2, x3, y3 = map(float, label.split(',')[:8])
                    _quad = np.array([x0, y0, x1, y1, x2, y2, x3, y3]).reshape(-4, 2)
                    area_ratio.append(plg.Polygon(_quad).area() / area)
            f.close()
        except:
            print(path)
    return area_ratio

if __name__ == '__main__':
    ###MSRA-TD500 or USTB-SV1K
    root_dir = 'E:/USTB-SV1K_V1/USTB-SV1K_V1/'
    convert_gt_to_txt(src_dir=os.path.join(root_dir, 'training/'),eval_dir=os.path.join(root_dir))
    ratios=compute_ratio_gt() #aspect_ratios
    area_ratio=get_text_area_ratios() #anchor_areas

    ###IC17-MLT IC13 MPSC
    ratios=compute_IC17_13()
"""
We count some value about scene text dataset to adapt the parameters: aspect_ratios and anchor_areas
"""
#MSRA-TD500:aspect_ratios
"""
trainset:
(0.037037037037037035, 0)(0.038461538461538464, 1)(0.04, 0)(0.041666666666666664, 0)(0.043478260869565216, 0)(0.045454545454545456, 1)(0.047619047619047616, 0)
(0.05, 0)(0.05263157894736842, 0)(0.05555555555555555, 0)(0.058823529411764705, 0)(0.0625, 0)(0.06666666666666667, 0)(0.07142857142857142, 2)
(0.07692307692307693, 4)(0.08333333333333333, 1)(0.09090909090909091, 1)(0.1, 2)(0.1111111111111111, 1)(0.125, 2)(0.14285714285714285, 2)
(0.16666666666666666, 4)(0.2, 11)(0.25, 15)(0.3333333333333333, 9)(0.5, 36)
(1.0, 56)(2.0, 89)(3.0, 166)(4.0, 139)(5.0, 128)(6.0, 100)(7.0, 51)(8.0, 59)(9.0, 34)(10.0, 24)(11.0, 26)(12.0, 18)(13.0, 21)(14.0, 12)
(15.0, 7)(16.0, 5)(17.0, 4)(18.0, 6)(19.0, 6)(20.0, 2)(21.0, 3)(22.0, 1)(23.0, 3)(24.0, 2)(25.0, 1)(26.0, 0)(27.0, 0)(28.0, 0)(29.0, 1)
(30.0, 0)(31.0, 0)(32.0, 0)(33.0, 1)(34.0, 2)(35.0, 0)(36.0, 2)(37.0, 2)(38.0, 2)(39.0, 0)(40.0, 0)(41.0, 0)(42.0, 0)(43.0, 0)(44.0, 0)\
(45.0, 1)(46.0, 0)(47.0, 1)(48.0, 1)
testset:
(0.0625, 1)(0.06666666666666667, 1)(0.07142857142857142, 2)(0.07692307692307693, 2)(0.08333333333333333, 0)(0.09090909090909091, 1)
(0.1, 0)(0.1111111111111111, 0)(0.125, 3)(0.14285714285714285, 2)(0.16666666666666666, 2)(0.2, 4)(0.25, 4)(0.3333333333333333, 12)(0.5, 15)
(1.0, 28)(2.0, 72)(3.0, 88)(4.0, 100)(5.0, 72)(6.0, 59)(7.0, 46)(8.0, 38)(9.0, 31)(10.0, 23)(11.0, 14)(12.0, 3)(13.0, 7)(14.0, 5)(15.0, 1)
(16.0, 4)(17.0, 2)(18.0, 2)(19.0, 0)(20.0, 2)(21.0, 1)(22.0, 0)(23.0, 2)(24.0, 0)(25.0, 0)(26.0, 1)
"""
###ICDAR2017-MLT anchor_areas
"""
train:
[(8.0, 12660),(16.0, 9341),(24.0, 7879),(32.0, 6135),(40.0, 4924), (48.0, 3948), (56.0, 3181), (64.0, 2600),
 (72.0, 2160), (80.0, 1886), (88.0, 1639), (96.0, 1483), (104.0, 1240), (112.0, 1122), (120.0, 961), (128.0, 807),
 (136.0, 740), (144.0, 627), (152.0, 576), (160.0, 468), (168.0, 448), (176.0, 397), (184.0, 359), (192.0, 315),
 (200.0, 273), (208.0, 272), (224.0, 234), (216.0, 210), (232.0, 193), (248.0, 164), (240.0, 153), (264.0, 133),

 (272.0, 125), (256.0, 116), (280.0, 108), (288.0, 85), (296.0, 77), (320.0, 76), (304.0, 72), (312.0, 67),
 (336.0, 51), (352.0, 49), (328.0, 47), (344.0, 41), (368.0, 40), (360.0, 39), (400.0, 30), (408.0, 28), (384.0, 27),
 (376.0, 25), (392.0, 21), (432.0, 15), (448.0, 15), (416.0, 14), (456.0, 11), (496.0, 11), (464.0, 10), (424.0, 9), (472.0, 9),
 (440.0, 8), (480.0, 5), (488.0, 5), (504.0, 4), (512.0, 3), (520.0, 3), (528.0, 3), (560.0, 3), (536.0, 2), (544.0, 2),
 (568.0, 2), (576.0, 2), (656.0, 2), (552.0, 1), (600.0, 1), (608.0, 1), (616.0, 1), (632.0, 1), (640.0, 1),
 (664.0, 1), (672.0, 1), (696.0, 1), (720.0, 1)
test:
[(8.0, 3215),(16.0, 2207), (24.0, 1787), (32.0, 1468), (40.0, 1090), (48.0, 848), (56.0, 743), (64.0, 563),
 (72.0, 521), (80.0, 470), (96.0, 370), (88.0, 369), (112.0, 298), (104.0, 271), (120.0, 219), (128.0, 208),
 (136.0, 191), (144.0, 146), (152.0, 133), (160.0, 131), (168.0, 110), (176.0, 97), (184.0, 85), (192.0, 78),
 (208.0, 69), (200.0, 60), (216.0, 55), (224.0, 49), (256.0, 36), (232.0, 35), (248.0, 35), (240.0, 33),
 (264.0, 33), (280.0, 30), (272.0, 26), (288.0, 25), (304.0, 19), (296.0, 18), (320.0, 15), (328.0, 14),
 (312.0, 13), (384.0, 13), (336.0, 12), (368.0, 12), (344.0, 11), (392.0, 8), (360.0, 7), (352.0, 6),
 (376.0, 6), (432.0, 5), (440.0, 5), (464.0, 5), (400.0, 4), (408.0, 4), (424.0, 4), (496.0, 3), (416.0, 2),
 (472.0, 2), (448.0, 1), (456.0, 1), (480.0, 1), (488.0, 1), (504.0, 1), (528.0, 1),(536.0, 1), (592.0, 1), (656.0, 1)
"""