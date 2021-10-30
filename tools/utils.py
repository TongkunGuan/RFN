'''Some helper functions for PyTorch.'''
import os
import sys
import time
import math
import imageio
import cv2
import torch
import torch.nn as nn
import numpy as np

def get_mean_and_std(dataset, max_load=10000):
    '''Compute the mean and std value of dataset.'''
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i in range(N):
        print(i)
        im,_,_ = dataset.load(1)
        for j in range(3):
            mean[j] += im[:,j,:,:].mean()
            std[j] += im[:,j,:,:].std()
    mean.div_(N)
    std.div_(N)
    return mean, std

def mask_select(input, mask, dim=0):
    '''Select tensor rows/cols using a mask tensor.

    Args:
      input: (tensor) input tensor, sized [N,M].
      mask: (tensor) mask tensor, sized [N,] or [M,].
      dim: (tensor) mask dim.

    Returns:
      (tensor) selected rows/cols.

    Example:
    >>> a = torch.randn(4,2)
    >>> a
    -0.3462 -0.6930
     0.4560 -0.7459
    -0.1289 -0.9955
     1.7454  1.9787
    [torch.FloatTensor of size 4x2]
    >>> i = a[:,0] > 0
    >>> i
    0
    1
    0
    1
    [torch.ByteTensor of size 4]
    >>> masked_select(a, i, 0)
    0.4560 -0.7459
    1.7454  1.9787
    [torch.FloatTensor of size 2x2]
    '''
    index = mask.nonzero().squeeze(1)
    return input.index_select(dim, index)

def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x)  #v3
    b = torch.arange(0,y)        #v3

    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1).float() if row_major else torch.cat([yy,xx],1).float()

def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4] or [N, 8] .
      order: (str) one of ['xyxy2xywh','xywh2xyxy', 'xywh2quad', 'quad2xyxy']

    Returns:
      (tensor) converted bounding boxes, sized [N,4] or [N,8].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy', 'xywh2quad', 'quad2xyxy','quad2xywh',"quad2xyxy_8"]
    
    if order is 'xyxy2xywh':
        a = boxes[:,:2]
        b = boxes[:,2:]
        new_boxes = torch.cat([(a+b)/2,b-a+1], 1)
        
    elif order is 'xywh2xyxy':
        a = boxes[:,:2]
        b = boxes[:,2:]
        new_boxes = torch.cat([a-b/2,a+b/2], 1)
        
    elif order is 'xywh2quad':
        x0, y0, w0, h0 = torch.split(boxes, 1, dim=1)
        
        new_boxes = torch.cat([x0-w0/2, y0-h0/2,  
                               x0+w0/2, y0-h0/2, 
                               x0+w0/2, y0+h0/2, 
                               x0-w0/2, y0+h0/2], dim=1)
        
    elif order is "quad2xyxy":
        """quad : [num_boxes, 8] / rect : [num_boxes, 4] #yxyx"""
        boxes = boxes.view((-1, 4, 2))
        
        new_boxes = torch.cat([torch.min(boxes[:, :, 0:1], dim=1)[0],
                               torch.min(boxes[:, :, 1:2], dim=1)[0],
                               torch.max(boxes[:, :, 0:1], dim=1)[0],
                               torch.max(boxes[:, :, 1:2], dim=1)[0]], dim=1)
    elif order is "quad2xywh":
        boxes = boxes.view((-1, 4, 2))
        
        new_boxes = torch.cat([torch.min(boxes[:, :, 0:1], dim=1)[0],
                               torch.min(boxes[:, :, 1:2], dim=1)[0],
                               torch.max(boxes[:, :, 0:1], dim=1)[0],
                               torch.max(boxes[:, :, 1:2], dim=1)[0]], dim=1)
        a = new_boxes[:,:2]
        b = new_boxes[:,2:]
        new_boxes = torch.cat([(a+b)/2,b-a+1], 1)
    elif order is "quad2xyxy_8":
        boxes = boxes.view((-1, 4, 2))

        new_boxes = torch.cat([torch.min(boxes[:, :, 0:1], dim=1)[0],
                               torch.min(boxes[:, :, 1:2], dim=1)[0],
                               torch.max(boxes[:, :, 0:1], dim=1)[0],
                               torch.max(boxes[:, :, 1:2], dim=1)[0]], dim=1)
        # x_min,y_min,x_max,y_max
        x_min, y_min, x_max, y_max = torch.split(new_boxes, 1, dim=1)
        new_boxes = torch.cat([x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max], 1)
    return new_boxes
def box_iou_xyxy(box1,box2):
    box1=change_box_order(torch.Tensor(box1), 'quad2xyxy')
    box2=change_box_order(torch.Tensor(box2), 'quad2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt + 1).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)  # [N,]
    area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)

    return iou
def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) anchor_rect_boxes, sized [N,4]. xywh
      box2: (tensor) gt_rect_boxes, sized [M,4]. xyxy

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    box1 = change_box_order(box1, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)

    return iou

def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def softmax(x):
    '''Softmax along a specific dimension.

    Args:
      x: (tensor) input tensor, sized [N,D].

    Returns:
      (tensor) softmaxed tensor, sized [N,D].
    '''
    xmax, _ = x.max(1)
    x_shift = x - xmax.view(-1,1)
    x_exp = x_shift.exp()
    return x_exp / x_exp.sum(1).view(-1,1)

def one_hot_v3(batch, depth):
    emb = nn.Embedding(depth, depth)
    emb.weight.data = torch.eye(depth)
    return emb(batch)

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]

def msr_init(net):
    '''Initialize layer parameters.'''
    for layer in net:
        if type(layer) == nn.Conv2d:
            n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2./n))
            layer.bias.data.zero_()
        elif type(layer) == nn.BatchNorm2d:
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif type(layer) == nn.Linear:
            layer.bias.data.zero_()

def read_image_size(file):
    im = cv2.imread(file)
    if im is None:
        gif = imageio.mimread(file)
        if gif is not None:
            return gif.shape
        else:
            return None
    else:
        return im.shape

def generate_global_input_images_mask(boxes,img_shape):
    # Judge if str is hard samples
    if np.array(boxes).shape[1] >= 8:
        polys = np.array(boxes).reshape((-1,4,2)).astype(np.int32)
        # polys = map(int, polys)
        text_masks = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        # text_masks = cv2.fillPoly(text_masks,[poly for poly in polys], 1)
        for poly in polys:
            # generate text mask for per text region poly
            text_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
            text_masks =cv2.fillPoly(text_masks, [poly], 1)
            text_masks+=text_mask
        return text_masks
def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    poly_=np.array(poly)
    assert poly_.shape == (4,2), 'poly shape should be 4,2'
    edge=[
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
        ]
    return np.sum(edge)/2.

def calculate_distance(c1, c2):
    return math.sqrt(math.pow(c1[0]-c2[0], 2) + math.pow(c1[1]-c2[1], 2))

def choose_best_begin_point(pre_result):
    """
    find top-left vertice and resort
    """
    final_result = []
    for coordinate in pre_result:
        x1 = coordinate[0][0]
        y1 = coordinate[0][1]
        x2 = coordinate[1][0]
        y2 = coordinate[1][1]
        x3 = coordinate[2][0]
        y3 = coordinate[2][1]
        x4 = coordinate[3][0]
        y4 = coordinate[3][1]
        xmin = min(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        xmax = max(x1, x2, x3, x4)
        ymax = max(y1, y2, y3, y4)
        combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                     [[x2, y2], [x3, y3], [x4, y4], [x1, y1]], 
                     [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], 
                     [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
        dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        force = 100000000.0
        force_flag = 0
        for i in range(4):
            temp_force = calculate_distance(combinate[i][0], dst_coordinate[0]) + calculate_distance(combinate[i][1], dst_coordinate[1]) + calculate_distance(combinate[i][2], dst_coordinate[2]) + calculate_distance(combinate[i][3], dst_coordinate[3])
            if temp_force < force:
                force = temp_force
                force_flag = i
        #if force_flag != 0:
        #    print("choose one direction!")
        final_result.append(combinate[force_flag])
        
    return np.array(final_result)
def check_and_validate_polys(polys):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    '''

    if polys.shape[0] == 0:
        return polys
    
    validated_polys = []

    # find top-left and clockwise
    polys = choose_best_begin_point(polys)

    for poly in polys:
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            #print('invalid poly')
            continue
        if p_area > 0:
            #print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
    return np.array(validated_polys)

def convert_polyons_into_angle_upgrade(boxes):
    boxes = boxes.detach().cpu().numpy().astype(np.float)
    length = len(boxes)
    angle = np.array([0 for i in range(length)]).astype(np.float)
    width = np.array([0 for i in range(length)]).astype(np.float)
    height = np.array([0 for i in range(length)]).astype(np.float)
    pt1 = boxes[:, 0:2]
    pt2 = boxes[:, 2:4]
    pt3 = boxes[:, 4:6]
    pt4 = boxes[:, 6:8]
    edge1 = np.sqrt((pt1[:, 0] - pt2[:, 0]) * (pt1[:, 0] - pt2[:, 0]) + (pt1[:, 1] - pt2[:, 1]) * (pt1[:, 1] - pt2[:, 1]))
    edge2 = np.sqrt((pt2[:, 0] - pt3[:, 0]) * (pt2[:, 0] - pt3[:, 0]) + (pt2[:, 1] - pt3[:, 1]) * (pt2[:, 1] - pt3[:, 1]))
    x_ctr = (pt1[:, 0] + pt3[:, 0]) / 2
    y_ctr = (pt1[:, 1] + pt3[:, 1]) / 2
    #############################################################edge1>edge2
    edge1_large_edge2 = np.argwhere((edge1 > edge2) == 1).reshape(-1)
    width[edge1_large_edge2] = edge1[edge1_large_edge2]
    height[edge1_large_edge2] = edge2[edge1_large_edge2]

    judge = pt1[edge1_large_edge2, 0] - pt2[edge1_large_edge2, 0]
    ######################################pt1[0]-pt2[0]!=0
    edge1_large_edge2_angle = edge1_large_edge2[np.argwhere(judge != 0).reshape(-1)]
    angle[edge1_large_edge2_angle] = -np.arctan(
        (pt1[edge1_large_edge2_angle, 1] - pt2[edge1_large_edge2_angle, 1]).astype(np.float)
        / (pt1[edge1_large_edge2_angle, 0] - pt2[edge1_large_edge2_angle, 0]).astype(np.float)) / 3.1415926 * 180
    ######################################pt1[0]-pt2[0]==0
    angle[edge1_large_edge2[np.argwhere(judge == 0).reshape(-1)]] = 90.0

    #############################################################edge2>=edge1
    edge2_large_edge1 = np.argwhere((edge2 >= edge1) == 1).reshape(-1)
    width[edge2_large_edge1] = edge2[edge2_large_edge1]
    height[edge2_large_edge1] = edge1[edge2_large_edge1]

    judge = pt2[edge2_large_edge1, 0] - pt3[edge2_large_edge1, 0]
    ######################################pt2[0]-pt3[0]!=0
    edge2_large_edge1_angle = edge2_large_edge1[np.argwhere(judge != 0).reshape(-1)]
    angle[edge2_large_edge1_angle] = -np.arctan(
        (pt2[edge2_large_edge1_angle, 1] - pt3[edge2_large_edge1_angle, 1]).astype(np.float)
        / (pt2[edge2_large_edge1_angle, 0] - pt3[edge2_large_edge1_angle, 0]).astype(np.float)) / 3.1415926 * 180
    ######################################pt2[0]-pt3[0]==0
    angle[edge2_large_edge1[np.argwhere(judge == 0).reshape(-1)]] = 90.0

    #############################################################angle<-45.0
    angle[np.argwhere(angle < -45.0)] += 180
    polyon_angle = np.concatenate(
        [x_ctr.reshape(length, 1), y_ctr.reshape(length, 1), width.reshape(length, 1), height.reshape(length, 1),
         angle.reshape(length, 1)], 1)
    return polyon_angle
def convert_polyons_into_angle_cuda(boxes):
    # boxes=scale_boxes.cpu().numpy()
    polyon_angle = []
    for box in boxes:
        gt_ind = box
        # gt_ind = torch.Tensor(gt_ind)

        pt1 = (gt_ind[0], gt_ind[1])
        pt2 = (gt_ind[2], gt_ind[3])
        pt3 = (gt_ind[4], gt_ind[5])
        pt4 = (gt_ind[6], gt_ind[7])

        edge1 = torch.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        edge2 = torch.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

        angle = 0

        if edge1 > edge2:

            width = edge1
            height = edge2
            if pt1[0] - pt2[0] != 0:
                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
            else:
                angle = 90.0
        elif edge2 >= edge1:
            width = edge2
            height = edge1
            # print pt2[0], pt3[0]
            if pt2[0] - pt3[0] != 0:
                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
            else:
                angle = 90.0
        if angle < -45.0:
            angle = angle + 180

        x_ctr = (pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
        y_ctr = (pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

        polyon_angle.append([x_ctr, y_ctr, width, height, angle])
    return polyon_angle


def convert_polyons_into_angle(boxes):
    # boxes=scale_boxes.cpu().numpy()
    polyon_angle = []
    for box in boxes:
        gt_ind = box
        gt_ind = torch.Tensor(gt_ind)

        pt1 = (gt_ind[0], gt_ind[1])
        pt2 = (gt_ind[2], gt_ind[3])
        pt3 = (gt_ind[4], gt_ind[5])
        pt4 = (gt_ind[6], gt_ind[7])

        edge1 = torch.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        edge2 = torch.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

        angle = 0

        if edge1 > edge2:

            width = edge1
            height = edge2
            if pt1[0] - pt2[0] != 0:
                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
            else:
                angle = 90.0
        elif edge2 >= edge1:
            width = edge2
            height = edge1
            # print pt2[0], pt3[0]
            if pt2[0] - pt3[0] != 0:
                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
            else:
                angle = 90.0
        if angle < -45.0:
            angle = angle + 180

        x_ctr = (pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
        y_ctr = (pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

        polyon_angle.append([x_ctr, y_ctr, width, height, angle])
    return polyon_angle


def select_top_predictions(predictions, confidence_threshold):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def convert_angle_into_polygons(boxes):
    rotated_pts_all = []
    for idx in range(len(boxes)):
        cx, cy, w, h, angle = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]
        # need a box score larger than thresh
        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)
        rotated_pts_all.append(rotated_pts[:, :2])
    return np.array(rotated_pts_all)
def convert_angle_into_polygons_and_refine_Diamond_box(boxes):
    rotated_pts_all = []
    for idx in range(len(boxes)):
        cx, cy, w, h, angle = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]
        #refine_Diamond_box
        if angle>30 and angle<60:
            if np.abs(w-h)<10:
                angle=0
        # need a box score larger than thresh

        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)
        rotated_pts_all.append(rotated_pts[:, :2])
    return np.array(rotated_pts_all)