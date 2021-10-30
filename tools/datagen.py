'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
import cv2
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop
import utils
from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.config import cfg
from utils import convert_polyons_into_angle
class ListDataset(data.Dataset):
    def __init__(self, root, dataset, train, transform, input_size, multi_scale=False,encoder=None):
        '''
        Args:
          root: (str) DB root ditectory.
          dataset: (str) Dataset name(dir).
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
          multi_scale: (bool) use multi-scale training or not.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.multi_scale = multi_scale
        self.MULTI_SCALES = [512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]
        self.encoder = encoder
        self.path="/home/amax/GTK/"


        if "SynthText" in dataset:
            self.get_SynthText()
        if "ICDAR2015" in dataset:
            self.get_ICDAR2015()
        if "MLT" in dataset:
            self.get_MLT()
        if "ICDAR2013" in dataset:
            self.get_ICDAR2013()
        if "MSC2020_build" in dataset:
            self.get_MSC2020_build()
        if "S-MSC2020" in dataset:
            self.get_S_MSC2020()
        if "USTB-SV1K" in dataset:
            self.get_SV1K()

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) dataset index.

        Returns:
          image: (tensor) image array.
          boxes: (tensor) boxes array.
          labels: (tensor) labels array.
        '''
        # Load image, boxes and labels.
        fname = self.fnames[idx]

        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = self.boxes[idx].copy()
        labels = self.labels[idx]

        return {"image" : img, "boxes" : boxes, "labels" : labels}

    def collate_fn(self, batch):
        '''bbox encode and make batch

        Args:
          batch: (dict list) images, boxes and labels

        Returns:
          batch_images, batch_loc, batch_cls
        '''
        size = self.input_size
        if self.multi_scale: # get random input_size for multi-scale traininig
            random_choice = random.randint(0, len(self.MULTI_SCALES)-1)
            size = self.MULTI_SCALES[random_choice]

        inputs = torch.zeros(len(batch), 3, size, size)
        loc_targets = []
        cls_targets = []
        gt_global_mask_all=torch.zeros(len(batch), size, size)
        target_polyons=[]
        for n, data in enumerate(batch):

            img, boxes, labels = self.transform(data['image'], data['boxes'], data['labels'])
            if (np.inf in boxes.reshape(-1)) or ((-1*np.inf) in boxes.reshape(-1)):
                print("target_inf")
                return inputs, torch.stack(loc_targets), torch.stack(cls_targets),None,None
            inputs[n] = img
            loc_target, cls_target = self.encoder.encode(boxes, labels, input_size=(size, size))
            #aaa=self.encoder.encode_save(img, boxes, labels, input_size=(size, size))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
            '''
            _masks=[]
            for box in boxes:
                text_mask=utils.generate_global_input_images_mask(boxes, (size,size))
                _masks.append(text_mask)
            gt_global_mask = np.sum(_masks, axis=-1).reshape((size, size)).astype(np.uint8)
            '''
            text_masks=utils.generate_global_input_images_mask(boxes, (size,size))
            gt_global_mask = 1-(text_masks<1).astype(np.int)
            gt_global_mask_all[n]=torch.from_numpy(gt_global_mask).float()
            boxes = convert_polyons_into_angle(boxes.reshape(-1, 8))
            boxes = torch.Tensor(boxes)
            boxes[:, 2:4] *= cfg.MODEL.RRPN.GT_BOX_MARGIN
            boxlist = RBoxList(boxes, (size, size))
            boxlist.add_field("difficult", 1 - labels)
            boxlist.add_field("labels", labels)
            target_polyons.append(boxlist)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets),gt_global_mask_all,target_polyons

    def __len__(self):
        return self.num_samples

    def get_SynthText(self):
        import scipy.io as sio
        data_dir = os.path.join(self.root, 'SynthText/')

        gt = sio.loadmat(data_dir + 'gt.mat')
        dataset_size = gt['imnames'].shape[1]
        img_files = gt['imnames'][0]
        labels = gt['wordBB'][0]
        #lines=gt['txt'][0]
        self.num_samples = dataset_size
        print("Training on SynthText : ", dataset_size)

        for i in range(dataset_size):
            img_file = data_dir + str(img_files[i][0])
            label = labels[i]


            _quad = []
            _classes = []

            if label.ndim == 3:
                for i in range(label.shape[2]):
                    _x0 = label[0][0][i]
                    _y0 = label[1][0][i]
                    _x1 = label[0][1][i]
                    _y1 = label[1][1][i]
                    _x2 = label[0][2][i]
                    _y2 = label[1][2][i]
                    _x3 = label[0][3][i]
                    _y3 = label[1][3][i]

                    _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                    _classes.append(1)

            else:
                _x0 = label[0][0]
                _y0 = label[1][0]
                _x1 = label[0][1]
                _y1 = label[1][1]
                _x2 = label[0][2]
                _y2 = label[1][2]
                _x3 = label[0][3]
                _y3 = label[1][3]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
    def get_S_MSC2020(self):
        #root='/home/amax/GTK/SynthText-python3/results/gen_pictures/'
        data_dir = self.root
        train_list = os.listdir(data_dir)
        train_path=[]
        for path in train_list:
            train_path.append(os.listdir(os.path.join(data_dir,path)))
        train_path_all=[]
        for i in train_path:
            train_path_all.extend(i)
        dataset_list = [l[:-4] for l in train_path_all if "jpg" in l]
        file_list=[l.split('_')[1] for l in train_path_all if "jpg" in l]
        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on S-MSC2020_build: ", dataset_size)

        for i in range(len(dataset_list)):
            img_file = data_dir + "%s/%s.jpg" % (file_list[i], dataset_list[i])
            label_file = open(self.path+"SynthText-python3/results/gen_labels/%s/%s.txt" % (file_list[i],dataset_list[i]))
            label_file = label_file.readlines()

            _quad = []
            _classes = []

            for label in label_file:
                _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3, txt = label.replace('\n','').split(",")[:9]

                if "###" in txt:
                    continue

                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1,_x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1,_x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
    def get_MSC2020_build(self):
        # root='/home/amax/GTK/MSC2020_build/'
        data_dir = self.root
        dataset_list = os.listdir(data_dir)
        dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on MSC2020_build: ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "%s.jpg" % (i)
            label_file = open("../../../../../MSC_GroundTruth/%s/gt_%s.txt" % (mode, i[4:]))
            label_file = label_file.readlines()
            _quad = []
            _classes = []
            # print("{:}".format(i),end=" ")
            for label in label_file:
                _x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, txt = label.replace('\n', '').split(",")[:9]

                if "###" in txt:
                    continue

                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1, _x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1, _x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
    def get_ICDAR2015(self):
        data_dir = os.path.join(self.root, 'ICDAR2015_Incidental/')

        dataset_list = os.listdir(data_dir + "train")
        dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on ICDAR2015 : ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "%s/%s.jpg" % (mode, i)
            label_file = open(data_dir + "%s/gt_%s.txt" % (mode, i))
            label_file = label_file.readlines()

            _quad = []
            _classes = []

            for label in label_file:
                _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3, txt = label.split(",")[:9]

                if "###" in txt:
                    continue

                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1,_x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1,_x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))

    def get_MLT(self):
        data_dir = os.path.join(self.root, 'MLT/')

        dataset_list = os.listdir(data_dir + "train")
        dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on MLT : ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "%s/%s.jpg" % (mode, i)
            label_file = open(data_dir + "%s/gt_%s.txt" % (mode, i))
            label_file = label_file.readlines()

            _quad = []
            _classes = []

            for label in label_file:
                _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3, lang, txt = label.split(",")[:10]

                if "###" in txt:
                    continue

                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1,_x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1,_x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
            
    def get_ICDAR2013(self):
        data_dir = os.path.join(self.root, 'ICDAR2013_FOCUSED/')

        dataset_list = os.listdir(data_dir + "train")
        dataset_list = [l[:-4] for l in dataset_list if "jpg" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on ICDAR2013 : ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "%s/%s.jpg" % (mode, i)
            label_file = open(data_dir + "%s/gt_%s.txt" % (mode, i))
            label_file = label_file.readlines()

            _quad = []
            _classes = []

            for label in label_file:
                _xmin, _ymin, _xmax, _ymax = label.split(" ")[:4]

                _x0 = _xmin
                _y0 = _ymin
                _x1 = _xmax
                _y1 = _ymin
                _x2 = _xmax
                _y2 = _ymax
                _x3 = _xmin
                _y3 = _ymax

                _x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3 = [int(p) for p in [_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3]]

                _quad.append([_x0, _y0, _x1, _y1,_x2, _y2, _x3, _y3])
                _classes.append(1)

            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
    def get_MSRA(self):
        data_dir = self.root
        import glob
        dataset_list = glob.glob(self.root + "train/*.JPG")
        dataset_list = [l.split('/')[-1][:-4] for l in dataset_list if "JPG" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on MSRA-TD500 : ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "train/%s.JPG" % (i)
            label_file = open(data_dir + "MSRA_train_GT/%s.txt" % (i))
            label_file = label_file.readlines()

            _quad=[]
            _classes = []

            for label in label_file:
                _x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, txt = label.split(",")[:9]
                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1, _x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1, _x2, _y2, _x3, _y3]]
                _quad.append([_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3])
                _classes.append(1)
            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
    def get_SV1K(self):
        data_dir = self.root
        import glob
        dataset_list = glob.glob(self.root + "training/*.jpg")
        dataset_list = [l.split('/')[-1][:-4] for l in dataset_list if "jpg" in l]

        dataset_size = len(dataset_list)
        mode = 'train' if self.train else 'test'

        self.num_samples = dataset_size
        print(mode, "ing on USTB-SV1K : ", dataset_size)

        for i in dataset_list:
            img_file = data_dir + "training/%s.jpg" % (i)
            label_file = open(data_dir + "train_GT/gt_img_%s.txt" % (i))
            label_file = label_file.readlines()

            _quad=[]
            _classes = []

            for label in label_file:
                _x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3, txt = label.split(",")[:9]
                try:
                    _x0 = int(_x0)
                except:
                    _x0 = int(_x0[1:])

                _y0, _x1, _y1, _x2, _y2, _x3, _y3 = [int(p) for p in [_y0, _x1, _y1, _x2, _y2, _x3, _y3]]
                _quad.append([_x0, _y0, _x1, _y1, _x2, _y2, _x3, _y3])
                _classes.append(1)
            if len(_quad) is 0:
                self.num_samples -= 1
                continue
            self.fnames.append(img_file)
            self.boxes.append(np.array(_quad, dtype=np.float32))
            self.labels.append(np.array(_classes))
'''
import torchvision

from augmentations import Augmentation_traininig

dataset = ListDataset(root='/home/amax/GTK/SynthText/',
                      dataset='SynthText', train=True, transform=Augmentation_traininig, input_size=600, multi_scale=True)

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
count=0
for n, (img, boxes, labels,masks) in enumerate(dataloader):
    print(img.size(), boxes.size())
    #exit()

    img = img.data.numpy()
    img1 = img[7].transpose((1, 2, 0)) * 255

    img1 = np.array(img1, dtype=np.uint8)

    img1 = Image.fromarray(img1)
    draw = ImageDraw.Draw(img1)

    boxes = boxes.data.numpy()
    boxes1 = boxes[5].reshape(-1, 4, 2)

    for box in boxes1:
        draw.polygon(np.expand_dims(box,0), outline=(0,255,0))
    plt.imshow(img1)
    plt.imshow(masks[7])

    
    if n==0:
        break

def test2():
    import torchvision

    from augmentations import Augmentation_traininig
    
    dataset = ListDataset(root='/root/DB/',
                          dataset='ICDAR2015', train=True, transform=Augmentation_traininig, input_size=600, multi_scale=True)

    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    for i in range(10):
        data = dataset.__getitem__(i)
        
        random_choice = random.randint(0, len(dataset.MULTI_SCALES)-1)
        size = dataset.MULTI_SCALES[random_choice]
    
        img, boxes, labels = dataset.transform(size=size)(data['image'], data['boxes'], data['labels'])

        img = img.data.numpy()
        img = img.transpose((1, 2, 0))
        img *= (0.229,0.224,0.225)
        img += (0.485,0.456,0.406)
        img *= 255.
    
        img = np.array(img, dtype=np.uint8)

        boxes = boxes.data.numpy()
        boxes = boxes.reshape(-1, 4, 2).astype(np.int32)

        img = cv2.polylines(img, boxes, True, (255,0,0), 4)
        #cv2.imwrite('/home/beom/samba/%d.jpg' % i, img)
        
        img = Image.fromarray(img)
        img.save('/home/beom/samba/%d.jpg' % i)
'''

