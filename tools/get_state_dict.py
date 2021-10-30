'''Init RFN with pretrained ResNet50 model.
Download model se_resnet50-ce0d4300.pth
https://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth
Other available lists:
resnet50-bn.pth : https://download.pytorch.org/models/resnet50-19c8e357.pth
resnet101-bn.pth : https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
resnet152-bn.pth : https://download.pytorch.org/models/resnet152-b121ed2d.pth
se-resnet50-bn : https://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth
se-resnet101-bn : https://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth
se-resnet152-bn : https://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from fpn import FPN50
from model import RFN
from maskrcnn_benchmark.config import cfg

print('Loading pretrained ResNet50 model..')
d = torch.load("../../../../se_resnet50-ce0d4300.pth")
print('Loading into FPN50..')
fpn = FPN50()
dd = fpn.state_dict()

for k in d.keys():
    #if not k.startswith('fc'):  # skip fc layers
    if 'last_linear' in k:
        print("break : ", k)
        break
    dd[k] = d[k]

print('Saving RFN..')
###focal_loss
config_file='../configs/R_50_C4_1x_train.yaml'
cfg.merge_from_file(config_file)
cfg.freeze()
net = RFN(1,input_size=768,bn_type=None,cfg=cfg)
###OHEM_loss
#net = RetinaNet(2)

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

pi = 0.01
init.constant_(net.cls_head[-1].bias, -math.log((1-pi)/pi))

net.fpn.load_state_dict(dd)
###focal_loss
torch.save(net.state_dict(), '../weights/retinanet_se50_with_mask.pth')
###OHEM_loss
#torch.save(net.state_dict(), 'weights/retinanet_v2_se50_OHEM.pth')
print('Done!')
