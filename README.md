# Industrial Scene Text Detection with Refined Feature-attentive Network 
This is the code of "Industrial Scene Text Detection with Refined Feature-attentive Network". 
For more details, please refer to our [TCSVT paper (Early Access)](https://ieeexplore.ieee.org/document/9726175) or [Poster](RFN_Poster.pdf).

[comment]: <> (and testing speed can reach 6.99 fps with 768px &#40;tested in single GPU of Tesla V100&#41;)
## Environments
- Ubuntu 16.04
- Cuda 10
- python >=3.5
- pytorch 1.0
- Other packages like cv2, Polygon3, tensorboardX, Scipy.

## Highlights
- **Training and evaluation checked:** Testing in MPSC test set with training data in {SynthMPSC, MPSC}. Other scene text datasets are test with pre-training data in SynthText.
- **Dataset link:**
  - MPSC&SynthMPSC: Please refer to [IndustrialTextDataset.md](IndustrialTextDataset.md) for dataset download.
  - [Synthtext](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
  - [MSRA-TD500](http://www.iapr-tc11.org/mediawiki/index.php/MSRA_Text_Detection_500_Database_(MSRA-TD500))
  - [ICDAR2013](https://rrc.cvc.uab.es/?ch=2&com=downloads)
  - [ICDAR2017-MLT](https://rrc.cvc.uab.es/?ch=8&com=downloads)
  - [USTB-SV1K](http://prir.ustb.edu.cn/TexStar/MOMV-text-detection/)
## Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## Configuring your dataset
- Update datset root path in`$RFN_ROOT/train.py`.
- Process dataset can be set in `$RFN_ROOT/tools/datagen.py`.  
- Modify test path in `$RFN_ROOT/multi_image_test_ocr.py`.
- Modify some settings in `$RFN_ROOT/tools/encoder.py`, including anchor_areas, aspect_ratios.
```bash
# refer to /data_process/Compute aspect_ratios and area_ratios.py
self.anchor_areas = [16*16., 32*32., 64*64., 128*128., 256*256, 512*512.]
self.aspect_ratios = [1., 2., 3., 5., 1./2., 1./3., 1./5.,7.]
```
## Training 
```bash
# create your data cache directory
cd RFN_ROOT
# Download pretrained ResNet50 model(https://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth)
# Init RFN with pretrained ResNet50 model
python ./tools/get_state_dict.py
python train.py --config_file=./configs/R_50_C4_1x_train.yaml
```

- The training size is set to a multiple of 128.
- Multi-GPU phase is not testing yet, be careful to use GPU more than 1.

## Test and eval
- Our provide script: `$RFN_ROOT/multi_image_test_ocr.py` and `$RFN_ROOT/test/`
- Modify path settings and choose the dataset you want to evaluate on.
- option parameters: save_img, show_mask
```bash
### test each image
python test.py --dataset=MPSC --config_file=./configs/R_50_C4_1x_train.yaml --test --save_img
```
```bash
### eval result
python test.py --dataset=MPSC --eval
```

## Pretrained Weights for Training and Testing
- Here we provide some pretained weights for testing in baidu drive:
```bash
  Pretrain SynthMPSC : https://pan.baidu.com/s/1BI2T4ncowKu908dcd9tT7g (0ke0)
```

## More Results 
- Model | Dataset | Precision | Recall | F-Measure | MODEL link | Extraction code
- RFN | MPSC | 89.30 | 83.33 | 86.21 | [model](https://pan.baidu.com/s/1j22FSpGBKQgPkVncvQ41ng) | 6u6y
- RFN* | MPSC | 89.82 | 84.45 | 87.05 | [model](https://pan.baidu.com/s/1lHUEmXKra9CTubBDR_a7xA) | xrni

## Visualizations of MPSC dataset
![examples1](visualization/MPSC.png)

## Visualizations of MSRA-TD500, USTB-SV1K, ICDAR2013, ICDAR2017-MLT dataset
![examples2](visualization/SceneTextDataset.png)

## Citation
```bash
If you find our method useful for your reserach, please cite

@ARTICLE{9726175,
  author={Guan, Tongkun and Gu, Chaochen and Lu, Changsheng and Tu, Jingzheng and Feng, Qi and Wu, Kaijie and Guan, Xinping},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Industrial Scene Text Detection With Refined Feature-Attentive Network}, 
  year={2022},
  volume={32},
  number={9},
  pages={6073-6085},
  doi={10.1109/TCSVT.2022.3156390}}

@article{guan2021industrial,
  title={Industrial Scene Text Detection with Refined Feature-attentive Network},
  author={Guan, Tongkun and Gu, Chaochen and Lu, Changsheng and Tu, Jingzheng and Feng, Qi and Wu, Kaijie and Guan, Xinping},
  journal={arXiv preprint arXiv:2110.12663},
  year={2021}
}
```
## License
```bash
- This code are only free for academic research purposes and licensed under the 2-clause BSD License - see the LICENSE file for details.
```
