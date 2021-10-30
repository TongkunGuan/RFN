## Environments
- Ubuntu 16.04
- Cuda 10.0 or Cuda 10.1
- python >=3.5 (we set 3.6.2)
- pytorch 1.0
- Other packages like cv2, Polygon3, tensorboardX.
## Installation
```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name RFN python==3.xxx
source activate RFN

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib

# follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch

# install pycocotools
cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd ~/github
git clone https://github.com/TongkunGuan/RFN.git
cd RFN
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

#-------
python rotation_setup.py install

# If you use Python 3.x to compile, the lib folder will be like 'lib.xxxxx'
mv build/lib/rotation/*.so ./rotation
#-------
