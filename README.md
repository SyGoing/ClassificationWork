### Training code for Common Classification task 

​		This repo is used to train the common classification model with many backbones and tricks (algorithm) for improve the model's  performance. I implement it with pytorch.

- Data Augmentation: mixup, cut-mix, random crop, flip, random hsv, random brightness and so on.
- Some Algorithm: mean-teacher, knowledge-distilling, warm-up training
- backbone: resnet, mobilenetv2, mobilenetv3, resnet10, chostnet, efficientnet,efficientv2,RepVgg, shufflenetv1/v2. Some models can be create by the pretrained models in torchbision

### Requirments

- torch >=1.7.1+cu101
- python3.7
- numpy>=1.12.1
- scipy>=0.19.0
- Cython
- pycocotools
- lap
- motmetrics
- opencv-python
- funcy