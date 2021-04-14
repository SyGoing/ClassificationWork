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

### Usage

1.  Prepare Data

   - Firstly,put your dataset to the DataSet dir. Different class with different dir . Here is a demo: 

	DataSet/maskdata, you refer this dir structure

   pos(face with mask)

   neg(face without mask)

   - Secondly, use the train_val_caffe.py to generate the label list file. Modify the line50-54 according your dataset

     ```
     # classes name list
     classNames=["neg","pos"]
     cat_ids = {v: i for i, v in enumerate(classNames)}
     #class data root in Dataset
     train_root="maskdata/train"
     test_root="maskdata/val"
     ```

2. Training

   ​	You can choose different .py to train your model. For common training method, you can choose the common_train.py , Just modify  the params in the argparse for your task and data.  And there are other methods such as:

   -   knowledge_distiliing_training.py,

   -   mean_teacher_training.py

   -   mean_teacher_training_resnet10.py

3. export to onnx

   please refer the export_onnx.py, and modify some params(argparse) according to your model