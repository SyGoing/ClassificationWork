#coding=utf-8

from __future__ import print_function, division
from __future__ import absolute_import

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import  DataLoader

import torch.backends.cudnn as cudnn
import os
import argparse
import csv
import cv2

from networks import *
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

from utils import progress_bar, mixup_data, mixup_criterion,cutmix_data
from utils import checkpoint,adjust_learning_rate,adjust_learning_rate_warmup,get_mean_and_std
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

#1 数据预处理以及数据集制作
from data_transforms import TrainAugmentation,TestTransform
from dataset import MaskDataSet
from label_smooth import *

import torch.hub
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import time

# 3 mixup argumentation training
# Training
def train(device,epoch,net,trainloader,criterion,optimizer,use_cuda,batches_per_epoch,mixType='mixup'):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_num=batch_idx+int(epoch*batches_per_epoch)
        adjust_learning_rate_warmup(optimizer,epoch,args.base_lr,batch_num)
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        if mixType=='mixup':
            inputs, targets_a, targets_b, lam = mixup_data(device,inputs, targets, args.alpha, use_cuda)
        elif mixType=='cutmix':
            inputs, targets_a, targets_b, lam =cutmix_data(device,inputs, targets, args.alpha, use_cuda)

        optimizer.zero_grad()
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        outputs = net(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return (train_loss/batch_idx, 100.*correct/total)

def test(device,epoch,net,testloader,criterion,use_cuda):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    #     best_acc = acc
    #     checkpoint(acc, epoch,net)
    #checkpoint(acc, epoch, net)
    if not os.path.isdir(args.model_save):
        os.mkdir(args.model_save)
    torch.save(net.state_dict(), args.model_save+'/model_last%d.pth' % epoch)
    return (test_loss/batch_idx, 100.*correct/total)

def infer(model,transform_forward,img):
    input=transform_forward(img)
    input = input.unsqueeze(0)
    input=input.cuda()
    out=model(input)
    pred = F.softmax(out)
    #pred = torch.argmax(out,1)
    pred=pred.cpu()
    result=pred.detach().numpy()

    if result[0][0]>0.5:
        return 0,result[0][0]
    else:
        return 1,result[0][1]


def main_train(args):
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)

    # device and super params
    use_cuda = torch.cuda.is_available()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batch_size = args.batch_size
    base_learning_rate = args.base_lr
    burn_in = 424
    if use_cuda:
        # data parallel
        n_gpu = 1
        batch_size *= n_gpu
        base_learning_rate *= n_gpu

    # Data
    print('==> Preparing data..')
    # means,std=get_mean_and_std(MaskDataSet(label_file=args.train_label, data_root=args.data_root, transform=None))
    # print(means)
    # print(std)


    train_transform = TrainAugmentation(args.input_size, args.mean,args.std)
    test_transform =TestTransform(args.input_size, args.mean,args.std)

    train_dataset = MaskDataSet(label_file=args.train_label, data_root=args.data_root, transform=train_transform)
    test_dataset = MaskDataSet(label_file=args.test_label, data_root=args.data_root, transform=test_transform)


    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=8)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    num_samples=len(train_dataset)
    batches_per_epoch=num_samples/batch_size


    print('==> Building model..')
    #net=EfficientNet.from_pretrained('efficientnet-b5',num_classes=2)

    #net=MobileNetv2()
    #net=Net(7)
    #net=MobileNetV3_Large(7)
    net=resnet10(7)
    result_folder = './results/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    logname = result_folder + net.__class__.__name__ + '_' + args.sess + '_' + str(args.seed) + '.csv'

    if use_cuda:
        net.to(device)
        # net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')






    criterion = nn.CrossEntropyLoss()
    #criterion=LabelSmoothSoftmaxCEV1()

    #optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)
    optimizer = optim.Adam(net.parameters(), lr=base_learning_rate, weight_decay=args.decay)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=base_learning_rate, weight_decay=args.decay)  #固定参数

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

    for epoch in range(start_epoch, args.epochs):

        s=time.time()
        for param_group in optimizer.param_groups:
            print('current learning rate is: ',param_group['lr'])
        train_loss, train_acc = train(device,epoch, net, trainloader, criterion, optimizer,use_cuda,batches_per_epoch,mixType=args.mixType)
        test_loss, test_acc = test(device,epoch, net, testloader, criterion, use_cuda)
        e=time.time()


        print("cost time: ",(e-s),"s")
        print(train_loss, train_acc)
        print(test_loss, test_acc)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])


def main_test_imgs(args):
    #指定前向设备
    device =  torch.device("cuda:2" if (torch.cuda.is_available() ) else "cpu")

    #模型加载
    # net= torch.hub.load(
    # 'moskomule/senet.pytorch',
    # 'se_resnet50',
    # pretrained=True,)
    # net.fc= nn.Linear(2048 , 2)
    #net=EfficientNet.from_pretrained('efficientnet-b5',num_classes=2,)
    #net = ResNet(args.classNums, 50)
    net = MobileNetv2()

    net.load_state_dict(torch.load("checkpoint/model_last126.pth"))
    net=net.cuda()


    net.eval()

    #数据预处理
    transform_forward = TestTransform(args.input_size, args.mean,args.std)
    #transform_forward = TestTransform(args.input_size, [120.5996, 126.4566, 135.5834], [65.1329, 65.2607, 66.7312])

    img_root=args.test_data_dir
    files=os.listdir(img_root)
    num=len(files)


    f=open("res.csv",'w')

    neg_count=0
    pos_count=0
    id=0
    for file in files:
        id += 1
        jpg_path = os.path.join(img_root, file)
        image =cv2.imread(jpg_path)
        height, width, _ = image.shape
        save_img=image.copy()

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        name={0:'neg',1:'pos'}

        result,score=infer(net,transform_forward,image)

        print("%d,%s,%f" % (id,name[result], score))

        cv2.putText(image, name[result], (22,  22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        cv2.namedWindow('test',cv2.WINDOW_NORMAL)
        cv2.imshow('test',image)


        if result==0:
            neg_count+=1
            #cv2.imwrite(os.path.join("./DataSet/tmp_selcet/neg", file), save_img)
        else:
            pos_count+=1
            #cv2.imwrite(os.path.join("./DataSet/tmp_selcet/pos", file), save_img)

        cv2.waitKey()
    print("recall: ",float(pos_count)/float(neg_count+pos_count))



    # for id in range(num):
    #     jpg_path=os.path.join(img_root,id.__str__()+'.jpg')
    #     image =cv2.imread(jpg_path)
    #     height, width, _ = image.shape
    #     #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     name={0:'neg',1:'pos'}
    #
    #     result,score=infer(net,transform_forward,image)
    #     print("%d,%s,%f" %(id,name[result],score))
    #     f.write("%d,%s" %(id,name[result]) + '\n')
    #
    #     cv2.putText(image, name[result], (22,  22),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #
    #     cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    #     cv2.imshow('test',image)
    #     cv2.waitKey()

def export_onnx():
    # 模型加载
    # net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    # net.set_swish(memory_efficient=False)
    # net.load_state_dict(torch.load("checkpoint/model_last.pth"))

    net = MobileNetv2()
    net.load_state_dict(torch.load("checkpoint/model_last126.pth"))
    net.eval()

    input = torch.randn(1, 3, 224, 224, requires_grad=False)

    out = net(input)

    torch.onnx.export(net,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      "./mobilenetv2.onnx",  # where to save the model (can be a file or file-like object)
                      verbose=True,
                      )


    print("hello finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch classify Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default='my_session_1', type=str, help='session id')
    parser.add_argument('--seed', default=11111, type=int, help='rng seed')

    parser.add_argument('--data_root',default='./DataSet',type=str, help='training images root dir')
    parser.add_argument('--test_data_dir',default='./DataSet/tmp_test',type=str,help='test data dir')
    parser.add_argument('--train_label',default='./DataSet/trainval.txt',type=str, help='images root dir')
    parser.add_argument('--test_label', default='./DataSet/test.txt', type=str, help='images root dir')
    parser.add_argument('--input_size',default=8,type=int,help='net input size')
    parser.add_argument('--mean',default=128.0,type=float,help='mean value 128')
    parser.add_argument('--std',default=128.0,type=float,help='norms 128')
    parser.add_argument('--mixType',default='mixup',type=str,help="mixup or cutmix for data augmentation")

    parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
    parser.add_argument('--decay', default=5e-4, type=float, help='weight decay (default=1e-4)')
    parser.add_argument('--base_lr', default=4e-5,type=float,help='base learning rate')
    parser.add_argument('--epochs',default=300,type=int,help='epochs of training')
    parser.add_argument('--batch_size', default=128,type=int)
    parser.add_argument('--model_save', default='./checkpoint/resnet10_mixup')


    parser.add_argument('--classNums',default=7,type=int, help='class nums')
    parser.add_argument('--save_hist', default="./DataSet", type=str, help='model dir')



    args = parser.parse_args()

    best_acc = 0  # best test accuracy
    #
    main_train(args)
    #main_test_imgs(args)

    #export_onnx()
