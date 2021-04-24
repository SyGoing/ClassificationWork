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
import shutil
import argparse
import csv
import cv2

from networks import *
from efficientnet_pytorch import EfficientNet #https://github.com/lukemelas/EfficientNet-PyTorch
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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def sgd_optimizer(model, lr, momentum, weight_decay):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if 'bias' in key or 'bn' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            apply_lr = 2 * lr       #   Just a Caffe-style common practice. Made no difference.
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer

def warmup_training_stage(optimizer,base_learning_rate,batch_id,burn_in=1000):
    if batch_id<burn_in:
        lr=base_learning_rate*pow(float(batch_id)/float(burn_in),4)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# 3 mixup argumentation training
# Training
def train(epoch,net,trainloader,criterion,optimizer,use_cuda,batches_per_epoch,mixType,device,lr_scheduler,burnIn):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    nb=len(trainloader)
    pbar = enumerate(trainloader)
    pbar = tqdm(pbar, total=nb)

    #for batch_idx, (inputs, targets) in enumerate(trainloader):
    for batch_idx, (inputs, targets) in pbar:
        batch_num=batch_idx+int(epoch*batches_per_epoch)
        warmup_training_stage(optimizer,args.base_lr,batch_num,burn_in=burnIn)
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        if mixType=='mixup':
            inputs, targets_a, targets_b, lam = mixup_data(device,inputs, targets, args.alpha, use_cuda)
        elif mixType=='cutmix':
            inputs, targets_a, targets_b, lam = cutmix_data(device,inputs, targets, args.alpha, use_cuda)

        optimizer.zero_grad()
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        outputs = net(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()

        # if batch_num>burnIn:
        #     lr_scheduler.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return (train_loss/batch_idx, 100.*correct/total,batch_num)

def test(net,testloader,criterion,use_cuda,device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    nb=len(testloader)
    pbar = enumerate(testloader)
    pbar = tqdm(pbar, total=nb)

    #for batch_idx, (inputs, targets) in enumerate(testloader):
    for batch_idx, (inputs, targets) in pbar:
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    return (test_loss/batch_idx, acc)

def infer(model,transform_forward,img):
    input=transform_forward(img)
    input = input.unsqueeze(0)
    input=input.cuda()

    out=model(input)
    #pred = F.softmax(out)
    #pred = torch.argmax(out,1)
    pred=out.cpu()
    result=pred.detach().numpy()
    id = torch.argmax(out, 1)

    # if result[0][0]>0.5:
    #     return 0,result[0][0]
    # else:
    #     return 1,result[0][1]
    return id

    print(result)

    #return result[0][]

def main_train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % args.device
    device_cuda_str="cuda:"+args.device
    device = torch.device(device_cuda_str if (torch.cuda.is_available()) else "cpu")

    torch.manual_seed(args.seed)
    # device and super params
    use_cuda = torch.cuda.is_available()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batch_size = args.batch_size
    base_learning_rate = args.base_lr
    burn_in = 424
    if use_cuda:
        # data parallel
        n_gpu = torch.cuda.device_count()
        batch_size *= n_gpu
        base_learning_rate *= n_gpu

    # Data
    print('==> Preparing data..')
    train_transform = TrainAugmentation(args.input_size, args.mean,args.std)
    test_transform =TestTransform(args.input_size, args.mean,args.std)
    train_dataset = MaskDataSet(label_file=args.train_label, data_root=args.data_root, transform=train_transform)
    test_dataset = MaskDataSet(label_file=args.test_label, data_root=args.data_root, transform=test_transform)


    trainloader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)

    num_samples=len(train_dataset)
    batches_per_epoch=num_samples/batch_size


    print('==> Building model..')
    if args.model_arch.find("RepVGG")!=-1:
        repvgg_build_func = get_RepVGG_func_by_name(args.model_arch)
        net = repvgg_build_func(deploy=False, numClasses=2)
    elif args.model_arch.find("resnet")!=-1:
        id=args.model_arch.find("_")+1
        num_layers=int(args.model_arch[id:])
        net=get_resnet(class_num=args.classNums,num_layers=num_layers)
    elif args.model_arch.find("efficientnet")!=-1:
        net=EfficientNet.from_pretrained(args.model_arch,num_classes=args.classNums)
    elif args.model_arch.find("res10")!=-1:
        net=resnet10(args.classNums)

    if use_cuda:
        if torch.cuda.device_count()>1:
            net = torch.nn.DataParallel(net).cuda()
        elif torch.cuda.device_count()==1:
            net.to(device)

        # net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print('Using CUDA..',args.device)

    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.Adam(net.parameters(), lr=base_learning_rate, weight_decay=args.decay)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=base_learning_rate, weight_decay=args.decay)  #固定参数
    optimizer=optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=args.epochs)

    writer = SummaryWriter('./logs')
    best_acc=0
    for epoch in range(start_epoch, args.epochs):
        for param_group in optimizer.param_groups:
            print('current learning rate is: ',param_group['lr'])
            break
        train_loss, train_acc ,batch_num= train(epoch, net, trainloader, criterion, optimizer,use_cuda,batches_per_epoch,
                                      mixType="mixup",device=device,lr_scheduler=lr_scheduler,burnIn=burn_in)
        test_loss, test_acc = test(net, testloader, criterion, use_cuda,device)

        if batch_num>burn_in:
            lr_scheduler.step(epoch)

        state = {
            'epoch': epoch + 1,
            'arch': args.model_arch,
            'state_dict': net.state_dict(),
            'best_acc1': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }
        if not os.path.isdir(args.model_save):
            os.mkdir(args.model_save)

        filename = args.model_save + '/model_last%d.pth' % epoch
        filebestName=args.model_save+"/model_best.pth"
        torch.save(state, filename)
        if test_acc > best_acc:
            shutil.copyfile(filename, filebestName)

        print(train_loss, train_acc)
        print(test_loss, test_acc)
        writer.add_scalar('loss/train: ',train_loss)
        writer.add_scalar('Acc/train: ',train_acc)
        writer.add_scalar('loss/test: ', test_loss)
        writer.add_scalar('Acc/test: ', test_acc)


def main_test_videos(args,videos_dir):
    #指定前向设备
    device =  torch.device("cuda:0" if (torch.cuda.is_available()
                                        ) else "cpu")

    #模型加载
    net = MobileNetv2(2,ave_size=7,is_train=True)
    net.load_state_dict(torch.load(args.model_path))
    net=net.cuda()
    net.eval()
    #数据预处理
    transform_forward0 = TestTransform(args.input_size, args.mean,args.std)

    name = {0: 'neg', 1: 'pos'}
    videofiles=os.listdir(videos_dir)
    total_num=len(videofiles)
    posNum=0
    negNum=0
    acc_num=0


    id=0

    for file in videofiles:
        id+=1
        videoDir=os.path.join(videos_dir,file)
        vc = cv2.VideoCapture(videoDir)
        rval = vc.isOpened()

        neg_num=0
        pos_num=1

        total=0
        while (rval):
            rval, frame = vc.read()

            if rval:
                total+=1
                img=frame[:,165:,:]
                result, score = infer(net, transform_forward0, img)

                if result==1:
                    pos_num+=1
                else:
                    neg_num+=1

                cv2.putText(frame, name[result], (22, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.namedWindow('test', cv2.WINDOW_NORMAL)
                cv2.imshow('test', frame)
                cv2.waitKey(1)

        resFlag=0
        if neg_num<pos_num:
            resFlag = 1
            posNum+=1
            print(id,"have tie: ",file," ",pos_num)
        else:
            negNum+=1
            print(id,"have no tie: ", file," ",neg_num)
        print("total: ",total)





        cv2.waitKey()

        vc.release()

    ratio=float(posNum)/float(total_num)

    print("posNUM: ",posNum)
    print("totalNum: ",total_num)
    print("Right Ratio: ", ratio)


    pass

def main_test_imgs(args):
    #指定前向设备
    device =  torch.device("cuda:0" if (torch.cuda.is_available()
                                        ) else "cpu")

    #模型加载
    # net= torch.hub.load(
    # 'moskomule/senet.pytorch',
    # 'se_resnet50',
    # pretrained=True,)
    # net.fc= nn.Linear(2048 , 2)
    #net=EfficientNet.from_pretrained('efficientnet-b5',num_classes=2,)
    #net = ResNet(args.classNums, 50)
    #net = MobileNetv2(2,ave_size=7,is_train=True)
    net = ResNet(7)
    net.load_state_dict(torch.load(args.model_path,map_location='cpu'))
    net=net.cuda()


    net.eval()

    #数据预处理
    transform_forward0 = TestTransform(args.input_size, args.mean,args.std)

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

        #img=image[:,330:,:]
        img = image
        height, width, _ =  img.shape

        save_img=img.copy()


        # name={0:'neg',1:'pos'}
        name = {0: 'listen', 1: 'write',2:'read',3:'stand',4:'lean',5:'hand_up',6:'others'}


        result,score=infer(net,transform_forward0, img)

        print("%d,%s,%f" % (id,name[result], score))

        cv2.putText(image, name[result], (22,  22),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        cv2.namedWindow('test',cv2.WINDOW_NORMAL)
        cv2.imshow('test',image)


        if result==0:
            neg_count+=1
            #print("neg num: ",neg_count)
            #print(file)
            #shutil.move(jpg_path,os.path.join("./DataSet/tmp_selcet/neg", file))
            #cv2.imwrite(os.path.join("./DataSet/test_select/neg1", file), save_img)
            #cv2.waitKey()
        else:
            pos_count+=1
           # print("pos num: ", pos_count)
            #shutil.move(jpg_path, os.path.join("./DataSet/tmp_selcet/pos", file))
            #cv2.imwrite(os.path.join("./DataSet/test_select/pos1", file), save_img)

        cv2.waitKey()
    print("neg num: ",neg_count)
    print("pos num:",pos_count)
    print("recall: ",float(pos_count)/float(neg_count+pos_count))



    # for id in range(num):
    #     jpg_path=os.path.join(img_root,id.__str__()+'.jpg')
    #     image =cv2.imread(jpg_path)
    #     height, width, _ = image.shape
    #
    #     name={0:'neg',1:'pos'}
    #
    #     result,score=infer(net,transform_forward0,image)
    #     print("%d,%s,%f" %(id,name[result],score))
    #     f.write("%d,%s" %(id,name[result]) + '\n')
    #
    #     cv2.putText(image, name[result], (22,  22),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    #
    #     cv2.namedWindow('test',cv2.WINDOW_NORMAL)
    #     cv2.imshow('test',image)
    #     cv2.waitKey(1)

def export_onnx():
    # 模型加载
    # net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    # net.set_swish(memory_efficient=False)
    # net.load_state_dict(torch.load("checkpoint/model_last.pth"))

    #net = resnet10(num_classes=args.classNums)
    net = Net(2)
    net.load_state_dict(torch.load("model_student100.pth",map_location='cpu'))
    net.eval()
    input = torch.randn(1, 3, 128, 128, requires_grad=False)
    out = net(input)
    torch.onnx.export(net,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      "./mynet_kinder.onnx",  # where to save the model (can be a file or file-like object)
                      verbose=True,
                      )


    print("hello finished")

def export_RepVgg():
    repvgg_build_func = get_RepVGG_func_by_name(args.model_arch)
    net = repvgg_build_func(deploy=False, numClasses=2)
    net.load_state_dict(torch.load(args.model_path,map_location='cpu'))

    deploy_model=repvgg_model_convert(net)
    deploy_model.eval()

    input = torch.randn(1, 3, 128, 128, requires_grad=False)
    out = net(input)
    torch.onnx.export(net,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      "./RepVggA01.onnx",  # where to save the model (can be a file or file-like object)
                      verbose=True,
                      )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch classify Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default='my_session_1', type=str, help='session id')
    parser.add_argument('--seed', default=11111, type=int, help='rng seed')
    parser.add_argument('--alpha', default=1.2, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    parser.add_argument('--decay', default=5e-4, type=float, help='weight decay (default=1e-4)')
    parser.add_argument('--base_lr', default=4e-5,type=float,help='base learning rate')
    parser.add_argument('--epochs',default=300,type=int,help='epochs of training')
    parser.add_argument('--batch_size', default=8,type=int)
    parser.add_argument('--data_root',default='./DataSet',type=str, help='training images root dir')
    parser.add_argument('--test_data_dir',default='E:/AVA_AND_KINDERGATEN/student/subsec/handup',type=str,help='test data dir')
    parser.add_argument('--train_label',default='./DataSet/trainval.txt',type=str, help='images root dir')
    parser.add_argument('--test_label', default='./DataSet/trainval.txt', type=str, help='images root dir')
    parser.add_argument('--input_size',default=128,type=int,help='net input size')
    parser.add_argument('--mean',default=128.0,type=float,help='mean value 128')
    parser.add_argument('--std',default=128.0,type=float,help='norms 128')
    parser.add_argument('--classNums',default=2,type=int, help='class nums')
    parser.add_argument('--model_path', default="checkpoint/model_last11.pth", type=str, help='model file path')
    parser.add_argument('--model_save', default="checkpoint/common_train", type=str, help='model save dir')
    parser.add_argument('--model_arch',default="RepVGG-A0")
    parser.add_argument('--device', default="0",help="cpu,0,1,2,3.....")


    args = parser.parse_args()

    best_acc = 0  # best test accuracy"
    #
    main_train(args)
    #main_test_imgs(args)
    #main_test_videos(args,'E:/save_tie/videos')


    #export_onnx()
   # export_RepVgg()