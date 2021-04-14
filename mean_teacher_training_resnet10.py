#coding=utf-8

from __future__ import print_function, division
from __future__ import absolute_import

import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from networks import *
from data_transforms import *
from dataset import MeanTeacherDataSet
from losses_mt import *
from torch.utils.data import  DataLoader
import time
from torchvision import transforms

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from utils import progress_bar, mixup_data, mixup_criterion,cutmix_data
from utils import checkpoint,adjust_learning_rate,adjust_learning_rate_warmup,get_mean_and_std

from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('./resnet10')

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# Training
def train(epoch,net,model_teacher,trainloader,criterion,optimizer,device,batches_per_epoch,batchSize):
    print('\nEpoch: %d' % epoch)
    net.train()
    correct_student = 0
    correct_teacher=0
    total = 0

    train_class_loss=0
    train_softmax_mse=0
	#device=torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    for batch_idx, (x1,x2, targets) in enumerate(trainloader):
        batch_num=batch_idx+int(epoch*batches_per_epoch)
        adjust_learning_rate(optimizer,epoch,args.base_lr)

        # if use_cuda:
        #     x1,x2,targets = x1.cuda(), x2.cuda(),targets.cuda()

        x1=x1.to(device)
        x2=x2.to(device)
        targets=targets.to(device)

        pred_student=net(x1)
        pred_teacher=model_teacher(x2)

        lsm = nn.LogSoftmax(dim=1)
        y1=lsm(pred_student)

        #class_loss = criterion(pred_student,targets)
        
        class_loss = criterion(y1,targets)
        softmax_mse= softmax_mse_loss(pred_student,pred_teacher)/batchSize
        sum_loss=class_loss+softmax_mse
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()

        update_ema_variables(net,model_teacher,0.99,batch_num)

        train_class_loss += class_loss.data.item()
        train_softmax_mse+=softmax_mse

        _, predicted_student = torch.max(pred_student.data, 1)
        _, predicted_teacher = torch.max(pred_teacher.data, 1)
        correct_student+=predicted_student.eq(targets.data).cpu().sum()
        correct_teacher+=predicted_teacher.eq(targets.data).cpu().sum()
        total += targets.size(0)


        if batch_num%100==0:
            writer.add_scalar('train/class_loss: ', train_class_loss/(batch_idx+1))
            writer.add_scalar('train/mse_loss: ', train_softmax_mse/(batch_idx+1))
            writer.add_scalar('train_student/Acc: ', 100*correct_student/total)
            writer.add_scalar('train_teacher/Acc: ', 100*correct_teacher/total)

        progress_bar(batch_idx, len(trainloader), 'class_Loss: %.3f | mse_loss: %.3f | Acc_student: %.3f | Acc_teacher: %.3f%% (%d/%d)'
            % (train_class_loss/(batch_idx+1),train_softmax_mse/(batch_idx+1), 100*correct_student/total,100*correct_teacher/total ,correct_student, total))

    #checkpoint(100*correct_student/total, epoch, net)
    if not os.path.isdir(args.model_save):
        os.mkdir(args.model_save)
    torch.save(net.state_dict(), args.model_save+'/model_last%d.pth' % epoch)
    res={train_class_loss/batches_per_epoch,train_softmax_mse/batches_per_epoch,correct_student/total,correct_teacher/total }
    return res


def test(epoch,net,testloader,criterion,device,batches_per_epoch):
    net.eval()
    test_student_loss = 0
    correct = 0
    total = 0
    batch_num=0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        batch_num = batch_idx + int(epoch * batches_per_epoch)
        inputs=inputs.to(device)
        targets=targets.to(device)

        pred_student = net(inputs)
        class_loss = criterion(pred_student, targets)

        _, predicted = torch.max(pred_student.data, 1)

        test_student_loss+=class_loss.item()
        total += targets.size(0)
        correct+= predicted.eq(targets.data).cpu().sum()

        if batch_num%100==0:
            writer.add_scalar('test/class_loss: ', test_student_loss / (batch_idx + 1))
            writer.add_scalar('test/Acc: ', 100. * correct / total)

        progress_bar(batch_idx, len(testloader),
                     'student_class_Loss: %.3f  | Acc_student: %.3f%% (%d/%d)'
                     % (test_student_loss / (batch_idx + 1), 100. * correct / total,  correct, total))
        batch_num+=1


    res={test_student_loss/batch_num,100. * correct / total}

    return res

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
        n_gpu = 1
        batch_size *= n_gpu
        base_learning_rate *= n_gpu

    # Data
    print('==> Preparing data..')
    #train_transform = MeanTeacherTrain(args.input_size)
    train_transform=TrainAugmentation(args.input_size, args.mean,args.std)
    train_dataset = MeanTeacherDataSet(label_file=args.train_label, data_root=args.data_root, transform=train_transform)
    trainloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=8)

    test_transform=TestTransform(args.input_size, args.mean,args.std)
    test_dataset = MeanTeacherDataSet(label_file=args.test_label, data_root=args.data_root, transform=test_transform,is_train=False)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    num_samples=len(train_dataset)
    batches_per_epoch=num_samples/batch_size

    print('==> Building student and teacher model..')

    model_student=resnet10(7)  #resnet18
    model_teacher=resnet10(7)

    #model_student=Net(7)
    #model_teacher=Net(7)
    
    for param in model_teacher.parameters():
        param.detach_()

    if use_cuda:
        model_student.to(device)
        model_teacher.to(device)
        # net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')



    hist_path = os.path.join(args.save_hist, 'hist')
    if os.path.isfile(hist_path + '.npy'):
        hist = np.load(hist_path + '.npy')
    else:
        # Get class weights based on training data
        hist = np.zeros((args.classNums), dtype=np.float)
        for batch_idx, (x1,x2, yt) in enumerate(trainloader):
            h, bins = np.histogram(yt.numpy(), list(range(args.classNums+1)))
            hist += h

        hist = hist / (max(hist))  # Normalize histogram
        np.save(hist_path, hist)

    criterion_weight = 1 / np.log(1.02 + hist)
    print("class_weights: ",criterion_weight)
    class_criterion=nn.NLLLoss(weight=torch.from_numpy(criterion_weight).float().to(device),ignore_index=-1)
    #class_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_student.parameters()), lr=base_learning_rate, weight_decay=args.decay)  #固定参数
    #optimizer = optim.SGD(model_student.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)

    for epoch in range(start_epoch, args.epochs):
        s=time.time()
        for param_group in optimizer.param_groups:
            print('current learning rate is: ',param_group['lr'])
        train_class_loss,train_mes_loss, train_st_acc ,train_tea_acc= train(epoch, model_student, model_teacher,trainloader, class_criterion, optimizer,device,batches_per_epoch,batch_size)
        test_class_loss, test_st_acc = test(epoch,model_student, testloader, class_criterion, device,batches_per_epoch)
        e=time.time()
        print("cost time: ",(e-s)/60)
        print("train: ",train_class_loss,train_st_acc,train_tea_acc)
        print("test_loss: ", test_class_loss, "   test_acc: ", test_st_acc)

def main_test(args):
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)

    # device and super params
    use_cuda = torch.cuda.is_available()
    batch_size = args.batch_size
    base_learning_rate = args.base_lr
    if use_cuda:
        n_gpu = 1
        batch_size *= n_gpu
        base_learning_rate *= n_gpu
        cudnn.benchmark = True

    test_transform=TestTransform(args.input_size, args.mean,args.std)
    test_dataset = MeanTeacherDataSet(label_file=args.test_label, data_root=args.data_root, transform=test_transform,is_train=False)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    print('==> Building student and teacher model..')
    model_student = get_resnet(class_num=args.classNums,num_layers=18)
    model_student.to(device)
    model_student.load_state_dict(torch.load(args.model_path))

    criterion = nn.CrossEntropyLoss()
    test_class_loss,  test_st_acc= test(model_student, testloader,criterion, device)
    print("test_loss: ",test_class_loss,"   test_acc: ",test_st_acc)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch classify Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default='my_session_1', type=str, help='session id')
    parser.add_argument('--seed', default=11111, type=int, help='rng seed')
    parser.add_argument('--alpha', default=0.999, type=float, help='interpolation strength (uniform=1., ERM=0.)')
    parser.add_argument('--decay', default=5e-4, type=float, help='weight decay (default=1e-4)')
    parser.add_argument('--base_lr', default=4e-5, type=float, help='base learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='epochs of training')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_root', default='./DataSet', type=str, help='training images root dir')
    parser.add_argument('--test_data_dir', default='./DataSet/test_select/validneg', type=str, help='test data dir')
    parser.add_argument('--train_label', default='./DataSet/trainval.txt', type=str, help='images root dir')
    parser.add_argument('--test_label', default='./DataSet/test.txt', type=str, help='images root dir')
    parser.add_argument('--input_size', default=128, type=int, help='net input size')
    parser.add_argument('--mean', default=128.0, type=float, help='mean value 128')
    parser.add_argument('--std', default=128.0, type=float, help='norms 128')
    parser.add_argument('--classNums', default=8, type=int, help='class nums')
    parser.add_argument('--model_path', default="checkpoint/model_last199.pth", type=str, help='model dir')
    parser.add_argument('--save_hist', default="./DataSet", type=str, help='model dir')
    parser.add_argument('--model_save',default='./checkpoint/mean_teacher_resnet10')

    args = parser.parse_args()

    main_train(args)
    #main_test(args)
