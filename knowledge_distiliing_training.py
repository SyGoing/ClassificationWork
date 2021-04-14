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
from dataset import MeanTeacherDataSet,MaskDataSet
from losses_mt import *
from torch.utils.data import  DataLoader
import time
from torchvision import transforms

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from utils import progress_bar, mixup_data, mixup_criterion,cutmix_data
from utils import checkpoint,adjust_learning_rate,adjust_learning_rate_warmup,get_mean_and_std


from torch.utils.tensorboard import SummaryWriter


def make_weights_for_balanced_classes(labels, nclasses):
    count = {}
    for item in labels:
        #if count.has_key(item):
        if item in count:
            count[item] += 1
        else:
            count[item]=1
    weight_per_class ={}
    N = len(labels)
    for key,value in count.items():
        weight_per_class[key] = N/float(value)
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight

def make_weights_for_balanced_classes_equal(labels):
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = 1
    return weight



# Training in normal way
def train_single_model(epoch,net,trainloader,criterion,optimizer,device,batches_per_epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    correct= 0
    total = 0
    train_class_loss=0
    for batch_idx, (x, targets) in enumerate(trainloader):
        batch_num = batch_idx + int(epoch * batches_per_epoch)
        adjust_learning_rate_warmup(optimizer, epoch, args.base_lr,batch_num)   #warming up training
        x=x.to(device)
        targets=targets.to(device)

        optimizer.zero_grad()
        outputs = net(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_class_loss += loss.data.item()

        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum()
        total += targets.size(0)

        progress_bar(batch_idx, len(trainloader),
                     'class_Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_class_loss / (batch_idx + 1),100 * correct / total, correct, total))

    res = {train_class_loss / batches_per_epoch,100 * correct / total}
    return res

def train_with_teacher(epoch,net,model_teacher,trainloader,criterion,optimizer,device,batches_per_epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    correct_student = 0
    correct_teacher=0
    total = 0

    train_class_loss=0
    if epoch%50==0:
        args.alpha=args.alpha*0.8
    for batch_idx, (x, targets) in enumerate(trainloader):
        batch_num = batch_idx + int(epoch * batches_per_epoch)
        adjust_learning_rate_warmup(optimizer, epoch, args.base_lr,batch_num)   #warming up training

        x=x.to(device)
        targets=targets.to(device)

        optimizer.zero_grad()
        student_out=net(x)
        teacher_out=model_teacher(x)

        _p = F.log_softmax(student_out / args.T, dim=1)
        _q = F.softmax(teacher_out / args.T, dim=1)
        soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        hard_loss=criterion(student_out,targets)
        loss=args.alpha*soft_loss*args.T*args.T+(1-args.alpha)*hard_loss

        loss.backward()
        optimizer.step()

        train_class_loss += hard_loss.data.item()

        _, predicted_student = torch.max(student_out.data, 1)
        _, predicted_teacher = torch.max(teacher_out.data, 1)
        correct_student+=predicted_student.eq(targets.data).cpu().sum()
        correct_teacher+=predicted_teacher.eq(targets.data).cpu().sum()
        total += targets.size(0)

        progress_bar(batch_idx, len(trainloader), 'class_Loss: %.3f | Acc_student: %.3f | Acc_teacher: %.3f%% (%d/%d)'
            % (train_class_loss/(batch_idx+1),100*correct_student/total,100*correct_teacher/total ,correct_student, total))

    res={train_class_loss/batches_per_epoch,correct_student/total,correct_teacher/total }
    return res

def test_single_model(epoch,net,testloader,criterion,device,batches_per_epoch):
    global best_acc
    net.eval()
    test_student_loss = 0

    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs=inputs.to(device)
        targets=targets.to(device)

        pred_student = net(inputs)

        class_loss = criterion(pred_student, targets)
        _, predicted = torch.max(pred_student.data, 1)
        test_student_loss+=class_loss.data.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'student_class_Loss: %.3f  | Acc_student: %.3f%% (%d/%d)' % (test_student_loss / (batch_idx + 1),  100. * correct / total, correct, total))

    # Save checkpoint.
    #test_student_acc= 100. * correct / total
    #checkpoint(test_student_acc, epoch, net)
    if not os.path.isdir(args.model_save):
        os.mkdir(args.model_save)
    if epoch%10==0:
        torch.save(net.state_dict(), args.model_save + '/model_student%d.pth' % epoch)

    res={test_student_loss/batches_per_epoch,100. * correct / total}
    return res

def test_with_teacher(epoch,net,model_teacher,testloader,criterion,device,batches_per_epoch):
    global best_acc
    net.eval()
    test_student_loss = 0
    test_teacher_loss=0
    correct1 = 0
    correct2 = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs.to(device)
        targets.to(device)

        pred_student = net(inputs)
        pred_teacher = model_teacher(inputs)

        class_loss1 = criterion(pred_student, targets)
        class_loss2 = criterion(pred_teacher, targets)

        _, predicted1 = torch.max(pred_student.data, 1)
        _, predicted2 = torch.max(pred_teacher.data, 1)

        test_student_loss+=class_loss1.data.item()
        test_teacher_loss+=class_loss2.data.item()

        total += targets.size(0)
        correct1 += predicted1.eq(targets.data).cpu().sum()
        correct2 += predicted2.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),
                     'student_class_Loss: %.3f | teacher_class_loss: %.3f | Acc_student: %.3f | Acc_teacher: %.3f%% (%d/%d)'
                     % (test_student_loss / (batch_idx + 1), test_teacher_loss / (batch_idx + 1),
                        100. * correct1 / total, 100. * correct2 / total, correct1, total))

    # Save checkpoint.
    test_student_acc= 100. * correct1 / total


    checkpoint(test_student_acc, epoch, net)

    res={test_student_loss/batches_per_epoch,test_teacher_loss / batches_per_epoch,100. * correct1 / total,100. * correct2 / total}

    return res

def main_train_with_teacher(args):
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")

    torch.manual_seed(args.seed)

    # device and super params
    use_cuda = torch.cuda.is_available()
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    batch_size = args.batch_size
    base_learning_rate = args.base_lr
    if use_cuda:
        n_gpu = 1
        batch_size *= n_gpu
        base_learning_rate *= n_gpu

    # Data
    print('==> Preparing data..')
    train_transform = TrainAugmentation(args.input_size, args.mean,args.std)
    test_transform =TestTransform(args.input_size, args.mean,args.std)

    train_dataset = MaskDataSet(label_file=args.train_label, data_root=args.data_root, transform=train_transform)
    #加入重采样
    sample_weights=make_weights_for_balanced_classes(train_dataset.labels,args.classNums)
    sampler_ = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


    test_dataset = MaskDataSet(label_file=args.test_label, data_root=args.data_root, transform=test_transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=False, num_workers=8,sampler=sampler_)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    num_samples=len(train_dataset)
    batches_per_epoch=num_samples/batch_size

    print('==> Building student and teacher model..')
    model_student=resnet10(num_classes=args.classNums)
    #model_student=Net(classNum=args.classNums)

    model_teacher=get_resnet(class_num=args.classNums,num_layers=34)
    model_teacher.load_state_dict(torch.load(args.teacher_model_path, map_location='cpu'))
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
    #class_criterion=nn.NLLLoss(weight=torch.from_numpy(criterion_weight).float().cuda(),ignore_index=-1)
    class_criterion = nn.CrossEntropyLoss()

    if args.optimizer=='adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_student.parameters()), lr=base_learning_rate, weight_decay=args.decay)  #固定参数
    elif args.optimizer=='sgd':
        optimizer = optim.SGD(model_student.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)

    writer=SummaryWriter('./logs')
    for epoch in range(start_epoch, args.epochs):
        s=time.time()
        for param_group in optimizer.param_groups:
            print('current learning rate is: ',param_group['lr'])

        train_class_loss, train_st_acc ,train_tea_acc= train_with_teacher(epoch, model_student, model_teacher,trainloader, class_criterion, optimizer,device,batches_per_epoch)
        test_class_loss, test_st_acc  = test_single_model(epoch, model_student, testloader, class_criterion, device,batches_per_epoch)
        e=time.time()
        print("cost time: ",(e-s)/60)
        print("train: ",train_class_loss,train_st_acc,train_tea_acc)
        print("test: ",test_class_loss,test_st_acc)

        writer.add_scalar('loss/train: ',train_class_loss)
        writer.add_scalar('Acc/train: ',train_st_acc)
        writer.add_scalar('loss/test: ', test_class_loss)
        writer.add_scalar('Acc/test: ', test_st_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch classify Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--sess', default='my_session_1', type=str, help='session id')
    parser.add_argument('--seed', default=11111, type=int, help='rng seed')

    parser.add_argument('--decay', default=5e-4, type=float, help='weight decay (default=1e-4)')
    parser.add_argument('--base_lr', default=4e-5, type=float, help='base learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='epochs of training')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_root', default='./DataSet', type=str, help='training images root dir')
    parser.add_argument('--test_data_dir', default='./DataSet/test_select/validneg', type=str, help='test data dir')
    parser.add_argument('--train_label', default='./DataSet/trainval_st.txt', type=str, help='images root dir')
    parser.add_argument('--test_label', default='./DataSet/test_st.txt', type=str, help='images root dir')
    parser.add_argument('--input_size', default=128, type=int, help='net input size')
    parser.add_argument('--mean', default=128.0, type=float, help='mean value 128')
    parser.add_argument('--std', default=128.0, type=float, help='norms 128')

    parser.add_argument('--classNums', default=8, type=int, help='class nums')
    parser.add_argument('--model_path', default="checkpoint/model_last.pth", type=str, help='model dir')
    parser.add_argument('--teacher_model_path', default="./checkpoint/mean_teacher/model_teacher180.pth", type=str, help='model dir')
    parser.add_argument('--save_hist', default="./DataSet", type=str, help='model dir')
    parser.add_argument('--optimizer',default='adam',type=str,help='optimizer type')
    parser.add_argument('--model_save', default='./checkpoint/knowledge_distilling')

    parser.add_argument('--T', default=6.0, type=float, help='model dir')
    parser.add_argument('--alpha', default=0.5, type=float, help='model dir')

    args = parser.parse_args()

    main_train_with_teacher(args)
