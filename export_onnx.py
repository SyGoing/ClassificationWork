#coding=utf-8
import torch
from networks import *
from efficientnet_pytorch import EfficientNet #https://github.com/lukemelas/EfficientNet-PyTorch
import argparse
import os

def export_onnx(args):
    # 模型加载
    # net = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    # net.set_swish(memory_efficient=False)
    # net.load_state_dict(torch.load("checkpoint/model_last.pth"))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % args.device
    device_cuda_str="cuda:"+args.device
    device = torch.device(device_cuda_str if (torch.cuda.is_available()) else "cpu")

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
    elif args.model_arch.find("mynet")!=-1:
        net=Net(args.classNums)

    net.load_state_dict(torch.load(args.model_path,map_location=device))
    net.eval()
    net.to(device)

    input = torch.randn(1, 3, args.input_size, args.input_size, requires_grad=False)
    input=input.to(device)

    out = net(input)

    torch.onnx.export(
                      net,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      args.onnx_save,  # where to save the model (can be a file or file-like object)
                      verbose=True,
                      opset_version=11
                      )
    print("export to onnx succeed !")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch classify Training')
    parser.add_argument('--input_size',default=128,type=int,help='net input size')
    parser.add_argument('--classNums',default=2,type=int, help='class nums')
    parser.add_argument('--model_path', default="checkpoint/model_student100.pth", type=str, help='model file path')
    parser.add_argument('--onnx_save', default="onnxmodel/model.onnx", type=str, help='model save dir')
    parser.add_argument('--model_arch',default="mynet")
    parser.add_argument('--device', default="0",help="cpu,0,1,2,3.....")

    args = parser.parse_args()
    export_onnx(args)