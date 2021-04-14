import torch
import torch.nn as nn

from torchvision.models import resnet

class ResNetWrapper(nn.Module):
    def __init__(self,  num_classes=2,num_layers=18):
        super(ResNetWrapper, self).__init__()
        #base = resnet.resnet18(pretrained=False)
        if num_layers == 18:
            base = resnet.resnet18(pretrained=True)
        elif num_layers == 34:
            base = resnet.resnet34(pretrained=True)
        elif num_layers == 50:
            base = resnet.resnet50(pretrained=True)
        elif num_layers == 101:
            base = resnet.resnet101(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(4, 1)
		#self.avgpool = nn.AvgPool2d(4,1)

        self.dropout=nn.Dropout(0.5)
        self.fc = nn.Linear(512 , num_classes)


    def forward(self,x):
        x=self.in_block(x)
        x=self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x=self.avgpool(x)
        x=self.dropout(x)
        x = torch.flatten(x, 1)
        x=self.fc(x)

        return x


def get_resnet(class_num=2,num_layers=18):
    model=ResNetWrapper(class_num, num_layers)
    return  model


if __name__ == '__main__':
    net = ResNetWrapper(7, 18)
    input = torch.randn(1, 3, 128, 128, requires_grad=False)
    out = net(input)


