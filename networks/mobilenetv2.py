import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v2

class MobileNetv2(nn.Module):
    def __init__(self,num_classes=2,ave_size=4,is_train=False):
        super(MobileNetv2, self).__init__()
        net=mobilenet_v2(pretrained=True)
        self.features=net.features
        self.avgpool = nn.AvgPool2d(ave_size,1)
        self.is_train=is_train

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes),
        )

    def forward(self,x):
        x=self.features(x)
        #x = x.mean([2, 3])
        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.classifier(x)

        if not self.is_train:
            x=F.softmax(x)

        return x
