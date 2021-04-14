import torch.nn as nn
import torch
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.net_Conv=nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
        )
        self.net_Linear=nn.Sequential(
            nn.Linear(128 * 6 * 6,128),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self,inputs):
        x=self.net_Conv(inputs)
        x=x.view(-1,self.num_flat_features(x))
        x=self.net_Linear(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s in size:
            num_features*=s
        return num_features
