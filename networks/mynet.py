import torch.nn as nn
import torch
class Net(nn.Module):
    def __init__(self,classNum):
        super(Net,self).__init__()
        self.class_num=classNum
        self.net_Conv=nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.LeakyReLU(inplace=True),
        )
        self.net_Linear=nn.Sequential(
            nn.Linear(256 * 6 * 6,256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.class_num)
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
        x=torch.flatten(x, 1)
        x=self.net_Linear(x)
        return x


if __name__ == '__main__':
    net = Net(classNum=7)
    input = torch.randn(1, 3, 128, 128, requires_grad=False)
    out = net(input)

