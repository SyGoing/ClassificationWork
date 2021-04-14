import torch.nn as nn
import torch
class SmallNet(nn.Module):
    def __init__(self,classNum):
        super(SmallNet,self).__init__()
        self.class_num=classNum
        self.conv1=nn.Conv2d(3, 32, 3, stride=1)
        self.prelu1=nn.PReLU()
        self.pool1=nn.MaxPool2d(3, stride=2)


        self.net_Conv=nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 2, stride=1),
            nn.PReLU(),
        )
        self.net_Linear=nn.Sequential(
            nn.Linear(128 * 6 * 6,256),
            nn.Dropout(0.5),
            nn.PReLU(),
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
    net = SmallNet(classNum=2)
    input = torch.randn(1, 3, 48, 48, requires_grad=False)
    out = net(input)

