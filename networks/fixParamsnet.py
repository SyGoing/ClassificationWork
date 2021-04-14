import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet

class FixedNet(nn.Module):
    def __init__(self):
        super(FixedNet,self).__init__()
        basemodel=EfficientNet.from_pretrained('efficientnet-b6',num_classes=2)
        self.conv_stem =basemodel._conv_stem
        self.bn0 = basemodel._bn0
        self.blocks=basemodel._blocks

        # 固定模型部分参数，不进行训练
        for p in self.parameters():
            p.requires_grad = False

        self.conv_head = basemodel._conv_head
        self.bn1 = basemodel._bn1



        # 仅仅训练全连接的新任务
        self.avg_pooling = basemodel._avg_pooling
        self.dropout = basemodel._dropout
        self.fc =basemodel._fc
        self._swish =basemodel._swish

        self._global_params=basemodel._global_params

    def forward(self,inputs):
        bs = inputs.size(0)

        #extract features
        x = self._swish(self.bn0(self.conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self.bn1(self.conv_head(x)))

        x = self.avg_pooling(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x








