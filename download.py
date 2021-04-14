from efficientnet_pytorch import EfficientNet

net=EfficientNet.from_pretrained('efficientnet-b1',num_classes=2)

print('hello')