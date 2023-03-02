import torch
import torch.nn as nn


def conv3x3(input_chanels, output_chanels, stride=1, padding=1):
    return nn.Conv2d(input_chanels, output_chanels, kernel_size=3, stride=stride, padding=padding)


def conv1x1(input_chanels, output_chanels, stride=1, padding=1):
    return nn.Conv2d(input_chanels, output_chanels, kernel_size=1, stride=stride, padding=padding)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_chanels, output_chanels, stride=1, downsample=None, norm_Layer=None):
        super(BasicBlock, self).__init__()
        if norm_Layer is None:
            norm_Layer = nn.BatchNorm2d
        self.conv1 = conv3x3(input_chanels, output_chanels, stride, padding=1)
        self.bn1 = norm_Layer(output_chanels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(output_chanels, output_chanels, 1, padding=1)
        self.bn2 = norm_Layer(output_chanels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # print('idemtity shape pre', identity.shape)
        # print('self.downsample', self.downsample)

        if self.downsample:
            identity = self.downsample(x)

        # print('self.stride', self.stride)
        # print('out shape', out.shape)
        # print('idemtity shape', identity.shape)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_chanels, output_chanels, stride=1, downsample=None, norm_Layer=None):
        super(Bottleneck, self).__init__()
        if norm_Layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(input_chanels, output_chanels)
        self.bn1 = norm_layer(output_chanels)

        self.conv2 = conv3x3(output_chanels, output_chanels, stride)
        self.bn2 = norm_layer(output_chanels)

        self.conv3 = conv1x1(output_chanels, output_chanels * self.expansion)
        self.bn3 = norm_layer(output_chanels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classses=10, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.input_chanels = 64

        self.conv1 = nn.Conv2d(3, self.input_chanels, kernel_size=7, stride=2, padding=3)  # 224/2 = 112
        self.bn1 = norm_layer(self.input_chanels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 112/2 = 56

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classses)

    def _make_layer(self, block, output_chanels, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.input_chanels != output_chanels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.input_chanels, output_chanels * block.expansion, stride, 0),
                norm_layer(output_chanels * block.expansion)
            )

        layers = []
        layers.append(block(self.input_chanels, output_chanels, stride, downsample, norm_layer))
        self.input_chanels = output_chanels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.input_chanels, output_chanels, norm_Layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

#  [1,  1300] train_loss: 0.036  test_accuracy: 0.908


def resnet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])



# input1 = torch.rand([32,3,244,244])
model = resnet34() # 模式实例化
print(model) # 看一下模型结构
# output = model(input1)


