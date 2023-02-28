import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # 要继承于nn.Moudule父类
    def __init__(self):
        # 初始化函数

        super(LeNet, self).__init__()
        # 使用super函数，解决多继承可能遇到的一些问题；调用基类的构造函数


        self.conv1 = nn.Conv2d(3, 16, 5) # 调用卷积层 （in_channels,out_channels(也是卷积核个数。输出的通道数),kernel_size（卷积核大小）,stride）
        self.pool1 = nn.MaxPool2d(2, 2)  # 最大池化层，进行下采样
        self.conv2 = nn.Conv2d(16, 32, 5) # 输出的通道数为32
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32*5*5, 120) # 全连接层输入是一维向量，这里是32x5x5，我们要展平，120是节点的个数
        # 32是通道数
        # Linear(input_features,output_features)

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x是输入数据，是一个tensor
        # 正向传播
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        # 数据通过view展成一维向量，第一个参数-1是batch，自动推理；32x5x5是展平后的个数
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        # 为什么没有用softmax函数 --- 在网络模型中已经计算交叉熵以及概率
        return x

import torch
input1 = torch.rand([32,3,32,32])
model = LeNet() # 模式实例化
print(model) # 看一下模型结构
output = model(input1)


#  train_loss: 0.034  test_accuracy: 0.704

