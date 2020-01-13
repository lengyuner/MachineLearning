"""
其他论文中的实验模型：NeurIPS 2018 paper， Deep Defense: Training DNNs with Improved Adversarial Robustness.
https://github.com/ZiangYan/deepdefense.pytorch

适用于：Cifar10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, padding=2)
        # 3
        #x = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, groups=1)
        #self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4)
        self.conv5 = nn.Conv2d(64, 10, 1)

        for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            w = self.__getattr__(k)
            torch.nn.init.kaiming_normal_(w.weight.data)
            w.bias.data.fill_(0)

        self.out = dict()

    def forward(self, x):
        # print(size(x))
        # x=torch.float32(x)
        # print('_______________________________')

        x = self.conv1(x)

        # print(x.size())
        x, pool1_ind = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # print(x.size())
        x = F.relu(x)

        # print(x.size())
        x = self.conv2(x)

        # print(x.size())
        x = F.relu(x)
        # print(x.size())
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # print(x.size())
        x = self.conv3(x)

        # print(x.size())
        x = F.relu(x)

        # print(x.size())
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)

        x = x.view(-1, 10)

        return x
