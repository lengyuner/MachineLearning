

if_new=1
adv_method=1
k_channel=1#mnist=1
#

import torch
import torch.nn as nn
#
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import torchvision

#
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#
import numpy as np
import cv2

'''自建包'''
import _YSZ_ as ysz
from _YSZ_ import *

'''
设置超参数
'''
param = {
    'model_path': './train2_advT.pkl',  # 模型存储路径
    'batch_size': 100,  # 训练时每次批量处理图片的数量
    'test_batch_size': 100,  # 测试时每次批处理图片的数量
    'num_epochs': 100,  # 对所有样本训练的轮数
    'learning_rate': 1e-4,  # 学习率
    # 'weight_decay': 5e-4,  # 权重衰减，相当于在损失函数后加入正则化项，降低整体权重值，防止过拟合
    'weight_decay': 0,  # 权重衰减，相当于在损失函数后加入正则化项，降低整体权重值，防止过拟合
    'epsilon': 0.02
}
batch_size=100
'''
加载数据集
'''
#train_dataset = datasets.MNIST(root='./Data3/',
#                                train=True,
#                                transform=transforms.ToTensor(),
#                                download=True)
# test_dataset = datasets.MNIST(root='./data20200113',train=False,transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,                                         batch_size=batch_size,shuffle=False)


train_dataset, test_dataset = Data_Reader.Mnist.Mnist_dataset().get_dataset()
#train_dataset=datasets
#train_dataset, test_dataset = Data_Reader.Cifar10.Cifar10_dataset().get_dataset()  # 训练、测试数据集
#dataset=train_dataset
if (if_new==1):
    def Laplace_img_data_in_torch(dataset):
        K_all = len(dataset)# K_all = 10
        # import torch
        dataset_new = []
        # train_dataset_new = torch.zeros(K_all)#[4,32,32][2]
        for k1 in range(K_all):  # k1=0
            img_data = dataset[k1][0]
            img_PIL = transforms.ToPILImage()(img_data)
            img_np = np.asarray(img_PIL)
            #img_laplace = cv2.Laplacian(img_np, -1, ksize=3)
            #img_laplace_grey = cv2.cvtColor(img_laplace, cv2.COLOR_BGR2GRAY)
            img_laplace_grey = cv2.bilateralFilter(img_np, 9, 75, 75)
            #x_train_0_org_bilateral_canny = cv2.Canny(x_train_0_org_bilateral, 75, 200)
            # x_img_laplace_matrix = np.asmatrix(img_laplace_grey)
            # plt.imshow(x_img_laplace_matrix, cmap='Greys_r')
            # plt.show()
            img_after_change_4 = np.zeros([(k_channel+1), 28, 28])
            for k3 in range(k_channel):
                img_after_change_4[k3] = img_np[:, :]#, k3]
            img_after_change_4[k_channel] = img_laplace_grey
            img_new_4 = torch.from_numpy(img_after_change_4 / 255)
            img_new_4 = img_new_4.float()
            dataset_new.append((img_new_4, dataset[k1][1]))
            # train_dataset_new[k1][0] = a
            # train_dataset_new[k1][1]=train_dataset[k1][1]
        return dataset_new


print('begin to change pictures from 3 to 4')
train_dataset_new = Laplace_img_data_in_torch(train_dataset)
test_dataset_new = Laplace_img_data_in_torch(test_dataset)
print('pictures have been changed from 3 to 4')
loader_train = Data_Reader.get_dataloader(dataset=train_dataset_new, batch_size=param['batch_size'])  # 训练集批量加载器
loader_test = Data_Reader.get_dataloader(dataset=test_dataset_new, batch_size=param['test_batch_size'])  # 测试集批量
    # 加载器

if (if_new==0):
    loader_train = Data_Reader.get_dataloader(dataset=train_dataset, batch_size=param['batch_size'])  # 训练集批量加载器
    loader_test = Data_Reader.get_dataloader(dataset=test_dataset, batch_size=param['test_batch_size'])  # 测试集批量



#loader_train = Data_Reader.get_dataloader(dataset=train_dataset, batch_size=param['batch_size'])  # 训练集批量加载器
# loader_test = Data_Reader.get_dataloader(dataset=test_dataset, batch_size=param['test_batch_size'])  # 测试集批量加载器

print('size of picture:')
print(train_dataset_new[0][0].shape)


'''
搭建模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F




modelpath = './train3_AdvT.pkl'
net = Models.LeNet5.Lenet5()  # 加载模型
net = Models.load_state_dict(net, modelpath)
Transformer.enable_cuda(net)  # 使用cuda
num_correct, num_samples, acc = Optimizer.test(net, loader_test)  # 测试一下最初的效果
print('[Start] right predict:(%d/%d) ,pre test_acc=%.4f%%' % (num_correct, num_samples, acc))




'''定义对抗训练方法'''
if (adv_method==1):
    adversary = Adversary.FGSM.FGSM(net, param['epsilon'])  # 攻击方法

if (adv_method==2):
    adversary = Adversary.LinfPGD.LinfPGDAttack(net, param['epsilon'])


'''
训练模型
'''
net.train()  # 模型模式->训练模式
criterion = nn.CrossEntropyLoss()  # 损失函数

optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
                                weight_decay=param['weight_decay'])  # 定义优化器
acc_all=np.zeros(50000)
K_all=0

for epoch in range(param['num_epochs']):  # 数据集训练的轮数=
    print('------------ Epoch %d / %d ------------' % (epoch + 1, param['num_epochs']))  # 输出当前轮数
    for t, (x, y) in enumerate(loader_train):  # 批量训练
        #x=x.float()
        #(x, y)=Laplace_img_data_in_torch((x, y))
        x_var, y_var = Transformer.to_var(x), Transformer.to_var(y.long())  # 转换格式
        X_adv = adversary.perturb(x.numpy(), y)  # 加入扰动，生成对抗样本
        advLoss = criterion(net(X_adv.cuda()), y_var)

        loss = (criterion(net(x_var), y_var) + advLoss)/2  # 计算损失
        optimizer.zero_grad()  # 把上一轮的梯度清零
        loss.backward()  # 反向传播求导
        optimizer.step()  # 优化参数

        if (t + 1) % 100 == 0:  # 每训练n批数据看一下loss
            print('t = %d, loss = %.8f' % (t + 1, loss.item()))
        if (t + 1) % 100 == 0:  # 每训练m批数据看一下测试精度，保存模型
            num_correct, num_samples, acc = Optimizer.test(net, loader_test)
            print('[train2_AdcT] E-t = %d-%d, right predict:(%d/%d) ,pre test_acc=%.4f%%' % (epoch + 1,
            t + 1, num_correct, num_samples, acc))
            torch.save(net.state_dict(), param['model_path'])  # 保存模型到文件
            K_all = K_all + 1
            acc_all[K_all]=acc
