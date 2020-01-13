

if_new=1
adv_method=1
import torch
import torch.nn as nn
#
import torchvision
from torchvision import transforms
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

'''
加载数据集
'''
train_dataset, test_dataset = Data_Reader.Cifar10.Cifar10_dataset().get_dataset()  # 训练、测试数据集

if (if_new==1):
    def Laplace_img_data_in_torch(dataset):
        K_all = len(dataset)
        # import torch
        dataset_new = []
        # train_dataset_new = torch.zeros(K_all)#[4,32,32][2]
        for k1 in range(K_all):  # k1=0
            img_data = dataset[k1][0]
            img_PIL = transforms.ToPILImage()(img_data)
            img_np = np.asarray(img_PIL)
            img_laplace = cv2.Laplacian(img_np, -1, ksize=3)
            img_laplace_grey = cv2.cvtColor(img_laplace, cv2.COLOR_BGR2GRAY)
            # x_img_laplace_matrix = np.asmatrix(img_laplace_grey)
            # plt.imshow(x_img_laplace_matrix, cmap='Greys_r')
            # plt.show()
            img_after_change_4 = np.zeros([4, 32, 32])
            for k3 in range(3):
                img_after_change_4[k3] = img_np[:, :, k3]
            img_after_change_4[3] = img_laplace_grey
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
#print(train_dataset_new[0][0].shape)

# train_dataset[1]
#
# new_tensor=torch.zeros([2,2])
# new_tensor[0]=1
# new_tensor=torch.Tensor(2, 3)
# new_tensor[1]=1
# new_tensor
# tensor(new_tensor[0],1)
#
# b=train_dataset[0][0]
# b.shape
#
# Laplace_img_data_in_torch(train_dataset)



# img_data=train_dataset[0][0]
# img_PIL = transforms.ToPILImage()(img_data)
# img_np=np.asarray(img_PIL)




# img_laplace = cv2.Laplacian(img_np, -1, ksize=3)
# img_laplace_grey = cv2.cvtColor(img_laplace, cv2.COLOR_BGR2GRAY)



# x_img_laplace_matrix=np.asmatrix(img_laplace_grey)
# plt.imshow(x_img_laplace_matrix,cmap='Greys_r')
# plt.show()
#
# img_after_change_4=np.zeros([4,32,32])
# for k3 in range(3):
#     img_after_change_4[k3]=img_np[:,:,k3]
# img_after_change_4[3]=img_laplace_grey
# train_dataset_new=(torch.from_numpy(img_after_change_4/255),
#                    train_dataset[0][1])


# a=torch.from_numpy(x_img_laplace_matrix/255)
# a1=train_dataset[0]
# a1.__sizeof__()
# a1.shape
# # a1[1]
# # Out[79]: 6
# a1[1]=7

# a2=(a,train_dataset[0][1])
# a1[0]=a[5]


# img_np.shape
# (32, 32, 3)

# for k1 in range(1):#len(train_dataset)):
#     k1=0
#     img_data=train_dataset[k1][0]
#     #   tensor->numpy
#     # img_np=img_data.numpy()*255
#     # img_np
#     img_PIL = transforms.ToPILImage()(img_data)
#     np.asarray(img_PIL)
#
#     #   tensor->PIL
#
#     #img_data=transforms.ToPILImage()(img_data)
#
#     #.shape[0]


# train_dataset[0][0]
# tuple(train_dataset[0])
# a=train_dataset[0][0]
# print(a)
# import numpy as np
# np.asmatrix(a)
# #t_out = torch.randn(3,10,10)
# #img1 = transforms.ToPILImage()(t_out)
# #img1.show()
#
#
# img1 = transforms.ToPILImage()(train_dataset[0][0])
# img1.show()
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
# plt.imshow(img1)
# plt.show()


'''
搭建模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


#net=ConvNet()
net = ysz.Models.ConvNet.ConvNet()
net = Models.load_state_dict(net, param['model_path'])  # 如果已经有参数则继续训练
Transformer.enable_cuda(net)  # 模型调用cuda计算(如果可用)


#
num_correct, num_samples, acc = Optimizer.test(net, loader_test)  # 测试一下最初的效果
print('[Current] right predict:(%d/%d), pre test_acc=%.4f%%' % (num_correct, num_samples, acc))  # 输出模型当前精度

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
