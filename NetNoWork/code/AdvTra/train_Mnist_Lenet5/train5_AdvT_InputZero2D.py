import torch
import torch.nn as nn
import copy

'''自建包'''
from _YSZ_ import *

# ------------------------------

'''
设置超参数
'''
param = {
    'batch_size': 100,  # 训练时每次批量处理图片的数量
    'test_batch_size': 100,  # 测试时每次批处理图片的数量
    'num_epochs': 1000,  # 对所有样本训练的轮数
    # 'learning_rate': 1e-4,  # 学习率
    'learning_rate': 1e-5,  # 学习率
    'weight_decay': 5e-5,  # 权重衰减，相当于在损失函数后加入正则化项，降低整体权重值，防止过拟合
    # 'weight_decay': 0,  # 权重衰减，相当于在损失函数后加入正则化项，降低整体权重值，防止过拟合
    'epsilon': 0.2,
    'model_path': './train5_AdvT_InputZero2D_wd.pkl'  # 模型被保存的位置
}

'''
加载数据集
'''
train_dataset, test_dataset = Data_Reader.Mnist.Mnist_dataset().get_dataset()
loader_train = Data_Reader.get_dataloader(dataset=train_dataset,
                                          batch_size=param['batch_size'])
loader_test = Data_Reader.get_dataloader(dataset=test_dataset,
                                         batch_size=param['test_batch_size'])

'''
搭建模型
模型在model.py里面搭建好了，这里直接调用
'''

modelpath = param['model_path']
net = Models.LeNet5.Lenet5()  # 加载模型
net = Models.load_state_dict(net, modelpath)
Transformer.enable_cuda(net)  # 使用cuda
num_correct, num_samples, acc = Optimizer.test(net, loader_test)  # 测试一下最初的效果
print('[Start] right predict:(%d/%d) ,pre test_acc=%.4f%%' % (num_correct, num_samples, acc))

'''
训练模型
'''
net.train()  # 训练模式
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
                                weight_decay=param['weight_decay'])  # 优化器，具体怎么优化，学习率、正则化等等

adversary = Adversary.LinfPGD.LinfPGDAttack(net, param['epsilon'])

for epoch in range(param['num_epochs']):
    print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
    for t, (x, y) in enumerate(loader_train):
        x_var, y_var = Transformer.to_var(x), Transformer.to_var(y.long())  # 转换格式
        x_calcVar = copy.copy(x_var)
        out = net(x_var)

        # 计算input的梯度
        x_calcVar = copy.copy(x_var)
        x_calcVar.requires_grad = True
        for p in net.parameters():
            p.requires_grad = True
        scores = net(x_calcVar)
        losss = nn.CrossEntropyLoss()(scores, y_var)
        gradX = torch.autograd.grad(losss, x_calcVar, create_graph=True)  # 计算input的一阶导数
        gradX = gradX[0].view(out.size(0), -1)
        gradXloss = sum(torch.abs(gradX[0]))  # 计算input的梯度之和作为gradloss

        X_adv = adversary.perturb(x.numpy(), y)  # 加入扰动，生成对抗样本

        # 优化loss=gradloss+原始的loss
        x_var.requires_grad = False
        loss = gradXloss + (criterion(out, y_var) + criterion(net(X_adv.cuda()), y_var)) / 2
        optimizer.zero_grad()  # 把上一轮的梯度清零
        loss.backward()  # 反向传播求导
        optimizer.step()  # 优化参数

        if (t + 1) % 10 == 0:  # 每训练10批数据看一下loss
            print('t = %d, loss = %.8f, gradXloss=%.8f' % (t + 1, loss.item(), gradXloss))
        if (t + 1) % 100 == 0:
            num_correct, num_samples, acc = Optimizer.test(net, loader_test)
            print('[train] t = %d, right predict:(%d/%d) ,pre test_acc=%.4f%%' % (t + 1, num_correct, num_samples, acc))
            torch.save(net.state_dict(), modelpath)  # 保存模型到文件
