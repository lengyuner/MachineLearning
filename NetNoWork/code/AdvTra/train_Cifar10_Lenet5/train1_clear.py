import torch
import torch.nn as nn

'''自建包'''
import _YSZ_ as ysz
from _YSZ_ import *

'''
设置超参数
'''
param = {
    'model_path': './train1_clear.pkl',  # 模型存储路径
    'batch_size': 10,  # 训练时每次批量处理图片的数量
    'test_batch_size': 10,  # 测试时每次批处理图片的数量
    'num_epochs': 100,  # 对所有样本训练的轮数
    # 'learning_rate': 1e-4,  # 学习率
    'learning_rate': 1e-5,  # 学习率
    # 'weight_decay': 5e-5,  # 权重衰减，相当于在损失函数后加入正则化项，降低整体权重值，防止过拟合
    'weight_decay': 0,  # 权重衰减，相当于在损失函数后加入正则化项，降低整体权重值，防止过拟合
    'epsilon': 0.02
}
'''
加载数据集
'''
train_dataset, test_dataset = Data_Reader.Cifar10.Cifar10_dataset().get_dataset()  # 训练、测试数据集
loader_train = Data_Reader.get_dataloader(dataset=train_dataset, batch_size=param['batch_size'])  # 训练集批量加载器
loader_test = Data_Reader.get_dataloader(dataset=test_dataset, batch_size=param['test_batch_size'])  # 测试集批量加载器

'''
搭建模型
'''
net = Models.LeNet5.Lenet5_cifar10()  # 加载模型
net = Models.load_state_dict(net, param['model_path'])  # 如果已经有参数则继续训练

Transformer.enable_cuda(net)  # 模型调用cuda计算(如果可用)
num_correct, num_samples, acc = Optimizer.test(net, loader_test)  # 测试一下最初的效果
print('[Current] right predict:(%d/%d), pre test_acc=%.4f%%' % (num_correct, num_samples, acc))  # 输出模型当前精度

'''
训练模型
'''
net.train()  # 模型模式->训练模式
criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
                                weight_decay=param['weight_decay'])  # 定义优化器

for epoch in range(param['num_epochs']):  # 数据集训练的轮数
    print('------------ Epoch %d / %d ------------' % (epoch + 1, param['num_epochs']))  # 输出当前轮数

    for t, (x, y) in enumerate(loader_train):  # 批量训练
        x_var, y_var = Transformer.to_var(x), Transformer.to_var(y.long())  # 转换格式
        loss = criterion(net(x_var), y_var)  # 计算损失
        optimizer.zero_grad()  # 把上一轮的梯度清零
        loss.backward()  # 反向传播求导
        optimizer.step()  # 优化参数

        if (t + 1) % 100 == 0:  # 每训练n批数据看一下loss
            print('t = %d, loss = %.8f' % (t + 1, loss.item()))
        if (t + 1) % 100 == 0:  # 每训练m批数据看一下测试精度，保存模型
            num_correct, num_samples, acc = Optimizer.test(net, loader_test)
            print('[train] t = %d, right predict:(%d/%d) ,pre test_acc=%.4f%%' % (t + 1, num_correct, num_samples, acc))
            torch.save(net.state_dict(), param['model_path'])  # 保存模型到文件

            # print("wait:60s......")
            # Optimizer.wait(60)

