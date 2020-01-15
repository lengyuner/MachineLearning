


from time import time
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

'''自建包'''
import _Base_ as base
import Data_Reader
import Models
import Optimizer
import Adversary

'''超参数'''
param = {
    'test_batch_size': 100,
    'epsilon': 0.2,
}

'''加载测试数据'''
test_dataset = Data_Reader.Mnist.Mnist_dataset().get_test_dataset()
loader_test = Data_Reader.get_dataloader(dataset=test_dataset, batch_size=param['test_batch_size'], shuffle=False)

'''加载模型'''
net = Models.Lenet5.Lenet5()
net = Models.load_state_dict(net, './lenet5_dict.pkl')
num_correct, num_samples, acc = Optimizer.test(net, loader_test)
print('[Start] right predict:(%d/%d) ,pre test_acc=%.4f' % (num_correct, num_samples, acc))

'''模型测试过程'''
base.enable_cuda(net)  # 使用cuda
net.eval()  # 推断模式
for p in net.parameters():  # 将模型参数的梯度获取设为false（）
    p.requires_grad = False
Optimizer.test(net, loader_test)  # 测试干净样本的性能

'''对抗样本攻击'''
adversary = Adversary.FGSM.FGSM(net, param['epsilon'])  # 攻击方法
#adversary = Adversary.LinfPGD.LinfPGDAttack(net, param['epsilon'])
t0 = time()  # 当前时间

total_correct = 0
total_samples = 0
for t, (X, y) in enumerate(loader_test):
    y_pred = Optimizer.pred_batch(X, net)  # 获得干净样本的结果
    X_adv = adversary.perturb(X.numpy(), y_pred)  # 加入扰动，生成对抗样本

    y_pred_adv = Optimizer.pred_batch(X_adv, net)  # 预测对抗样本

    total_correct += (y_pred_adv.numpy() == y.numpy()).sum()  # 正确的个数
    total_samples += param['test_batch_size']  # 总个数

    if t % 10 == 0:#输出结果
        acc = total_correct / total_samples
        print('Got %d/%d correct (%.2f%%) on the perturbed data'
              % (total_correct, total_samples, 100 * acc))

    # 画对比图
    if t == 0:
        img1 = to_pil_image(X_adv[0])
        img2 = to_pil_image(X[0])
        for i in range(5):
            plt.subplot(3, 5, i + 1)
            plt.imshow(to_pil_image(X[i]))
        for i in range(5):
            plt.subplot(3, 5, 5 + i + 1)
            plt.imshow(to_pil_image(X_adv[i]))
        plt.ioff()
        plt.show()


total_samples = len(loader_test.dataset)
acc = total_correct / total_samples
print('Got %d/%d correct (%.2f%%) on the perturbed data'
      % (total_correct, total_samples, 100 * acc))

print('Time: %.2f s.' %(time() - t0))  # 输出运行时间
