
from time import time

from PIL import Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# D:\code\Python\Experiment\newNetwork\new
# D:\\code\\Python\\Experiment\\newNetwork\\new\\yszxx\\yszxx

# sys.path.append('D:\\code\\Python\\Experiment\\newNetwork\\new')
# sys.path.append('D:\\code\\Python\\Experiment\\newNetwork\\new\\yszxx\\yszxx')

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
# 10000-1

'''加载模型'''
net = Models.Lenet5.Lenet5()
net = Models.load_state_dict(net, './lenet5_dict.pkl')
num_correct, num_samples, acc = Optimizer.test(net, loader_test)
print('[Start] right predict:(%d/%d) ,pre test_acc=%.4f' % (num_correct, num_samples, acc))
# right predict:(936/10000) ,pre test_acc=9.3600

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

import numpy as np
img_adv=np.zeros([10000,28,28])   #  img=to_pil_image(X_adv[0])
K_all=0

for t, (X, y) in enumerate(loader_test):
    y_pred = Optimizer.pred_batch(X, net)  # 获得干净样本的结果
    X_adv = adversary.perturb(X.numpy(), y_pred)  # 加入扰动，生成对抗样本

    for K_temp in range(len(X_adv)):
        img = to_pil_image(X_adv[K_temp])
        img_adv[K_all] = np.asarray(img)
        K_all += 1

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


import PIL
from PIL import Image

img_org_img = to_pil_image(X[0])
img_org_np= np.asarray(img_org_img)
img_adv_img = to_pil_image(X_adv[0])
img_adv_np= np.asarray(img_adv_img)
img_diff_np=img_adv_np-img+img_org_np

plt.subplot(3, 3, 1)
plt.imshow(img_org_np,cmap='Greys_r')
plt.subplot(3, 3, 2)
plt.imshow(img_adv_np,cmap='Greys_r')
plt.subplot(3, 3, 3)
plt.imshow(img_diff_np,cmap='Greys_r')
plt.ioff()
plt.show()

#  cmap='Greys_r'
#print(img1)
#print(img01)
#print(img11)

import cv2
#cv2.imshow('img01',img_org_np)


    #('gray',img1)

# 均值滤波
img_adv_mean = cv2.blur(img_adv_np, (3,3)) #img_mean = cv2.blur(img_adv_np, (5,5))
# 高斯滤波
img_adv_Guassian = cv2.GaussianBlur(img_adv_np,(3,3),0) # cv2.GaussianBlur(img_adv_np,(5,5),0)
# 中值滤波
img_adv_median = cv2.medianBlur(img_adv_np, 3) #cv2.medianBlur(img_adv_np, 5)
# 双边滤波
img_adv_bilater = cv2.bilateralFilter(img_adv_np,9,75,75)

# 展示不同的图片
titles = ['origin','adv','adv_mean', 'adv_Gaussian', 'adv_median', 'adv_bilateral']
imgs = [img_org_np, img_adv_np, img_adv_mean, img_adv_Guassian, img_adv_median, img_adv_bilater]

for i in range(6):
    plt.subplot(2,3,i+1)    # 注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i],cmap='Greys_r')
    plt.title(titles[i])
plt.show()
#img_org_np

# img_adv_edges = cv2.Canny(img_adv_np, 75, 200)

plt.subplot(2,2,1)
img_org_edges = cv2.Canny(img_org_np, 75, 200)
plt.imshow(img_org_edges,cmap='Greys_r')
plt.title('org_edges')
#plt.show()
plt.subplot(2,2,2)
img_adv_bilater_edges = cv2.Canny(img_adv_bilater, 75, 200)
plt.imshow(img_adv_bilater_edges,cmap='Greys_r')
plt.title('adv_bilater_edges')
#plt.show()
plt.subplot(2,2,3)
img_adv_laplacian=cv2.Laplacian(img_adv_np, -1, ksize=3)
plt.imshow(img_adv_laplacian,cmap='Greys_r')
plt.title('adv_laplacian')
plt.subplot(2,2,4)
# img_adv_laplacian_2 = cv2.cvtColor(img_adv_laplacian, cv2.COLOR_BGR2GRAY)

ret,img_adv_laplacian_2=cv2.threshold(img_adv_laplacian, 127, 255, cv2.THRESH_BINARY)
plt.imshow(img_adv_laplacian_2,cmap='Greys_r')
plt.title('adv_laplacian_2')
plt.show()


#img1.shape







plt.subplot(3, 3, 1)
plt.matshow(img0)
plt.show()
plt.subplot(3, 3, 2)
plt.matshow(img1)
plt.show()
plt.subplot(3, 3, 3)
plt.matshow(img2)
#plt.ioff()
plt.show()

plt.matshow(img0)
plt.show()
plt.matshow(img1)
plt.show()
plt.matshow(img2)
#plt.ioff()
plt.show()

plt.show()

# plt.subplot(3, 5, 1)
# plt.matshow(to_pil_image(X[0]))
# plt.subplot(3, 5, 2)
# plt.imshow(to_pil_image(X_adv[i]))
# plt.subplot(3, 5, 3)
# plt.matshow(to_pil_image(X[0]-X_adv[0]))
# plt.ioff()
# plt.show()


# X[1]
# plt.matshow(img_adv[1, :, :])
# plt.show()

# test=img_adv[1]
# test.shape

# img1=Image.fromarray(np.uint8(img_adv[10000-1]))
# img1.show()


# img = np.asarray(image)


total_samples = len(loader_test.dataset)
acc = total_correct / total_samples
print('Got %d/%d correct (%.2f%%) on the perturbed data'
      % (total_correct, total_samples, 100 * acc))

print('Time: %.2f s.' %(time() - t0))  # 输出运行时间











####################################################################################
#
# ### the experiment group
# import cv2
#
# import numpy as np
#
# import PIL
# from PIL import Image
#
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
#
#
# import keras # 导入Keras
# from keras.datasets import mnist # 从keras中导入mnist数据集
# from keras.models import Sequential # 导入序贯模型
# from keras.layers import Dense # 导入全连接层
# from keras.optimizers import SGD
# from keras.utils import to_categorical
#
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
# from keras.losses import categorical_crossentropy
# from keras.optimizers import Adadelta
#
# from keras.datasets import cifar10
#
#
# (x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
# print('train:',len(x_img_train))
# print('test :',len(x_img_test))
# print('train_image :',x_img_train.shape)
# print('train_label :',y_label_train.shape)
# print('test_image :',x_img_test.shape)
# print('test_label :',y_label_test.shape)
#
# data=x_img_train[0]
#
#
# x_train_new=np.zeros((50000, 32, 32, 4))
# for k in range(50000):
#     x_train_new[k, :, :, 0:3] = x_img_train[k]
#     x_train_temp=x_img_train[k]
#     x_train_temp=cv2.bilateralFilter(x_train_temp, 9, 75, 75)
#     x_train_temp = cv2.Canny(x_train_temp, 75, 200)
#     #x_train_temp = cv2.Laplacian(x_train_temp, -1, ksize=3)
#     #x_train_temp = cv2.cvtColor(x_train_temp, cv2.COLOR_BGR2GRAY)
#     x_train_new[k, :, :, 3] = x_train_temp
#
# x_test_new=np.zeros((10000, 32, 32, 4))
# for k in range(10000):
#     x_test_new[k, :, :, 0:3] = x_img_test[k]
#     x_test_temp=x_img_test[k]
#     x_test_temp = cv2.bilateralFilter(x_test_temp, 9, 75, 75)
#     x_test_temp = cv2.Canny(x_test_temp, 75, 200)
#     #x_test_temp = cv2.Laplacian(x_test_temp, -1, ksize=3)
#     #x_test_temp = cv2.cvtColor(x_test_temp, cv2.COLOR_BGR2GRAY)
#     x_test_new[k, :, :, 3] = x_test_temp
#
# x_img_train=x_train_new
# x_img_test=x_test_new
#
#
# print("train data:",'images:',x_img_train.shape,
#       " labels:",y_label_train.shape)
# print("test  data:",'images:',x_img_test.shape ,
#       " labels:",y_label_test.shape)
# # 归一化
# x_img_train_normalize = x_img_train.astype('float32') / 255.0
# x_img_test_normalize = x_img_test.astype('float32') / 255.0
#
#
#
# # One-Hot Encoding
# from keras.utils import np_utils
# y_label_train_OneHot = np_utils.to_categorical(y_label_train)
# y_label_test_OneHot = np_utils.to_categorical(y_label_test)
# y_label_test_OneHot.shape
#
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
# # 搭面包架子
# model = Sequential()
# # 加面包：卷积层1 和 池化层1
# model.add(Conv2D(filters=32,kernel_size=(3,3),
#                  input_shape=(32, 32, 4),
#                  activation='relu',
#                  padding='same'))
# model.add(Dropout(rate=0.25))
# model.add(MaxPooling2D(pool_size=(2, 2))) # 16* 16
# # 加面包：卷积层2 和 池化层2
# model.add(Conv2D(filters=64, kernel_size=(3, 3),
#                  activation='relu', padding='same'))
# model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2))) # 8 * 8
# #Step3	建立神經網路(平坦層、隱藏層、輸出層)
# model.add(Flatten()) # FC1,64个8*8转化为1维向量
# model.add(Dropout(rate=0.25))
# model.add(Dense(1024, activation='relu')) # FC2 1024
# model.add(Dropout(rate=0.25))
# model.add(Dense(10, activation='softmax')) # Output 10
# print(model.summary())
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam', metrics=['accuracy'])
#
# train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
#                         validation_split=0.2,
#                         epochs=10, batch_size=128, verbose=1)
#
# model.save_weights("my_model.h5")
# # model.load_weights("my_model_weights.h5")
#
#
# import matplotlib.pyplot as plt
# def show_train_history(train_acc,test_acc):
#     plt.plot(train_history.history[train_acc])
#     plt.plot(train_history.history[test_acc])
#     plt.title('Train History')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#
# show_train_history('acc','val_acc')
#
# show_train_history('loss','val_loss')
#
# x_img_train_normalize = x_img_train.astype('float32') / 255.0
# x_img_test_normalize = x_img_test.astype('float32') / 255.0
#
# scores = model.evaluate(x_img_test_normalize,
#                         y_label_test_OneHot,verbose=0)
# scores[1]
#
# prediction=model.predict_classes(x_img_test_normalize)
# prediction[:10]
#
# label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
#               5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
# import matplotlib.pyplot as plt
#
#
# def plot_images_labels_prediction(images, labels, prediction,
#                                   idx, num=10):
#     fig = plt.gcf()
#     fig.set_size_inches(12, 14)
#     if num > 25: num = 25
#     for i in range(0, num):
#         ax = plt.subplot(5, 5, 1 + i)
#         ax.imshow(images[idx], cmap='binary')
#
#         title = str(i) + ',' + label_dict[labels[i][0]]
#         if len(prediction) > 0:
#             title += '=>' + label_dict[prediction[i]]
#
#         ax.set_title(title, fontsize=10)
#         ax.set_xticks([]);
#         ax.set_yticks([])
#         idx += 1
#     plt.show()
#
# plot_images_labels_prediction(x_img_test,y_label_test,prediction,0,10)
#
#
# Predicted_Probability = model.predict(x_img_test_normalize)
# label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
#               5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
# def show_Predicted_Probability(y, prediction, x_img, Predicted_Probability, i):
#     print('label:', label_dict[y[i][0]], 'predict:', label_dict[prediction[i]])
#     plt.figure(figsize=(2, 2))
#     plt.imshow(np.reshape(x_img_test[i], (32, 32, 4)))
#     plt.show()
#     for j in range(10):
#         print(label_dict[j] + ' Probability:%1.9f' % (Predicted_Probability[i][j]))
#
#
#
# show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,0)
#
# print(y_label_test)
# print(y_label_test.reshape(-1))# 转化为1维数组
# import pandas as pd
# print(label_dict)
# pd.crosstab(y_label_test.reshape(-1),prediction,rownames=['label'],colnames=['predict'])

