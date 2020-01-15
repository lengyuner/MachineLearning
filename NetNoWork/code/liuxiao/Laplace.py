

import cv2
import numpy as np

import PIL
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import keras # 导入Keras
from keras.datasets import mnist # 从keras中导入mnist数据集
from keras.models import Sequential # 导入序贯模型
from keras.layers import Dense # 导入全连接层
from keras.optimizers import SGD
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

from keras.datasets import cifar10


(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()
print('train:',len(x_img_train))
print('test :',len(x_img_test))
print('train_image :',x_img_train.shape)
print('train_label :',y_label_train.shape)
print('test_image :',x_img_test.shape)
print('test_label :',y_label_test.shape)

#data=x_img_train[0]

def     Laplace_number(img):
    outcome=0
    for k1 in range(1,31):
        for k2 in range(1,31):
            outcome=outcome+img[k1-1,k2]+img[k1,k2-1]+img[k1+1,k2]+img[k1,k2+1]-4*img[k1,k2]
    return outcome

x_img=x_img_train[0]
x_img_grey=cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
x_img_matrix=np.asmatrix(x_img_grey)
#x_img_matrix.shape
x_number=Laplace_number(x_img_matrix)

x_img_laplace = cv2.Laplacian(x_img, -1, ksize=3)
x_img_laplace_grey = cv2.cvtColor(x_img_laplace, cv2.COLOR_BGR2GRAY)
x_img_laplace_grey.shape
x_img_laplace_matrix=np.asmatrix(x_img_laplace_grey)
x_laplace_number=Laplace_number(x_img_laplace_matrix)

print(x_number)
print(x_laplace_number)

n_row=2
n_col=2
i=1
plt.subplot(n_row, n_col, i)
plt.imshow(x_img)
plt.show()
i=i+1
plt.subplot(n_row, n_col, i)
plt.imshow(x_img_grey, cmap='Greys_r')
plt.show()
i=i+1
plt.subplot(n_row, n_col, i)
plt.imshow(x_img_laplace_matrix,cmap='Greys_r')
plt.show()






plt.subplot(2,2,1)
x_train_0_org=x_img_train[0,:,:,0:3]
plt.imshow(x_train_0_org)
plt.title('org')

plt.subplot(2,2,2)
x_train_0_org_bilateral=cv2.bilateralFilter(x_train_0_org, 9, 75, 75)
x_train_0_org_bilateral_canny = cv2.Canny(x_train_0_org_bilateral, 75, 200)
plt.imshow(x_train_0_org_bilateral_canny)
plt.title('org_bilateral_canny')

plt.subplot(2,2,3)
x_train_0_org_laplace=cv2.Laplacian(x_train_0_org, -1, ksize=3)
x_train_0_org_laplace_grey = cv2.cvtColor(x_train_0_org_laplace, cv2.COLOR_BGR2GRAY)
plt.imshow(x_train_0_org_laplace_grey)
plt.title('org_laplace_grey')

plt.show()

n_row=5
n_col=3
for K in range(n_row):
    print(K)
    plt.subplot(n_row, 3, K*3+1)
    x_train_0_org = x_img_train[K, :, :, 0:3]
    # img_adv_laplacian=cv2.Laplacian(img_adv_np, -1, ksize=3)
    plt.imshow(x_train_0_org)
    # plt.axis["top", 'right'].set_visible(False)
    if (K==0):
        plt.title('org')
    plt.subplot(n_row, 3, K*3+2)
    x_train_0_org_bilateral = cv2.bilateralFilter(x_train_0_org, 9, 75, 75)
    x_train_0_org_bilateral_canny = cv2.Canny(x_train_0_org_bilateral, 75, 200)
    plt.imshow(x_train_0_org_bilateral_canny)
    if (K==0):
        plt.title('org_bilateral_canny')
    plt.subplot(n_row, 3, K*3+3)
    x_train_0_org_laplace = cv2.Laplacian(x_train_0_org, -1, ksize=3)
    x_train_0_org_laplace_grey = cv2.cvtColor(x_train_0_org_laplace, cv2.COLOR_BGR2GRAY)
    plt.imshow(x_train_0_org_laplace_grey)
    if (K==0):
        plt.title('org_laplace_grey')

plt.show()













x_train_new=np.zeros((50000, 32, 32, 4))
for k in range(50000):
    x_train_new[k, :, :, 0:3] = x_img_train[k]
    x_train_temp=x_img_train[k]
    x_train_temp=cv2.bilateralFilter(x_train_temp, 9, 75, 75)
    x_train_temp = cv2.Canny(x_train_temp, 75, 200)
    #x_train_temp = cv2.Laplacian(x_train_temp, -1, ksize=3)
    #x_train_temp = cv2.cvtColor(x_train_temp, cv2.COLOR_BGR2GRAY)
    x_train_new[k, :, :, 3] = x_train_temp

