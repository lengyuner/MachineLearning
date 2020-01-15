
### the experiment group

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

data=x_img_train[0]

x_train_new=np.zeros((50000, 32, 32, 4))
for k in range(50000):
    x_train_new[k, :, :, 0:3] = x_img_train[k]
    x_train_temp=x_img_train[k]
    x_train_temp=cv2.bilateralFilter(x_train_temp, 9, 75, 75)
    x_train_temp = cv2.Canny(x_train_temp, 75, 200)
    #x_train_temp = cv2.Laplacian(x_train_temp, -1, ksize=3)
    #x_train_temp = cv2.cvtColor(x_train_temp, cv2.COLOR_BGR2GRAY)
    x_train_new[k, :, :, 3] = x_train_temp

x_test_new=np.zeros((10000, 32, 32, 4))
for k in range(10000):
    x_test_new[k, :, :, 0:3] = x_img_test[k]
    x_test_temp=x_img_test[k]
    x_test_temp = cv2.bilateralFilter(x_test_temp, 9, 75, 75)
    x_test_temp = cv2.Canny(x_test_temp, 75, 200)
    #x_test_temp = cv2.Laplacian(x_test_temp, -1, ksize=3)
    #x_test_temp = cv2.cvtColor(x_test_temp, cv2.COLOR_BGR2GRAY)
    x_test_new[k, :, :, 3] = x_test_temp

x_img_train=x_train_new
x_img_test=x_test_new


print("train data:",'images:',x_img_train.shape,
      " labels:",y_label_train.shape)
print("test  data:",'images:',x_img_test.shape ,
      " labels:",y_label_test.shape)
# 归一化
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0



# One-Hot Encoding
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_test_OneHot.shape

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
# 搭面包架子
model = Sequential()
# 加面包：卷积层1 和 池化层1
model.add(Conv2D(filters=6,kernel_size=(3,3),
                 input_shape=(32, 32, 4),
                 activation='relu',
                 padding='same'))
model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2, 2))) # 16* 16

# 加面包：卷积层2 和 池化层2
model.add(Conv2D(filters=16, kernel_size=(5, 5),
                 activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2))) # 8 * 8
#Step3	建立神經網路(平坦層、隱藏層、輸出層)
model.add(Flatten()) # FC1,64个8*8转化为1维向量
model.add(Dropout(rate=0.25))

model.add(Dense(120, activation='relu')) # FC2 1024
model.add(Dropout(rate=0.25))

model.add(Dense(84, activation='relu')) # FC2 1024
model.add(Dropout(rate=0.25))

model.add(Dense(10, activation='softmax')) # Output 10
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.2,
                        epochs=10, batch_size=128, verbose=1)




import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')

show_train_history('loss','val_loss')

x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0

scores = model.evaluate(x_img_test_normalize,
                        y_label_test_OneHot,verbose=0)
scores[1]

prediction=model.predict_classes(x_img_test_normalize)
prediction[:10]

label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
import matplotlib.pyplot as plt


def plot_images_labels_prediction(images, labels, prediction,
                                  idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')

        title = str(i) + ',' + label_dict[labels[i][0]]
        if len(prediction) > 0:
            title += '=>' + label_dict[prediction[i]]

        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()

plot_images_labels_prediction(x_img_test,y_label_test,prediction,0,10)


Predicted_Probability = model.predict(x_img_test_normalize)
label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
def show_Predicted_Probability(y, prediction, x_img, Predicted_Probability, i):
    print('label:', label_dict[y[i][0]], 'predict:', label_dict[prediction[i]])
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(x_img_test[i], (32, 32, 4)))
    plt.show()
    for j in range(10):
        print(label_dict[j] + ' Probability:%1.9f' % (Predicted_Probability[i][j]))



show_Predicted_Probability(y_label_test,prediction,x_img_test,Predicted_Probability,0)

print(y_label_test)
print(y_label_test.reshape(-1))# 转化为1维数组
import pandas as pd
print(label_dict)
pd.crosstab(y_label_test.reshape(-1),prediction,rownames=['label'],colnames=['predict'])
