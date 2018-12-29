
# coding: utf-8

# In[1]:


import makeData_improve
import cv2
import os
import numpy as np

import keras
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

path = os.getcwd()
path_HandImage = path + '/Data/HandDataImage'
path_HandInfo = path + '/Data/HandInfo.csv'

#input_shape
im_rows = 32
im_cols = 32
in_shape = (im_rows, im_cols, 3)

#分類数
num_classes = 3

makeData = makeData_improve.makeData()

x = makeData.appendX(path_HandImage)
y = makeData.appendY(path_HandInfo)


number0 = 0
number1 = 0
number2 = 0

for number in y:
    if number[0] == 1:
        number0 += 1
    if number[1] == 1:
        number1 += 1
    if number[2] == 1:
        number2 += 1

print(number0)
print(number1)
print(number2)


print(len(x))
print(len(y))


#numpyを使う
numx = np.array(x)
numy = np.array(y)

# 学習用とテスト用に分離する
x_train, x_test, y_train, y_test = train_test_split(
    numx, numy, test_size = 0.2, train_size = 0.8, shuffle = True)


# モデルを定義 
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=in_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# モデルをコンパイル
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])




# 学習を実行
hist = model.fit(x_train, y_train,
    batch_size=100, epochs=120,
    verbose=1,
    validation_data=(x_test, y_test))



# モデルを評価 
score = model.evaluate(x_test, y_test, verbose=1)
print('正解率=', score[1], 'loss=', score[0])

model.save('Hand.h5')

