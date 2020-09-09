'''
첫번째 모델 파일
hand.py
현재까지 전신사진 테스트 결과 정확도는 첫번째 모델이 더 높음.
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

np.random.seed(3)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('hand/train_240', target_size=(240, 240), batch_size=4, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('hand/test_240', target_size=(240, 240), batch_size=3, class_mode='categorical')

model = Sequential()
model.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu', input_shape=(240, 240, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, kernel_size=(1, 1), strides=(1, 1), activation= 'relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation= 'relu'))
model.add(Conv2D(16, kernel_size=(1, 1), strides=(1, 1), activation= 'relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation= 'relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation= 'relu'))
model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation= 'relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=24, epochs=30, validation_data=test_generator, validation_steps=5)

model.save('test_720.h5')     #first_hand_test.h5