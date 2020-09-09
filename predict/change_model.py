'''
두번째 모델 파일
change_model.py
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf

np.random.seed(3)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('hand/train_120', target_size=(120, 120), batch_size=4, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('hand/test_120', target_size=(120, 120), batch_size=2, class_mode='categorical')

model = Sequential();
model.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu', input_shape=(120, 120, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=24, epochs=30, validation_data=test_generator, validation_steps=6)

model.save('change_model.h5')