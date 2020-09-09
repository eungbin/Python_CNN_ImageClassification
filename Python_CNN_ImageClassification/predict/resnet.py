from keras import models, layers
from keras.layers import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, \
    ZeroPadding2D, Add
from PIL import Image
import glob
import cv2

import os
import matplotlib.pyplot as plt
import numpy as np
import math
''' #이미지 크기 조절 부분
i = 1
images = glob.glob('hand/test_720/1/*.jpg')
print(images)
for fname in images:
    image = Image.open(fname)
    resize_image = image.resize((240,240))
    resize_image.save('hand/test_240/1/1_' + str(i) +'.jpg')
    i = i + 1
'''
#이미지를 잘라내는 부분
'''i = 1
while i < 38:
    src = cv2.imread('hand/images/origin/3/3_' + str(i) + '.jpg', cv2.IMREAD_COLOR)
    dst = src.copy()
    dst = src[110:610, 110:610]
    cv2.imwrite('hand/images/remake/3/3_' + str(i) + '.jpg', dst)
    i = i + 1'''

train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = os.path.join('hand/train_cut_110_500')
val_dir = os.path.join('hand/test_cut_110_500')

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=4, target_size=(500, 500), color_mode='rgb')
val_generator = val_datagen.flow_from_directory(val_dir, batch_size=3, target_size=(500, 500), color_mode='rgb')

# number of classes
K = 3

input_tensor = Input(shape=(500, 500, 3), dtype='float32', name='input')


def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    return x


def conv2_layer(x):
    x = MaxPooling2D((3, 3), 2)(x)

    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(16, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(shortcut)#########
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(16, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)  #########
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv3_layer(x):
    shortcut = x

    for i in range(4):
        if (i == 0):
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)  #########
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(128, (1, 1), strides=(1, 1), padding='same')(shortcut)#########
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)  #########
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv4_layer(x):
    shortcut = x

    for i in range(6):
        if (i == 0):
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)  #########
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='same')(shortcut)#########
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)  #########
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


def conv5_layer(x):
    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (1, 1), strides=(1, 1), padding='same')(x)  #########
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x) # origin:2048
            shortcut = Conv2D(512, (1, 1), strides=(1, 1), padding='same')(shortcut)#########
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut) #origin:2048
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)#########
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(128, (1, 1), strides=(1, 1), padding='same')(x)  #########
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x) # origin 2048
            x = BatchNormalization()(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)

            shortcut = x

    return x


x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)

x = GlobalAveragePooling2D()(x)
output_tensor = Dense(K, activation='softmax')(x)

resnet50 = Model(input_tensor, output_tensor)
resnet50.summary()

resnet50.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

resnet50.fit_generator(train_generator, steps_per_epoch=24, epochs=50, validation_data=val_generator, validation_steps=5)

resnet50.save('resnetTest.h5')