from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('hand/train_240', target_size=(240, 240), batch_size=4, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('hand/test_240', target_size=(240, 240), batch_size=3, class_mode='categorical')

np.random.seed(3)
model = tf.keras.Sequential()
model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu', input_shape=(240, 240, 3)))
model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=24, epochs=30, validation_data=test_generator, validation_steps=5)

model.save('android_test.h5')