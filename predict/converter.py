import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file('resnet.h5')
flat_data = converter.convert()

with open('resnet.tflite', 'wb') as f:
    f.write(flat_data)