'''
모델 테스트 파일
'''

from keras.models import load_model
import matplotlib.pyplot as plt
import model_test_function as mtf
from PIL import Image

model = load_model('first_hand_test.h5')
model.summary()
test1 = plt.imread('hand/real_test/120/test_image.jpg')
#test1 = test1[:,:,0]
test1 = (test1 > 125) * test1
test1 = test1.astype('float32') / 255.
print(test1)
print(len(test1))
print("test exit")
plt.imshow(test1, cmap='Greys', interpolation='nearest')
test1 = test1.reshape((1, 120, 120, 3))
print(test1)
print(len(test1))
print("test exit")
answer = mtf.test_function(model.predict_classes(test1))
print('The Answer is ', '['+answer+']')
plt.show()