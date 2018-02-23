from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten, Dropout
import mahotas
import numpy as np
from keras.utils import to_categorical
from keras.utils.training_utils import multi_gpu_model

class ZernikeMoments():
    def __init__(self, radius):
        self.radius = radius

    def describe(self, image):
        return mahotas.features.zernike_moments(image, self.radius, degree=10)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
nb_classes = len(np.unique(y_train))

print('size x_train', x_train.shape)
print('size x_test', x_test.shape)
print('Number of classes', nb_classes)
print('Shape of image', x_train[0].shape, 'Type of image', type(x_train[0]))


zm = ZernikeMoments(x_train[0].shape[0])
x_train_zm = np.asarray([zm.describe(x) for x in x_train])
x_test_zm = np.asarray([zm.describe(x) for x in x_test])
print('Shape of descrptors', x_train_zm.shape, 'Type of descriptor', type(x_train_zm))
print('Shape of descrptors', x_test_zm.shape, 'Type of descriptor', type(x_test_zm))


descriptor_shape = x_train_zm[0].shape
print('Input shape of descripor is:', descriptor_shape)


y_train_binary = to_categorical(y_train, nb_classes)
y_test_binary = to_categorical(y_test, nb_classes)


model = Sequential()
model.add(Dense(100, input_shape=descriptor_shape, activation='relu'))
Dropout(0.5)
model.add(Dense(100, input_shape=descriptor_shape, activation='relu'))
Dropout(0.5)
model.add(Dense(nb_classes, activation='softmax'))

model = multi_gpu_model(model, gpus=3)

print(model.summary())

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_zm, y_train_binary, batch_size=128, epochs=500, validation_data=(x_test_zm, y_test_binary))
