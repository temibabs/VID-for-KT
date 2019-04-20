import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical


class Data(object):
    def __init__(self, data='cifar-10'):
        if data == 'cifar-10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        else:
            self.x_train = np.random.random((1000, 20))
            self.y_train = to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
            self.x_test = np.random.random((100, 20))
            self.y_test = to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

        print("INITIAL")
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_test.shape)
        print(self.y_test.shape)

    def prepare(self):
        self.x_train, self.x_test = self.x_train.astype('float32') / 255, self.x_test.astype('float32') / 255
        self.y_train, self.y_test = to_categorical(self.y_train), to_categorical(self.y_test)
