from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import numpy as np

class MLP:
    def __init__(self, n_hidden_layers, nodes, activation='relu', dataset='cifar10', n_classes=10):

        if 'cifar' in dataset:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        else:
            self.x_train = np.random.random((1000, 20))
            self.y_train = to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
            self.x_test = np.random.random((100, 20))
            self.y_test = to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

        self.n_hidden_layers = n_hidden_layers
        self.n_classes = n_classes
        self.nodes = nodes
        self.activation = activation
        print("MLP BEFORE:")
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_test.shape)
        print(self.y_test.shape)

        self.model = self.init()


    def init(self):
        self.x_train, self.x_test = self.rescale()
        self.y_train, self.y_test = self.one_hot_encode()
        self.datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        self.datagen.fit(self.x_train)

        model = Sequential()
        model.add(Flatten(input_shape=self.x_train.shape[1:]))
        for i in range(self.n_hidden_layers):
            model.add(Dense(40, activation='relu'))

        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer='adadelta',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self):
        checkpointer = ModelCheckpoint(filepath="cifar10.model.mlp.hdf5", verbose=1, save_best_only=True)
        self.model.fit_generator(self.datagen.flow(self.x_train,self.y_train,
                                                   batch_size=32),
                                 steps_per_epoch=self.x_train.shape[0] // 32,
                                 epochs=51,
                                 verbose=2,
                                 validation_data=(self.x_test, self.y_test),
                                 callbacks=[checkpointer])
    def evaluate(self):
        result = self.model.evaluate()
        print("Result: " + result)

    def rescale(self):
        return self.x_train.astype('float32') / 255, self.x_test.astype('float32') / 255

    def one_hot_encode(self):
        return to_categorical(self.y_train), to_categorical(self.y_test)

