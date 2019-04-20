from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator

from vid.data import Data

class CNN(object):
    def __init__(self):
        self.datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
        self.init()

    def init(self):
        self.cnn = Sequential()
        self.data = Data()
        self.data.prepare()

        print("AFTER")
        print(self.data.x_train.shape)
        print(self.data.y_train.shape)
        print(self.data.x_test.shape)
        print(self.data.y_test.shape)

        self.cnn.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
        self.cnn.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=2))
        self.cnn.add(Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=2))
        self.cnn.add(Conv2D(filters=128, kernel_size=4, strides=1, padding='same', activation='relu'))
        self.cnn.add(MaxPooling2D(pool_size=2))
        self.cnn.add(Dropout(0.2))
        self.cnn.add(Flatten())
        self.cnn.add(Dropout(0.2))
        self.cnn.add(Dense(512, activation='relu'))
        self.cnn.add(Dropout(0.2))
        self.cnn.add(Dense(10, activation='softmax'))
        self.cnn.summary()
        #  Compile cnn
        self.cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):
        checkpointer = ModelCheckpoint(filepath="cifar10.model.cnn.hdf5", verbose=1, save_best_only=True)
        self.cnn.fit_generator(self.datagen.flow(self.data.x_train, self.data.y_train,
                                                   batch_size=32),
                               steps_per_epoch=self.data.x_train.shape[0] // 32,
                               epochs=51,
                               verbose=2,
                               validation_data=(self.data.x_test, self.data.y_test),
                               callbacks=[checkpointer])

    def evaluate(self):
        result = self.cnn.evaluate()
        print("Result: " + result)
