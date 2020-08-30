import pickle
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model


def network():
    """
    Define the network
    :return:
    """
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('softmax'))

    return model


def train(file_path, model):

    x_,y_ = pickle.load( open(file_path, "rb" ) )
    random_state = 130
    X_train, x_validation, y_train, y_validation = train_test_split(x_, y_, train_size = 0.80,
                                                                    test_size = 0.2,
                                                                    random_state = random_state)
    # preprocess data
    X_normalized = np.array(X_train / 255.0 - 0.5 )
    label_binarizer = LabelBinarizer()
    y_one_hot = label_binarizer.fit_transform(y_train)

    model.summary()
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_normalized, y_one_hot, nb_epoch=20, validation_split=0.2)

    model.save('model.h5')
    return history


