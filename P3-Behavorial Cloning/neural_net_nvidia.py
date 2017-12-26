# neural_net_nvidia.py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# dropped
# from keras.utils import np_utils
# from keras.optimizes import AdamOptimizer


def baseline_model():    
    model = Sequential()

    # normalization
    model.add(BatchNormalization(input_shape=(66, 200, 3)))
        
    # 1
    model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
    model.add(Activation('relu'))

    # 2
    model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
    model.add(Activation('relu'))

    # 3
    model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
    model.add(Activation('relu'))

    # 4
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    # 5
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    # 6 fully connected
    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))
    
    # compile
    model.compile(loss='mse', optimizer='adam')
    return model
