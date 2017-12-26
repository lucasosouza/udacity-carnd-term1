# neuralnet.py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

# dropped
# from keras.utils import np_utils


def baseline_model():  

    model = Sequential()
    
    # normalization
    model.add(BatchNormalization(input_shape=(16, 32, 3)))

    # 1
    model.add(Convolution2D(100, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 2
    model.add(Convolution2D(150, 4, 4))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # 3
    model.add(Flatten())
    model.add(Dense(300))
    # model.add(Dropout(.5))
    model.add(Activation('tanh'))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dense(1))

    
    # compile
    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model
