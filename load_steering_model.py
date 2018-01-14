#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Conv2D
from keras import backend as K
import keras
from concrete_dropout import ConcreteDropout


def define_model(time_len=1):
    """ 
    This is the model definition for the pre-trained comma-ai
    steering prediction model. It has been changed a little to be up to
    date with the newer keras API. I couldn't get loading it from the
    JSON to work properly, but this should be exactly equivalent in any
    case
    """

    K.set_image_data_format('channels_first')
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5),strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def define_cdropout_model():
    """
    This is a version of the model used by the comma ai people, but with dropout
    fc layers replaced by concrete dropout layers
    """
    # 7 hours of 20 hz video.
    N_DATA = 20 * 7 * 60 ** 2
    LENGTH_SCALE = 1e-2 
    MODEL_PRECISION = 1e-1
    WEIGHT_DECAY = LENGTH_SCALE ** 2 / (2 * N_DATA * MODEL_PRECISION)
    WEIGHT_REGULARIZER =  LENGTH_SCALE ** 2 / (N_DATA * MODEL_PRECISION)
    DROPOUT_REGULARIZER = 1 / (MODEL_PRECISION * N_DATA)

    K.set_image_data_format('channels_first')
    ch, row, col = 3, 160, 320  # camera format
  
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(ch, row, col),
              output_shape=(ch, row, col)))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())
    model.add(Conv2D(32, (5, 5),strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(ELU())
    model.add(ConcreteDropout(Dense(512, activation='elu'),
                              weight_regularizer=WEIGHT_REGULARIZER,
                              dropout_regularizer=DROPOUT_REGULARIZER))
    model.add(ELU())
    model.add(ConcreteDropout(Dense(1),
                              weight_regularizer=WEIGHT_REGULARIZER,
                              dropout_regularizer=DROPOUT_REGULARIZER))
  
    model.compile(optimizer="adam", loss="mse")
  
    return model
  
 
def define_linear_model(time_len=1):
    """
    Sanity check; just do least-squares regression and see what performance we get
    """
    K.set_image_data_format('channels_first')
    ch, row, col = 3, 160, 320  # camera format
    weight_decay = 0.0001
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
    model.add(Flatten())   
    model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(weight_decay)))
    return model
