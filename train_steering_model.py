#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Conv2D
from keras import backend as K
from server import client_generator
import keras

def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y, _ = tup
    Y = Y[:, -1]
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    yield X, Y


def get_model(time_len=1):
  K.set_image_data_format('channels_first') #this has been changed since this code was written
  ch, row, col = 3, 160, 320  # camera format

  weight_decay = 0.0001
  model = Sequential()
  model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(ch, row, col),
            output_shape=(ch, row, col)))
  model.add(Conv2D(16, (8, 8), strides=(4, 4), padding="same", kernel_regularizer=keras.regularizers.l2(weight_decay)))
  model.add(ELU())
  model.add(Conv2D(32, (5, 5),strides=(2, 2), padding="same", kernel_regularizer=keras.regularizers.l2(weight_decay)))
  model.add(ELU())
  model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same", kernel_regularizer=keras.regularizers.l2(weight_decay)))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512, kernel_regularizer=keras.regularizers.l2(weight_decay)))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(weight_decay)))


  model.compile(optimizer="adam", loss="mse")

  return model

def get_linear_model(time_len=1):
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
    model.add(Dense(1), kernel_regularizer=keras.regularizers.l2(weight_decay))
    return model

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=256, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.add_argument('--use_previous', type=bool,default=False, help='Start training from the pre-trained weights')
  parser.add_argument('--use_linear', type=bool, default=False, help='Train a linear model as a sanity check')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()
  if args.use_linear:
      args.use_previous=False
  if args.use_linear: 
    model = get_linear_model()
  else:
    model = get_model()
  if os.path.exists('./outputs/steering_model') and args.use_previous:
      model.load_weights('./outputs/steering_model/steering_angle.keras')
  model.fit_generator(
    gen(20, args.host, port=args.port),
    steps_per_epoch=10000 / args.batch, 
    epochs=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port),
    validation_steps=1000 / args.batch
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")
  if not args.use_linear:
      model.save_weights("./outputs/steering_model/steering_angle_new.keras", True)
      with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
  else:
      model.save_weights('./outputs/steering_model/steering_angle_linear.keras', True)
