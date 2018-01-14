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
from load_steering_model import define_model, define_linear_model, define_cdropout_model

def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X, Y, _ = tup
    Y = Y[:, -1]
    if X.shape[1] == 1:  # no temporal context
      X = X[:, -1]
    yield X, Y

def get_conv_layers(model):
    return [i for i in range(len(model.layers)) if 'conv' in model.layers[i].name]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
    parser.add_argument('--batch', type=int, default=256, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
    parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
    parser.add_argument('--use_previous', type=bool,
                        default=False, help='Start training from the pre-trained weights')
    parser.add_argument('--use_model', default='default', help='Train a linear model as a sanity check')
    parser.set_defaults(skipvalidate=False)
    parser.set_defaults(loadweights=False)
    args = parser.parse_args()
    if args.use_model == 'linear':
        args.use_previous=False
        model = define_linear_model()
    elif args.use_model == 'cdropout':
        model = define_cdropout_model()

    else:
        model = define_model()
    if os.path.exists('./outputs/steering_model') and args.use_previous:
        if args.use_model = 'default':
            model.load_weights('./outputs/steering_model/steering_angle.keras')
        elif args.use_model = 'cdropout':
            prev_model = define_model()
            prev_model.load_weights('./outputs/steering_model/steering_angle.keras')
            for i in get_conv_layers(model):
              model.layers[i].set_weights(
                prev_model.layers[i].get_weights()
                )

              
    model.fit_generator(
        gen(20, args.host, port=args.port),
        steps_per_epoch=10000 / args.batch, 
        epochs=args.epoch,
        validation_data=gen(20, args.host, port=args.val_port),
        validation_steps=1000 / args.batch
    )
    print("Saving model weights and configuration file.")

    if not os.path.exists("./outputs/steering_model_new"):
        os.makedirs("./outputs/steering_model_new")

    model.save_weights("./outputs/steering_model_new/steering_angle_" + args.use_model + ".keras", True)
