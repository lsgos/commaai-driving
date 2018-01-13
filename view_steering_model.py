#!/usr/bin/env python
import pdb
import argparse
import sys
import numpy as np
import h5py
import pygame
import json
from keras.models import model_from_json
from matplotlib import pyplot as plt
from matplotlib import animation as ani

from load_steering_model import define_linear_model, define_model


# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
  row, col = perspective_tform(x, y)
  def rnd(x):
    # pretty sure that this used to be OK in numpy, but now only integer indices are allowed
    return int(round(x))
  
  if row >= 0 and row < img.shape[0] and\
     col >= 0 and col < img.shape[1]:
    img[rnd(row-sz):rnd(row+sz),rnd(col-sz):rnd(col+sz)] = color

def draw_path(img, path_x, path_y, color):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255)):
  path_x = np.arange(0., 50.1, 0.5)
  path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
  draw_path(img, path_x, path_y, color)

# ***** main loop *****
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Path viewer')
  parser.add_argument('model', type=str, help='Path to model weights')
  parser.add_argument('--dataset', type=str, default="2016-06-08--11-46-01", help='Dataset/video clip name')
  parser.add_argument('--use_linear', type=bool, default=False)
  args = parser.parse_args()

  if args.use_linear:
    model = define_linear_model()
  else:
    model = define_model()
  model.load_weights(args.model)


  model.compile("sgd", "mse")

  # default dataset is the validation data on the highway
  dataset = args.dataset
  # skip to highway.
  skip = 300

  log = h5py.File("dataset/log/"+dataset+".h5", "r")
  cam = h5py.File("dataset/camera/"+dataset+".h5", "r")

  print log.keys()

  
  #some statistics to print
  running_mse = 0.0
  running_avg_error = 0.0 #check for bias
  largest_abs_error = 0
  n = 1
  #the original code used pygame to draw the video, but I've found this to be unsatisfactory
  def update_image(index,pic):
    global n
    global running_mse
    global running_avg_error
    global largest_abs_error
    img = cam['X'][log['cam1_ptr'][index]].swapaxes(0,2).swapaxes(0,1)

    predicted_steers = model.predict(img[None, :, :, :].transpose(0, 3, 1, 2))[0][0]

    angle_steers = log['steering_angle'][index]
    speed_ms = log['speed'][index]
    draw_path_on(img, speed_ms, -angle_steers/10.0)
    draw_path_on(img, speed_ms, -predicted_steers/10.0, (0, 255, 0))
    pic.set_data(img)

    psa = predicted_steers / 10 #predicted steering angle
    gtsa = angle_steers / 10    #ground truth steering angle 
    
    n +=1
    running_mse += ((psa - gtsa) ** 2 - running_mse)/n
    running_avg_error += ((psa - gtsa) - running_avg_error) / n
    abs_error = np.abs(psa - gtsa)
    largest_abs_error = abs_error if abs_error > largest_abs_error else largest_abs_error
    
    if index %100 == 0:
      print "%.2f seconds elapsed" % (index /100.0)
      print("predicted steer: {}, actual steer: {}, error: {}".format(psa,
                                                                      gtsa,
                                                                      gtsa - psa))
      print("Avg MSE: {}, Avg Error: {}, Largest Abs Error: {}".format(running_mse, running_avg_error, largest_abs_error))
    return pic, 

  fig = plt.figure()
  pic = plt.imshow(cam['X'][0].swapaxes(0,2).swapaxes(0,1))

  mov = ani.FuncAnimation(fig, update_image,  range(skip * 100, log['times'].shape[0]),interval=20, fargs=(pic,))
  plt.show()
