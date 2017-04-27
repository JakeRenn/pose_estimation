#!/usr/bin/env python
# encoding: utf-8

from datetime import datetime
import os
import random

import tensorflow as tf
import numpy as np

import cpm
import read_data


class Config():

  #
  batch_size = 1
  initialize = False
  steps = "-1"
  gpu = '/gpu:0'
  # the number of images in your test dataset
  test_num = 0

  # image config
  points_num = 15
  fm_channel = points_num + 1
  origin_height = 212
  origin_width = 256
  img_height = 216
  img_width = 256
  is_color = False


  # feature map config
  fm_width = img_width >> 1
  fm_height = img_height >> 1
  sigma = 2.0
  alpha = 1.0
  radius = 12

  # random distortion
  degree = 15

  # solver config
  wd = 5e-4
  #wd = None
  stddev = 5e-2
  use_fp16 = False
  moving_average_decay = 0.999

  # checkpoint path and filename
  logdir = "./log/train_log/"
  params_dir = "./params/"
  load_filename = "cpm" + '-' + steps
  save_filename = "cpm"

  # iterations config
  max_iteration = 500000
  checkpoint_iters = 2000
  summary_iters = 100
  validate_iters = 2000


def main():

    config = Config()
    with tf.Graph().as_default():

        # create a reader object
        reader = read_data.PoseReader("./labels/txt/validate_annos.txt",
            "./data/train_imgs/", config)

        # create a model object
        model = cpm.CPM(config)

        # feedforward
        predicts = model.build_fc(False)

        # return the loss
        loss = model.loss()

        # Initializing operation
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep = 100)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:

            sess.run(init_op)
            model.restore(sess, saver, config.load_filename)

            # start testing
            for idx in xrange(config.test_num):
                with tf.device("/cpu:0"):
                  imgs, fm, coords, begins, filename_list = reader.get_batch(distort=False)

                # feed data into the model
                feed_dict = {
                    model.images: imgs,
                    model.coords: coords,
                    model.labels: fm
                    }
                with tf.device(config.gpu):
                    #
                    predict_coords = sess.run(predicts, feed_dict=feed_dict)




if __name__ == "__main__":
    main()
