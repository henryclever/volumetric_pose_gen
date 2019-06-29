#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
#from skimage.feature import hog
#from skimage import data, color, exposure

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics
from sklearn.utils import shuffle

from kinematics_lib import KinematicsLib
from synthetic_lib import SyntheticLib
from preprocessing_lib import PreprocessingLib

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)



# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class UnpackBatchLib():
    def unpackage_batch_kin_pass(self, batch, is_training, model, CTRL_PNL):

        # 0:72: positions.
        batch.append(batch[1][:, 72:82])  # betas
        batch.append(batch[1][:, 82:154])  # angles
        batch.append(batch[1][:, 154:157])  # root pos
        batch.append(batch[1][:, 157:159])  # gender switch
        batch.append(batch[1][:, 159])  # synth vs real switch
        batch.append(batch[1][:, 160:161])  # mass, kg
        batch.append(batch[1][:, 161:162])  # height, kg

        if CTRL_PNL['regr_depth_maps'] == True and is_training == True:
            batch.append(batch[0][:, CTRL_PNL['num_input_channels_batch0'], : ,:]) #mesh depth matrix
            batch.append(batch[0][:, CTRL_PNL['num_input_channels_batch0']+1, : ,:]) #mesh contact matrix

            #cut off batch 0 so we don't have depth or contact on the input
            batch[0] = batch[0][:, 0:CTRL_PNL['num_input_channels_batch0'], :, :]

        # cut it off so batch[2] is only the xyz marker targets
        batch[1] = batch[1][:, 0:72]

        if is_training == True: #only do augmentation for real data that is in training mode
            batch[0], batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], batch[6],
                                                                    flip=True, shift=True, scale=False,
                                                                    bedangle=True,
                                                                    include_inter=CTRL_PNL['incl_inter'],
                                                                    loss_vector_type=CTRL_PNL['loss_vector_type'])

        images_up_non_tensor = PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy(), multiple=2)
        if is_training == True: #only add noise to training images
            images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(images_up_non_tensor))
        images_up = Variable(torch.Tensor(images_up_non_tensor).type(CTRL_PNL['dtype']), requires_grad=False)

        if CTRL_PNL['incl_ht_wt_channels'] == True: #make images full of stuff
            weight_input = torch.ones((images_up.size()[0], images_up.size()[2] * images_up.size()[3])).type(CTRL_PNL['dtype'])
            weight_input *= batch[7].type(CTRL_PNL['dtype'])
            weight_input = weight_input.view((images_up.size()[0], 1, images_up.size()[2], images_up.size()[3]))
            height_input = torch.ones((images_up.size()[0], images_up.size()[2] * images_up.size()[3])).type(CTRL_PNL['dtype'])
            height_input *= batch[8].type(CTRL_PNL['dtype'])
            height_input = height_input.view((images_up.size()[0], 1, images_up.size()[2], images_up.size()[3]))
            images_up = torch.cat((images_up, weight_input, height_input), 1)

        images, targets, betas = Variable(batch[0].type(CTRL_PNL['dtype']), requires_grad=False), \
                                 Variable(batch[1].type(CTRL_PNL['dtype']), requires_grad=False), \
                                 Variable(batch[2].type(CTRL_PNL['dtype']), requires_grad=False)

        angles_gt = Variable(batch[3].type(CTRL_PNL['dtype']), requires_grad=is_training)
        root_shift = Variable(batch[4].type(CTRL_PNL['dtype']), requires_grad=is_training)
        gender_switch = Variable(batch[5].type(CTRL_PNL['dtype']), requires_grad=is_training)
        synth_real_switch = Variable(batch[6].type(CTRL_PNL['dtype']), requires_grad=is_training)

        if CTRL_PNL['regr_depth_maps'] == True and is_training == True:
            mmb = batch[9].type(CTRL_PNL['dtype'])
            cmb = batch[10].type(CTRL_PNL['dtype'])
        else:
            mmb = None
            cmb = None

        scores, mmb_est, cmb_est, targets_est, betas_est = model.forward_kinematic_angles(images_up,
                                                             gender_switch,
                                                             synth_real_switch,
                                                             targets,
                                                             is_training=is_training,
                                                             betas=betas,
                                                             angles_gt=angles_gt,
                                                             root_shift=root_shift,
                                                             reg_angles=CTRL_PNL['regr_angles'],
                                                             reg_depth_maps = CTRL_PNL['regr_depth_maps'])  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.


        return scores, images, targets, targets_est, mmb, mmb_est, cmb, cmb_est


    def unpackage_batch_dir_pass(self, batch, is_training, model, CTRL_PNL):

        batch.append(batch[1][:, 159])  # synth vs real switch
        batch.append(batch[1][:, 160:161])  # mass, kg
        batch.append(batch[1][:, 161:162])  # height, kg

        # cut it off so batch[2] is only the xyz marker targets
        batch[1] = batch[1][:, 0:72]

        if is_training == True:
            batch[0], batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], batch[2],
                                                                 flip=False, shift=True, scale=False,
                                                                 bedangle=True,
                                                                 include_inter=CTRL_PNL['incl_inter'],
                                                                 loss_vector_type=CTRL_PNL['loss_vector_type'])

        images_up_non_tensor = PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy(), multiple=2)
        if is_training == True: #only add noise to training images
            images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(images_up_non_tensor))
        images_up = Variable(torch.Tensor(images_up_non_tensor).type(CTRL_PNL['dtype']), requires_grad=False)

        if CTRL_PNL['incl_ht_wt_channels'] == True: #make images full of stuff
            weight_input = torch.ones((images_up.size()[0], images_up.size()[2] * images_up.size()[3])).type(CTRL_PNL['dtype'])
            weight_input *= batch[3].type(CTRL_PNL['dtype'])
            weight_input = weight_input.view((images_up.size()[0], 1, images_up.size()[2], images_up.size()[3]))
            height_input = torch.ones((images_up.size()[0], images_up.size()[2] * images_up.size()[3])).type(CTRL_PNL['dtype'])
            height_input *= batch[4].type(CTRL_PNL['dtype'])
            height_input = height_input.view((images_up.size()[0], 1, images_up.size()[2], images_up.size()[3]))
            images_up = torch.cat((images_up, weight_input, height_input), 1)

        images, targets = Variable(batch[0].type(CTRL_PNL['dtype']), requires_grad=False), \
                          Variable(batch[1].type(CTRL_PNL['dtype']), requires_grad=False)

        scores, targets_est = model.forward_direct(images_up, targets, is_training=is_training)

        return scores, images, targets, targets_est
