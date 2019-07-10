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




class SyntheticLib():

    def synthetic_scale(self, images, targets, bedangles, synth_real_switch, extra_targets = None):


        x = np.arange(-10 ,11)
        xU, xL = x + 0.5, x - 0.05
        prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        multiplier = np.random.choice(x, size=images.shape[0], p=prob)
        multiplier = (multiplier *0.020) +1

        multiplier[synth_real_switch == 1] = 1

        #plt.hist(multiplier)
        #plt.show()
        #multiplier[:] = 1.2
        if self.include_inter == True:
            multiplier[bedangles[:,0,0] > 10] = 1 #we have to cut out the scaling where the bed isn't flat

        #print targets.shape
        #print multiplier.shape
        #print multiplier
        #print targets
        #print synth_real_switch
        #multiplier[synth_real_switch == 1] = 1
        #print multiplier



        # print multiplier
        tar_mod = np.reshape(targets, (targets.shape[0], targets.shape[1] / 3, 3) ) /1000
        if extra_targets is not None:
            extra_tar_mod = np.reshape(extra_targets, (extra_targets.shape[0], extra_targets.shape[1] / 3, 3) ) /1000


        for i in np.arange(images.shape[0]):
            if self.include_inter == True:
                resized = zoom(images[i, :, :, :], multiplier[i])
                resized = np.clip(resized, 0, 100)


                rl_diff = resized.shape[2] - images[i, :, :, :].shape[2]
                ud_diff = resized.shape[1] - images[i, :, :, :].shape[1]
                l_clip = np.int(math.ceil((rl_diff) / 2))
                # r_clip = rl_diff - l_clip
                u_clip = np.int(math.ceil((ud_diff) / 2))
                # d_clip = ud_diff - u_clip

                if rl_diff < 0:  # if less than 0, we'll have to add some padding in to get back up to normal size
                    resized_adjusted = np.zeros_like(images[i, :, :, :])
                    resized_adjusted[:, -u_clip:-u_clip + resized.shape[1], -l_clip:-l_clip + resized.shape[2]] = np.copy(
                        resized)
                    images[i, :, :, :] = resized_adjusted
                    shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
                elif rl_diff > 0:  # if greater than 0, we'll have to cut the sides to get back to normal size
                    resized_adjusted = np.copy \
                        (resized[:, u_clip:u_clip + images[i, :, :, :].shape[1], l_clip:l_clip + images[i, :, :, :].shape[2]])
                    images[i, :, :, :] = resized_adjusted
                    shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
                else:
                    shift_factor_x = 0

            else:
                #multiplier[i] = 0.8
                resized = zoom(images[i ,: ,:], multiplier[i])
                resized = np.clip(resized, 0, 100)

                rl_diff = resized.shape[1] - images[i ,: ,:].shape[1]
                ud_diff = resized.shape[0] - images[i ,: ,:].shape[0]
                l_clip = np.int(math.ceil((rl_diff) / 2))
                # r_clip = rl_diff - l_clip
                u_clip = np.int(math.ceil((ud_diff) / 2))
                # d_clip = ud_diff - u_clip

                if rl_diff < 0:  # if less than 0, we'll have to add some padding in to get back up to normal size
                    resized_adjusted = np.zeros_like(images[i ,: ,:])
                    resized_adjusted[-u_clip:-u_clip + resized.shape[0], -l_clip:-l_clip + resized.shape[1]] = np.copy(resized)
                    images[i ,: ,:] = resized_adjusted
                    shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
                elif rl_diff > 0: # if greater than 0, we'll have to cut the sides to get back to normal size
                    resized_adjusted = np.copy \
                        (resized[u_clip:u_clip + images[i ,: ,:].shape[0], l_clip:l_clip + images[i ,: ,:].shape[1]])
                    images[i ,: ,:] = resized_adjusted
                    shift_factor_x = INTER_SENSOR_DISTANCE * -l_clip
                else:
                    shift_factor_x = 0

            if ud_diff < 0:
                shift_factor_y = INTER_SENSOR_DISTANCE * u_clip
            elif ud_diff > 0:
                shift_factor_y = INTER_SENSOR_DISTANCE * u_clip
            else:
                shift_factor_y = 0
            # print shift_factor_y, shift_factor_x

            resized_tar = np.copy(tar_mod[i ,: ,:])

            # resized_tar = np.reshape(resized_tar, (len(resized_tar) / 3, 3))
            # print resized_tar.shape/
            resized_tar = (resized_tar + INTER_SENSOR_DISTANCE ) * multiplier[i]

            resized_tar[:, 0] = resized_tar[:, 0] + shift_factor_x  - INTER_SENSOR_DISTANCE #- 10 * INTER_SENSOR_DISTANCE * (1 - multiplier[i])

            # resized_tar2 = np.copy(resized_tar)
            resized_tar[:, 1] = resized_tar[:, 1] + NUMOFTAXELS_X * (1 - multiplier[i]) * INTER_SENSOR_DISTANCE + shift_factor_y  - INTER_SENSOR_DISTANCE #- 10 * INTER_SENSOR_DISTANCE * (1 - multiplier[i])

            # resized_tar[7,:] = [-0.286,0,0]
            resized_tar[:, 2] = np.copy(tar_mod[i, :, 2]) * multiplier[i]

            #print resized_root, multiplier[i]

            tar_mod[i, :, :] = resized_tar

            if extra_targets is not None:
                resized_extra_tar = np.copy(extra_tar_mod[i ,: ,:])

                # resized_tar = np.reshape(resized_tar, (len(resized_tar) / 3, 3))
                # print resized_tar.shape/
                resized_extra_tar = (resized_extra_tar + INTER_SENSOR_DISTANCE ) * multiplier[i]

                resized_extra_tar[:, 0] = resized_extra_tar[:, 0] + shift_factor_x  - INTER_SENSOR_DISTANCE #- 10 * INTER_SENSOR_DISTANCE * (1 - multiplier[i])

                # resized_tar2 = np.copy(resized_tar)
                resized_extra_tar[:, 1] = resized_extra_tar[:, 1] + NUMOFTAXELS_X * (1 - multiplier[i]) * INTER_SENSOR_DISTANCE + shift_factor_y  - INTER_SENSOR_DISTANCE #- 10 * INTER_SENSOR_DISTANCE * (1 - multiplier[i])

                # resized_tar[7,:] = [-0.286,0,0]
                resized_extra_tar[:, 2] = np.copy(extra_tar_mod[i, :, 2]) * multiplier[i]

                #print resized_root, multiplier[i]

                extra_tar_mod[i, :, :] = resized_extra_tar




        #print root_mod
        #print tar_mod[0,:,:], 'post'
        targets = np.reshape(tar_mod, (targets.shape[0], targets.shape[1])) * 1000

        if extra_targets is not None:
           extra_targets = np.reshape(extra_tar_mod, (targets.shape[0], targets.shape[1])) * 1000

        return images, targets, extra_targets


    def synthetic_shiftxy(self, images, targets, bedangles, synth_real_switch, extra_targets = None):

        #use bed angles to keep it from shifting in the x and y directions

        x = np.arange(-15, 16)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=2) - ss.norm.cdf(xL, scale=2) #scale is the standard deviation using a cumulative density function
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        modified_x = np.random.choice(x, size=images.shape[0], p=prob)
        modified_x[synth_real_switch == 1] = 0

        #plt.hist(modified_x)
        #plt.show()

        y = np.arange(-5, 6)
        yU, yL = y + 0.5, y - 0.5
        prob = ss.norm.cdf(yU, scale=2) - ss.norm.cdf(yL, scale=2)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        modified_y = np.random.choice(y, size=images.shape[0], p=prob)
        modified_y[bedangles[:,0,0] > 10] = 0 #we have to cut out the vertical shifts where the bed is not flat

        modified_y[synth_real_switch == 1] = 0

        #plt.hist(modified_y)
        #plt.show()

        tar_mod = np.reshape(targets, (targets.shape[0], targets.shape[1] / 3, 3))
        if extra_targets is not None:
            extra_tar_mod = np.reshape(extra_targets, (extra_targets.shape[0], extra_targets.shape[1] / 3, 3))

        # print images[0,30:34,10:14]
        # print modified_x[0]
        for i in np.arange(images.shape[0]):
            if self.include_inter == True:
                if modified_x[i] > 0:
                    images[i, :, :, modified_x[i]:] = images[i, :, :, 0:-modified_x[i]]
                    images[i, :, :, 0:modified_x[i]] = 0
                elif modified_x[i] < 0:
                    images[i, :, :, 0:modified_x[i]] = images[i, :, :, -modified_x[i]:]
                    images[i, :, :, modified_x[i]:] = 0

                if modified_y[i] > 0:
                    images[i, :, modified_y[i]:, :] = images[i, :, 0:-modified_y[i], :]
                    images[i, :, 0:modified_y[i], :] = 0
                elif modified_y[i] < 0:
                    images[i, :, 0:modified_y[i], :] = images[i, :, -modified_y[i]:, :]
                    images[i, :, modified_y[i]:, :] = 0

            else:
                if modified_x[i] > 0:
                    images[i, :, modified_x[i]:] = images[i, :, 0:-modified_x[i]]
                    images[i, :, 0:modified_x[i]] = 0
                elif modified_x[i] < 0:
                    images[i, :, 0:modified_x[i]] = images[i, :, -modified_x[i]:]
                    images[i, :, modified_x[i]:] = 0

                if modified_y[i] > 0:
                    images[i, modified_y[i]:, :] = images[i, 0:-modified_y[i], :]
                    images[i, 0:modified_y[i], :] = 0
                elif modified_y[i] < 0:
                    images[i, 0:modified_y[i], :] = images[i, -modified_y[i]:, :]
                    images[i, modified_y[i]:, :] = 0

            tar_mod[i, :, 0] += modified_x[i] * INTER_SENSOR_DISTANCE * 1000
            tar_mod[i, :, 1] -= modified_y[i] * INTER_SENSOR_DISTANCE * 1000
            if extra_targets is not None:
                extra_tar_mod[i, :, 0] += modified_x[i] * INTER_SENSOR_DISTANCE * 1000
                extra_tar_mod[i, :, 1] -= modified_y[i] * INTER_SENSOR_DISTANCE * 1000

        # print images[0, 30:34, 10:14]
        targets = np.reshape(tar_mod, (targets.shape[0], targets.shape[1]))
        if extra_targets is not None:
            extra_targets = np.reshape(extra_tar_mod, (extra_targets.shape[0], extra_targets.shape[1]))


        return images, targets, extra_targets


    def synthetic_fliplr(self, images, targets, synth_real_switch, extra_targets = None, extra_smpl_angles = None):
        coin = np.random.randint(2, size=images.shape[0])
        coin[synth_real_switch == 1] = 0

        #coin[0] = 1
        #coin[1] = 0

        modified = coin
        original = 1 - coin

        if self.include_inter == True:
            im_orig = np.multiply(images, original[:, np.newaxis, np.newaxis, np.newaxis])
            im_mod = np.multiply(images, modified[:, np.newaxis, np.newaxis, np.newaxis])
            # flip the x axis on all the modified pressure mat images
            im_mod = im_mod[:, :, :, ::-1]
        else:
            im_orig = np.multiply(images, original[:, np.newaxis, np.newaxis])
            im_mod = np.multiply(images, modified[:, np.newaxis, np.newaxis])
            # flip the x axis on all the modified pressure mat images
            im_mod = im_mod[:, :, ::-1]


        tar_orig = np.multiply(targets, original[:, np.newaxis])
        tar_mod = np.multiply(targets, modified[:, np.newaxis])
        # change the left and right tags on the target in the z, flip x target left to right
        tar_mod = np.reshape(tar_mod, (tar_mod.shape[0], tar_mod.shape[1] / 3, 3))

        # flip the x left to right
        tar_mod[:, :, 0] = (tar_mod[:, :, 0] - 657.8) * -1 + 657.8


        # swap in the z
        dummy = zeros((tar_mod.shape))
        dummy[:, [4, 7, 18, 20], :] = tar_mod[:, [4, 7, 18, 20], :]
        tar_mod[:, [4, 7, 18, 20], :] = tar_mod[:, [5, 8, 19, 21], :]
        tar_mod[:, [5, 8, 19, 21], :] = dummy[:, [4, 7, 18, 20], :]


        tar_mod = np.reshape(tar_mod, (tar_mod.shape[0], tar_orig.shape[1]))
        tar_mod = np.multiply(tar_mod, modified[:, np.newaxis])

        images = im_orig + im_mod
        targets = tar_orig + tar_mod

        if extra_smpl_angles is not None:
            X_smpl_angles_orig = np.multiply(extra_smpl_angles, original[:, np.newaxis])
            X_smpl_angles_mod = np.multiply(extra_smpl_angles, modified[:, np.newaxis])
            X_smpl_angles_mod = X_smpl_angles_mod.reshape(X_smpl_angles_mod.shape[0], X_smpl_angles_mod.shape[1] / 3, 3)

            dummy_smpl = zeros((X_smpl_angles_mod.shape))
            dummy_smpl[:, [1, 4, 7, 10, 13, 16, 18, 20, 22], :] = X_smpl_angles_mod[:,
                                                                  [1, 4, 7, 10, 13, 16, 18, 20, 22], :]
            X_smpl_angles_mod[:, [1, 4, 7, 10, 13, 16, 18, 20, 22], :] = X_smpl_angles_mod[:,
                                                                         [2, 5, 8, 11, 14, 17, 19, 21, 23], :]
            X_smpl_angles_mod[:, [2, 5, 8, 11, 14, 17, 19, 21, 23], :] = dummy_smpl[:,
                                                                         [1, 4, 7, 10, 13, 16, 18, 20, 22], :]

            X_smpl_angles_mod[:, :, 1:3] *= -1

            X_smpl_angles_mod = X_smpl_angles_mod.reshape(X_smpl_angles_mod.shape[0],
                                                          X_smpl_angles_mod.shape[1] * X_smpl_angles_mod.shape[2])

            extra_smpl_angles = X_smpl_angles_orig + X_smpl_angles_mod

        if extra_targets is not None:
            extra_tar_orig = np.multiply(extra_targets, original[:, np.newaxis])
            extra_tar_mod = np.multiply(extra_targets, modified[:, np.newaxis])

            #print pcons.shape, 'pconshape'

            # change the left and right tags on the target in the z, flip x target left to right
            extra_tar_mod = np.reshape(extra_tar_mod, (extra_tar_mod.shape[0], extra_tar_mod.shape[1] / 3, 3))

            # flip the x left to right
            extra_tar_mod[:, :, 0] = (extra_tar_mod[:, :, 0] - 657.8) * -1 + 657.8

            #print tar_mod.shape
            #print tar_mod[0, :, :]

            # swap in the z
            dummy = zeros((extra_tar_mod.shape))

            if extra_tar_mod.shape[1] > 1:
                dummy[:, [4, 7, 18, 20], :] = extra_tar_mod[:, [4, 7, 18, 20], :]
                extra_tar_mod[:, [4, 7, 18, 20], :] = extra_tar_mod[:, [5, 8, 19, 21], :]
                extra_tar_mod[:, [5, 8, 19, 21], :] = dummy[:, [4, 7, 18, 20], :]



            extra_tar_mod = np.reshape(extra_tar_mod, (extra_tar_mod.shape[0], extra_tar_orig.shape[1]))
            extra_tar_mod = np.multiply(extra_tar_mod, modified[:, np.newaxis])

            extra_targets = extra_tar_orig + extra_tar_mod


        return images, targets, extra_targets, extra_smpl_angles


    def synthetic_master(self, images_tensor, targets_tensor, synth_real_switch_tensor, num_images_manip,
                         flip=False, shift=False, scale=False, include_inter = False, loss_vector_type = False,
                         extra_targets = None, extra_smpl_angles = None):
        self.loss_vector_type = loss_vector_type
        self.include_inter = include_inter
        self.t1 = time.time()
        images_tensor = torch.squeeze(images_tensor)
        # images_tensor.torch.Tensor.permute(1,2,0)
        imagesangles = images_tensor.numpy()
        targets = targets_tensor.numpy()
        if extra_targets is not None:
            extra_targets = extra_targets.numpy()*1000.
        if extra_smpl_angles is not None:
            extra_smpl_angles = extra_smpl_angles.numpy()
        synth_real_switch = synth_real_switch_tensor.numpy()

        if len(imagesangles.shape) < 4:
            imagesangles = np.expand_dims(imagesangles, 0)

        #print num_images_manip, "NUMBER OF IMAGES TO MANIPULATE"

        if include_inter == True:
            images = imagesangles[:, 0:num_images_manip, :, :]
            bedangles = imagesangles[:, num_images_manip, :, :]
        else:
            images = imagesangles[:,0:num_images_manip-1,:,:]
            bedangles = imagesangles[:,num_images_manip-1,20,20]
            #print bedangles.shape
            #print targets.shape,'targets for synthetic code'


        if scale == True:
            images, targets, extra_targets = self.synthetic_scale(images, targets, bedangles, synth_real_switch, extra_targets)
        if flip == True:
            images, targets, extra_targets, extra_smpl_angles = self.synthetic_fliplr(images, targets, synth_real_switch, extra_targets, extra_smpl_angles)
        if shift == True:
            images, targets, extra_targets = self.synthetic_shiftxy(images, targets, bedangles, synth_real_switch, extra_targets)

        # print images[0, 10:15, 20:25]

        #print targets.shape
        for joint_num in range(24):
            if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]:
                targets[:, joint_num * 3] = targets[:, joint_num * 3] * synth_real_switch
                targets[:, joint_num * 3 + 1] = targets[:, joint_num * 3 + 1] * synth_real_switch
                targets[:, joint_num * 3 + 2] = targets[:, joint_num * 3 + 2] * synth_real_switch
                if extra_targets is not None:
                    if extra_targets.shape[1] > 3:
                        extra_targets[:, joint_num * 3] = extra_targets[:, joint_num * 3] * synth_real_switch
                        extra_targets[:, joint_num * 3 + 1] = extra_targets[:, joint_num * 3 + 1] * synth_real_switch
                        extra_targets[:, joint_num * 3 + 2] = extra_targets[:, joint_num * 3 + 2] * synth_real_switch


        if include_inter == True:
            imagesangles[:,0:num_images_manip,:,:] = images
        else:
            imagesangles[:,0:num_images_manip-1,:,:] = images
        images_tensor = torch.Tensor(imagesangles)


        targets_tensor = torch.Tensor(targets)

        if extra_targets is not None: extra_targets_tensor = torch.Tensor(extra_targets/1000.)
        else: extra_targets_tensor = None
        if extra_smpl_angles is not None: extra_smpl_angles = torch.Tensor(extra_smpl_angles)
        else: extra_smpl_angles = None

        # images_tensor.torch.Tensor.permute(2, 0, 1)
        try:
            self.t2 = time.time() - self.t1
        except:
            self.t2 = 0

        # print self.t2, 'elapsed time'
        return images_tensor, targets_tensor, extra_targets_tensor, extra_smpl_angles

