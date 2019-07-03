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



# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class TensorPrepLib():

    def load_files_to_database(self, database_file, creation_type, verbose = False):
        # load in the training or testing files.  This may take a while.
        #print "GOT HERE!!"
        dat = None
        for some_subject in database_file:
            #print creation_type, some_subject, 'some subject'
            if creation_type in some_subject:
                dat_curr = load_pickle(some_subject)
                print some_subject, dat_curr['bed_angle_deg'][0]
                for key in dat_curr:
                    if np.array(dat_curr[key]).shape[0] != 0:
                        for inputgoalset in np.arange(len(dat_curr['markers_xyz_m'])):
                            datcurr_to_append = dat_curr[key][inputgoalset]
                            if key == 'images' and np.shape(datcurr_to_append)[0] == 3948:
                                datcurr_to_append = list(
                                    np.array(datcurr_to_append).reshape(84, 47)[10:74, 10:37].reshape(1728))
                            try:
                                dat[key].append(datcurr_to_append)
                            except:
                                try:
                                    dat[key] = []
                                    dat[key].append(datcurr_to_append)
                                except:
                                    dat = {}
                                    dat[key] = []
                                    dat[key].append(datcurr_to_append)
            else:
                pass

        if dat is not None and verbose == True:
            for key in dat:
                print 'all data keys and shape', key, np.array(dat[key]).shape
        return dat

    def prep_images(self, im_list, dat_f, dat_m, num_repeats):
        for dat in [dat_f, dat_m]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    for i in range(num_repeats):
                        im_list.append(dat['images'][entry])
        return im_list

    def prep_angles(self, ang_list, dat_f, dat_m, num_repeats):
        for dat in [dat_f, dat_m]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    for i in range(num_repeats):
                        ang_list.append([dat['bed_angle_deg'][entry]])
        return ang_list

    def prep_depth_contact(self, depth_contact_list, dat_f, dat_m, num_repeats):
        for dat in [dat_f, dat_m]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    for i in range(num_repeats):
                        depth_contact_list.append([np.fliplr(dat['mesh_depth'][entry]), np.fliplr(dat['mesh_contact'][entry]), ])
        return depth_contact_list

    def prep_depth_contact_input_est(self, depth_contact_input_est_list, dat_f, dat_m, num_repeats):
        for dat in [dat_f, dat_m]:
            if dat is not None:
                for entry in range(len(dat['images'])):
                    for i in range(num_repeats):
                        mdm_est_pos = np.copy(dat['mdm_est'][entry])
                        mdm_est_neg = np.copy(dat['mdm_est'][entry])
                        mdm_est_pos[mdm_est_pos < 0] = 0
                        mdm_est_neg[mdm_est_neg > 0] = 0
                        mdm_est_neg *= -1
                        #depth_contact_input_est_list.append([dat['mdm_est'][entry], dat['cm_est'][entry], ])
                        depth_contact_input_est_list.append([mdm_est_pos, mdm_est_neg, dat['cm_est'][entry]*100, ])
        return depth_contact_input_est_list

    def append_input_depth_contact(self, train_xa, include_pmat_contact = False,
                                   mesh_depth_contact_maps_input_est = None, include_mesh_depth_contact_input_est = False,
                                   mesh_depth_contact_maps = None, include_mesh_depth_contact = False):

        train_contact = np.copy(train_xa[:, 0:1, :, :]) #get the pmat contact map
        train_contact[train_contact > 0] = 100.


        if include_mesh_depth_contact_input_est == True:
            mesh_depth_contact_maps_input_est = np.array(mesh_depth_contact_maps_input_est)
            train_xa = np.concatenate((mesh_depth_contact_maps_input_est, train_xa), axis = 1)

        if include_pmat_contact == True:
            train_xa = np.concatenate((train_contact, train_xa), axis=1)

        if include_mesh_depth_contact == True:
            mesh_depth_contact_maps = np.array(mesh_depth_contact_maps)
            train_xa = np.concatenate((train_xa, mesh_depth_contact_maps), axis=1)

        return train_xa

    def prep_labels(self, y_flat, dat, num_repeats, z_adj, gender, is_synth, loss_vector_type, initial_angle_est):
        if gender == "f":
            g1 = 1
            g2 = 0
        elif gender == "m":
            g1 = 0
            g2 = 1
        if is_synth == True:
            s1 = 1
        else:
            s1 = 0
        z_adj_all = np.array(24 * [0.0, 0.0, z_adj*1000])
        z_adj_one = np.array(1 * [0.0, 0.0, z_adj*1000])

        if is_synth == True and loss_vector_type != 'direct':
            if dat is not None:
                for entry in range(len(dat['markers_xyz_m'])):
                    c = np.concatenate((dat['markers_xyz_m'][entry][0:72] * 1000 + z_adj_all,
                                        dat['body_shape'][entry][0:10],
                                        dat['joint_angles'][entry][0:72],
                                        dat['root_xyz_shift'][entry][0:3] + np.array([0.0, 0.0, z_adj]),
                                        [g1], [g2], [s1],
                                        [dat['body_mass'][entry]],
                                        [(dat['body_height'][entry]-1.)*100],), axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    if initial_angle_est == True:
                        c = np.concatenate((c,
                                            dat['betas_est'][entry][0:10],
                                            dat['angles_est'][entry][0:72],
                                            dat['root_xyz_est'][entry][0:3]), axis = 0)
                    for i in range(num_repeats):
                        y_flat.append(c)

        elif is_synth == True and loss_vector_type == 'direct':
            if dat is not None:
                for entry in range(len(dat['markers_xyz_m_offset'])):
                    c = np.concatenate((np.array(9 * [0]),
                                        dat['markers_xyz_m_offset'][entry][3:6] * 1000 + z_adj_one,  # TORSO
                                        # fixed_torso_markers,  # TORSO
                                        dat['markers_xyz_m_offset'][entry][21:24] * 1000 + z_adj_one,  # L KNEE
                                        dat['markers_xyz_m_offset'][entry][18:21] * 1000 + z_adj_one,  # R KNEE
                                        np.array(3 * [0]),
                                        dat['markers_xyz_m_offset'][entry][27:30] * 1000 + z_adj_one,  # L ANKLE
                                        dat['markers_xyz_m_offset'][entry][24:27] * 1000 + z_adj_one,  # R ANKLE
                                        np.array(18 * [0]),
                                        dat['markers_xyz_m_offset'][entry][0:3] * 1000 + z_adj_one,  # HEAD
                                        # fixed_head_markers,
                                        np.array(6 * [0]),
                                        dat['markers_xyz_m_offset'][entry][9:12] * 1000 + z_adj_one,  # L ELBOW
                                        dat['markers_xyz_m_offset'][entry][6:9] * 1000 + z_adj_one,  # R ELBOW
                                        dat['markers_xyz_m_offset'][entry][15:18] * 1000 + z_adj_one,  # L WRIST
                                        dat['markers_xyz_m_offset'][entry][12:15] * 1000 + z_adj_one,  # R WRIST
                                        np.array(6 * [0]),
                                        np.array(85 * [0]),
                                        [g1], [g2], [s1],
                                        [dat['body_mass'][entry]],
                                        [(dat['body_height'][entry]-1.)*100],), axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    if initial_angle_est == True:
                        c = np.concatenate((c,
                                            dat['betas_est'][entry][0:10],
                                            dat['angles_est'][entry][0:72],
                                            dat['root_xyz_est'][entry][0:3]), axis = 0)
                    for i in range(num_repeats):
                        y_flat.append(c)


        elif is_synth == False:
            if dat is not None:
                for entry in range(len(dat['markers_xyz_m'])):
                    c = np.concatenate((np.array(9 * [0]),
                                        dat['markers_xyz_m'][entry][3:6] * 1000,  # TORSO
                                        # fixed_torso_markers,  # TORSO
                                        dat['markers_xyz_m'][entry][21:24] * 1000,  # L KNEE
                                        dat['markers_xyz_m'][entry][18:21] * 1000,  # R KNEE
                                        np.array(3 * [0]),
                                        dat['markers_xyz_m'][entry][27:30] * 1000,  # L ANKLE
                                        dat['markers_xyz_m'][entry][24:27] * 1000,  # R ANKLE
                                        np.array(18 * [0]),
                                        dat['markers_xyz_m'][entry][0:3] * 1000,  # HEAD
                                        # fixed_head_markers,
                                        np.array(6 * [0]),
                                        dat['markers_xyz_m'][entry][9:12] * 1000,  # L ELBOW
                                        dat['markers_xyz_m'][entry][6:9] * 1000,  # R ELBOW
                                        dat['markers_xyz_m'][entry][15:18] * 1000,  # L WRIST
                                        dat['markers_xyz_m'][entry][12:15] * 1000,  # R WRIST
                                        np.array(6 * [0]),
                                        np.array(85 * [0]),
                                        [g1], [g2], [s1],
                                        [dat['body_mass'][entry]],
                                        [(dat['body_height'][entry]-1.)*100],), axis=0)  # [x1], [x2], [x3]: female real: 1, 0, 0.
                    if initial_angle_est == True:
                        c = np.concatenate((c,
                                            dat['betas_est'][entry][0:10],
                                            dat['angles_est'][entry][0:72],
                                            dat['root_xyz_est'][entry][0:3]), axis = 0)
                    for i in range(num_repeats):
                        y_flat.append(c)




        return y_flat

