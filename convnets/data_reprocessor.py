#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *


import sys
import os
import time
import matplotlib.gridspec as gridspec
import math


#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable


import tf.transformations as tft
from smpl.smpl_webuser.serialization import load_model

# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


import cPickle as pkl
import random
from scipy import ndimage

np.set_printoptions(threshold='nan')

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 74#73 #taxels
NUMOFTAXELS_Y = 27#30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES = 10
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)


def visualize_pressure_map(p_map, targets_raw=None, scores_raw=None, p_map_val=None, targets_val=None, scores_val=None, block=False, title=' '):
    # p_map_val[0, :, :] = p_map[1, : ,:]

    try:
        p_map = p_map[0, :, :]  # select the original image matrix from the intermediate amplifier matrix and the height matrix
    except:
        pass

    plt.close()
    plt.pause(0.0001)

    fig = plt.figure()
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    # mngr.window.setGeometry(50, 100, 840, 705)

    plt.pause(0.0001)

    # set options
    if p_map_val is not None:
        try:
            p_map_val = p_map_val[0, :, :]  # select the original image matrix from the intermediate amplifier matrix and the height matrix
        except:
            pass
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        xlim = [-10.0, 37.0]
        ylim = [74.0, -10.0]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax1.set_facecolor('cyan')
        ax2.set_facecolor('cyan')
        ax1.imshow(p_map, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax2.imshow(p_map_val, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax1.set_title('Training Sample \n Targets and Estimates')
        ax2.set_title('Validation Sample \n Targets and Estimates')


    else:
        ax1 = fig.add_subplot(1, 1, 1)
        xlim = [-2.0, 49.0]
        ylim = [86.0, -2.0]
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_facecolor('cyan')
        ax1.imshow(p_map, interpolation='nearest', cmap=
        plt.cm.jet, origin='upper', vmin=0, vmax=100)
        ax1.set_title('Validation Sample \n Targets and Estimates \n' + title)

    # Visualize targets of training set
    if targets_raw is not None:
        if len(np.shape(targets_raw)) == 1:
            targets_raw = np.reshape(targets_raw, (len(targets_raw) / 3, 3))
        target_coord = targets_raw[:, :2] / INTER_SENSOR_DISTANCE
        target_coord[:, 0] -= 10
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
        target_coord[:, 1] *= -1.0
        ax1.plot(target_coord[:, 0], target_coord[:, 1], marker='o', linestyle='None', markerfacecolor='green',
                 markeredgecolor='black', ms=8)
    plt.pause(0.0001)

    # Visualize estimated from training set
    if scores_raw is not None:
        if len(np.shape(scores_raw)) == 1:
            scores_raw = np.reshape(scores_raw, (len(scores_raw) / 3, 3))
        target_coord = scores_raw[:, :2] / INTER_SENSOR_DISTANCE
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
        target_coord[:, 1] *= -1.0
        ax1.plot(target_coord[:, 0], target_coord[:, 1], marker='o', linestyle='None', markerfacecolor='yellow',
                 markeredgecolor='black', ms=8)
    plt.pause(0.0001)

    # Visualize targets of validation set
    if targets_val is not None:
        if len(np.shape(targets_val)) == 1:
            targets_val = np.reshape(targets_val, (len(targets_val) / 3, 3))
        target_coord = targets_val[:, :2] / INTER_SENSOR_DISTANCE
        target_coord[:, 0] -= 10
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
        target_coord[:, 1] *= -1.0
        ax2.plot(target_coord[:, 0], target_coord[:, 1], marker='o', linestyle='None', markerfacecolor='green',
                 markeredgecolor='black', ms=8)
    plt.pause(0.0001)

    # Visualize estimated from training set
    if scores_val is not None:
        if len(np.shape(scores_val)) == 1:
            scores_val = np.reshape(scores_val, (len(scores_val) / 3, 3))
        target_coord = scores_val[:, :2] / INTER_SENSOR_DISTANCE
        target_coord[:, 1] -= (NUMOFTAXELS_X - 1)
        target_coord[:, 1] *= -1.0
        ax2.plot(target_coord[:, 0], target_coord[:, 1], marker='o', linestyle='None', markerfacecolor='yellow',
                 markeredgecolor='black', ms=8)
    plt.pause(0.50001)


    plt.show(block=block)

if __name__ == "__main__":

    gender = "m"
    training_data_dict = {}
    training_data_dict['markers_xyz_m'] = []
    training_data_dict['joint_angles'] = []
    training_data_dict['body_shape'] = []
    training_data_dict['bed_angle_deg'] = []
    training_data_dict['images'] = []


    model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
    m = load_model(model_path)

    resting_pose_data_list = np.load('/home/henry/data/resting_poses/resting_pose_'+gender+'_sit_95_rightside_stiff.npy')
    training_database_pmat_height_list = np.load('/home/henry/data/pmat_height/pmat_height_'+gender+'_sit_95_rightside_stiff.npy')

    for resting_pose_data_ct in range(len(resting_pose_data_list)):
        resting_pose_data = resting_pose_data_list[resting_pose_data_ct]
        pmat = training_database_pmat_height_list[0, resting_pose_data_ct]
        height = training_database_pmat_height_list[1, resting_pose_data_ct]

        capsule_angles = resting_pose_data[0].tolist()
        root_joint_pos_list = resting_pose_data[1]
        body_shape_list = resting_pose_data[2]

        m.pose[0:3] = capsule_angles[0:3]
        m.pose[3:6] = capsule_angles[6:9]
        m.pose[6:9] = capsule_angles[9:12]
        m.pose[9:12] = capsule_angles[12:15]
        m.pose[12] = capsule_angles[15]
        m.pose[15] = capsule_angles[16]
        m.pose[18:21] = capsule_angles[17:20]
        m.pose[21:24] = capsule_angles[20:23]
        m.pose[24:27] = capsule_angles[23:26]
        m.pose[27:30] = capsule_angles[26:29]
        m.pose[36:39] = capsule_angles[29:32]  # neck
        m.pose[39:42] = capsule_angles[32:35]
        m.pose[42:45] = capsule_angles[35:38]
        m.pose[45:48] = capsule_angles[38:41]  # head
        m.pose[48:51] = capsule_angles[41:44]
        m.pose[51:54] = capsule_angles[44:47]
        m.pose[55] = capsule_angles[47]
        m.pose[58] = capsule_angles[48]
        m.pose[60:63] = capsule_angles[49:52]
        m.pose[63:66] = capsule_angles[52:55]

        for shape_param in range(10):
            m.betas[shape_param] = float(body_shape_list[shape_param])

        training_data_dict['joint_angles'].append(np.asarray(m.pose))
        training_data_dict['body_shape'].append(np.asarray(m.betas))

        x_positions = np.asarray(m.J_transformed)[:, 0] - np.asarray(m.J_transformed)[0, 0]+root_joint_pos_list[0]+0.374648 + 10*INTER_SENSOR_DISTANCE
        y_positions = np.asarray(m.J_transformed)[:, 1] - np.asarray(m.J_transformed)[0, 1]+root_joint_pos_list[1]+0.927099 + 10*INTER_SENSOR_DISTANCE
        z_positions = np.asarray(m.J_transformed)[:, 2] - np.asarray(m.J_transformed)[0, 2]+height

        xyz_positions = np.transpose([x_positions, y_positions, z_positions])
        xyz_positions_shape = np.shape(xyz_positions)
        xyz_positions = xyz_positions.reshape(xyz_positions_shape[0]*xyz_positions_shape[1])
        training_data_dict['markers_xyz_m'].append(xyz_positions)

        training_data_dict['images'].append(pmat.reshape(64*27))
        training_data_dict['bed_angle_deg'].append(60.)

    pickle.dump(training_data_dict, open(os.path.join('/home/henry/data/training/train_'+gender+'_sit_95_rightside_stiff.p'), 'wb'))

    for item in training_data_dict:
        print "item name: ", item
        print np.shape(training_data_dict[item])

    test_database_file = load_pickle('/home/henry/data/testing/trainval4_150rh1_sit120rh.p')
    #training_database_file.append(filepath_prefix_qt+'/trainval8_150rh1_sit120rh.p')


    for item in test_database_file:
        print "item name: ", item
        print np.shape(test_database_file[item])


    for i in range(95):
        #print training_data_dict['markers_xyz_m'][i].reshape(24, 3)
        #print test_database_file['markers_xyz_m'][i].reshape(10, 3)

        training_pmat = np.array(training_data_dict['images'][i]).reshape(1, 64, 27)*3
        training_targets = np.array(training_data_dict['markers_xyz_m'][i])

        validate_pmat = np.array(test_database_file['images'][i]).reshape(1, 84, 47)[:, 10:74, 10:37]
        validate_targets = np.concatenate((np.array(test_database_file['markers_xyz_m'][i]), np.array(test_database_file['pseudomarkers_xyz_m'][i])), 0)


        visualize_pressure_map(training_pmat, training_targets, None, validate_pmat, validate_targets)
