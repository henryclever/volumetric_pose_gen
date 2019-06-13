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


#import tf.transformations as tft
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

np.set_printoptions(threshold=sys.maxsize)

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



def fix_angles_in_dataset():

    filename = '/home/henry/data/training/train_f_lay_3850_none_stiff.p'

    laying_data = load_pickle(filename)
    print np.shape(laying_data['bed_angle_deg'])
    print laying_data['bed_angle_deg']

    laying_data['bed_angle_deg'] = list(np.array(laying_data['bed_angle_deg'])*0.)

    print laying_data['bed_angle_deg']

    pickle.dump(laying_data, open(os.path.join(filename), 'wb'))





if __name__ == "__main__":

    #fix_angles_in_dataset()

    import dart_skel_sim


    #gender = "f"
    #posture = "lay"
    #stiffness = "leftside"
    #num_resting_poses = 3722

    #all_data_names = [["f", "sit", "upperbody", 1000, 1302]]
    all_data_names = [#["f", "lay", "lowerbody", 2000, 2018],
                      #["f", "lay", "upperbody", 2000, 2051]
                      #["f", "lay", "leftside", 2000, 2034],
                      #["f", "lay", "rightside", 2000, 2052],
                      #["f", "lay", "none", 2000, 2029],
                      #["f", "sit", "lowerbody", 1000, 1126],
                      ["f", "sit", "upperbody", 1000, 1168],
                      ["f", "sit", "leftside", 1000, 1133]]
                      #["f", "sit", "rightside", 1000, 1121],
                      #["f", "sit", "none", 1000, 1113]]
                      #["m", "lay", "lowerbody", 2000, 2016],
                      #["m", "lay", "upperbody", 2000, 2024],
                      #["m", "lay", "leftside", 2000, 2015],
                      #["m", "lay", "rightside", 2000, 2016],
                      #["m", "lay", "none", 2000, 2008],
                      #["m", "sit", "lowerbody", 1000, 1166],
                      #["m", "sit", "upperbody", 1000, 1181],
                      #["m", "sit", "leftside", 1000, 1183],
                      #["m", "sit", "rightside", 1000, 1161],
                      #["m", "sit", "none", 1000, 1155]]


    for gpsn in all_data_names:
        gender = gpsn[0]
        posture = gpsn[1]
        stiffness = gpsn[2]
        num_resting_poses = gpsn[3]
        num_resting_poses_tried = gpsn[4]

        training_data_dict = {}
        training_data_dict['markers_xyz_m'] = []
        training_data_dict['root_xyz_shift'] = []
        training_data_dict['joint_angles'] = []
        training_data_dict['body_shape'] = []
        training_data_dict['body_mass'] = []
        training_data_dict['body_height'] = []
        training_data_dict['bed_angle_deg'] = []
        training_data_dict['images'] = []

        #training_data_dict['v_template'] = []
        #training_data_dict['shapedirs'] = []

        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)

        #m.betas = [0.29387146, -2.98438001, 2.66495728, -2.42446828, 1.01479387, -1.46582723, 0.14909998,
        #                0.94134736, -1.21646106, 0.69998872]
        #m.pose = [5.24451196e-01, 6.55976217e-03, 2.42505297e-02, -8.25246215e-01,
        #               4.93589081e-02, 3.45132113e-01, -1.18632483e+00, -7.60989010e-01,
        #               -3.11952025e-01, 7.02403903e-01, -7.65096024e-02, 8.40422511e-02,
        #               4.51518774e-01, 0.00000000e+00, 0.00000000e+00, 1.18769133e+00,
        #               0.00000000e+00, 0.00000000e+00, -1.77381076e-02, -2.51457542e-02,
        #               4.84700985e-02, 7.76620656e-02, 1.21850977e-02, -6.55247131e-04,
        #               2.12579081e-03, 1.08169392e-02, 4.29084338e-03, -6.83735982e-02,
        #               -2.70453375e-02, 2.89300494e-02, 0.00000000e+00, 0.00000000e+00,
        #               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        #               3.43895465e-01, -1.86359342e-02, 2.37760153e-02, -3.80625665e-01,
        #               2.97152400e-01, -5.88579237e-01, -2.91799139e-02, 4.58059739e-03,
        #               2.87017167e-01, 2.20573619e-01, 1.36713404e-02, -1.97675060e-02,
        #               1.39153078e-01, 2.51184583e-01, -7.67209888e-01, -8.91691685e-01,
        #               4.42599386e-01, 6.64221123e-02, 0.00000000e+00, -8.24406683e-01,
        #               0.00000000e+00, 0.00000000e+00, 1.30042803e+00, 0.00000000e+00,
        #               7.26327822e-02, -6.36567548e-02, -8.66599474e-03, -2.31130864e-03,
        #               3.80610838e-03, 4.34149348e-04, 0.00000000e+00, 0.00000000e+00,
        #               0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]

        #print m.betas
        #print m.pose
        #print "J x trans", m.J_transformed[:, 0]

        resting_pose_data_list = np.load('/media/henry/multimodal_data_2/data/resting_poses/side_up/resting_pose_'
                                        +gender+'_'+posture+'_'+str(num_resting_poses)+'_of_'+str(num_resting_poses_tried)+'_'+stiffness+'_stiff.npy',
                                        allow_pickle = True)
        training_database_pmat_height_list = np.load('/media/henry/multimodal_data_2/data/pmat_height/side_up/pmat_height_'
                                        +gender+'_'+posture+'_'+str(num_resting_poses)+'_of_'+str(num_resting_poses_tried)+'_'+stiffness+'_stiff.npy',
                                        allow_pickle = True)

        print len(resting_pose_data_list), len(training_database_pmat_height_list[0])

        for resting_pose_data_ct in range(len(resting_pose_data_list)):
            resting_pose_data = resting_pose_data_list[resting_pose_data_ct]
            pmat = training_database_pmat_height_list[0, resting_pose_data_ct]
            height = training_database_pmat_height_list[1, resting_pose_data_ct]
            capsule_angles = resting_pose_data[0].tolist()
            root_joint_pos_list = resting_pose_data[1]
            body_shape_list = resting_pose_data[2]

            #print "shape", body_shape_list

            print np.shape(resting_pose_data), np.shape(pmat), np.shape(height), np.shape(capsule_angles), np.shape(root_joint_pos_list), np.shape(body_shape_list)

            for shape_param in range(10):
                m.betas[shape_param] = float(body_shape_list[shape_param])



            m.pose[:] = np.random.rand(m.pose.size) * 0.
            dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture=posture, stiffness=None, check_only_distal = True, filepath_prefix='/home/henry', add_floor = False)
            #print self.m.pose
            volumes = dss.getCapsuleVolumes(mm_resolution = 1.)[2]
            dss.world.reset()
            dss.world.destroy()
            dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture=posture, stiffness=None, check_only_distal = True, filepath_prefix='/home/henry', add_floor = False, volume = volumes)
            #print self.m.pose
            training_data_dict['body_mass'].append(dss.body_mass)#, "person's mass"
            dss.world.reset()
            dss.world.destroy()
            training_data_dict['body_height'].append(np.abs(np.min(m.r[:, 1])-np.max(m.r[:, 1])))

            print training_data_dict['body_mass'][-1]*2.20462, 'MASS, lbs'
            print training_data_dict['body_height'][-1]*3.28084, 'HEIGHT, ft'




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


            training_data_dict['joint_angles'].append(np.array(m.pose).astype(float))
            training_data_dict['body_shape'].append(np.array(m.betas).astype(float))
            #print "dict", training_data_dict['body_shape'][-1]

            #training_data_dict['v_template'].append(np.asarray(m.v_template))
            #training_data_dict['shapedirs'].append(np.asarray(m.shapedirs))

            #print np.sum(np.array(m.v_template))
            #print np.sum(np.array(m.shapedirs))
            #print np.sum(np.zeros((np.shape(np.array(m.J_regressor)))) + np.array(m.J_regressor))

            root_shift_x = root_joint_pos_list[0]+0.374648 + 10*INTER_SENSOR_DISTANCE
            root_shift_y = root_joint_pos_list[1]+0.927099 + 10*INTER_SENSOR_DISTANCE
            #root_shift_z = height
            root_shift_z = root_joint_pos_list[2]-0.15
            print height, root_shift_z

            x_positions = np.asarray(m.J_transformed)[:, 0] - np.asarray(m.J_transformed)[0, 0] + root_shift_x
            y_positions = np.asarray(m.J_transformed)[:, 1] - np.asarray(m.J_transformed)[0, 1] + root_shift_y
            z_positions = np.asarray(m.J_transformed)[:, 2] - np.asarray(m.J_transformed)[0, 2] + root_shift_z

            if resting_pose_data_ct == 0:
                print m.betas
                print m.pose
                print "J x trans", m.J_transformed[:, 0]

            xyz_positions = np.transpose([x_positions, y_positions, z_positions])
            xyz_positions_shape = np.shape(xyz_positions)
            xyz_positions = xyz_positions.reshape(xyz_positions_shape[0]*xyz_positions_shape[1])
            training_data_dict['markers_xyz_m'].append(xyz_positions)
            training_data_dict['root_xyz_shift'].append([root_shift_x, root_shift_y, root_shift_z])
            training_data_dict['images'].append(pmat.reshape(64*27))
            if posture == "sit":
                training_data_dict['bed_angle_deg'].append(60.)
            elif posture == "lay":
                training_data_dict['bed_angle_deg'].append(0.)



        print training_data_dict['markers_xyz_m'][0]

        print "RECHECKING!"
        for entry in range(len(training_data_dict['markers_xyz_m'])):

            print entry, training_data_dict['markers_xyz_m'][entry][0:2], training_data_dict['body_shape'][entry][0:2], training_data_dict['joint_angles'][entry][0:2]


        pickle.dump(training_data_dict, open(os.path.join('/home/henry/data/synth/train_'+gender+'_'+posture+'_'+str(num_resting_poses)+'_of_'+str(num_resting_poses_tried)+'_'+stiffness+'_stiff.p'), 'wb'))

    for item in training_data_dict:
        print "item name: ", item
        print np.shape(training_data_dict[item])

    test_database_file = load_pickle('/home/henry/data/real/trainval4_150rh1_sit120rh.p')
    #training_database_file.append(filepath_prefix_qt+'/trainval8_150rh1_sit120rh.p')


    for item in test_database_file:
        print "item name: ", item
        print np.shape(test_database_file[item])

    '''
    for i in range(len(training_data_dict['markers_xyz_m'])):
        #print training_data_dict['markers_xyz_m'][i].reshape(24, 3)
        #print test_database_file['markers_xyz_m'][i].reshape(10, 3)

        print training_data_dict['markers_xyz_m'][i][0:2], training_data_dict['root_xyz_shift'][i]

        training_pmat = np.array(training_data_dict['images'][i]).reshape(1, 64, 27)*5.0
        training_targets = np.array(training_data_dict['markers_xyz_m'][i])

        validate_pmat = np.array(test_database_file['images'][i]).reshape(1, 84, 47)[:, 10:74, 10:37]
        validate_targets = np.concatenate((np.array(test_database_file['markers_xyz_m'][i]), np.array(test_database_file['pseudomarkers_xyz_m'][i])), 0)


        visualize_pressure_map(training_pmat, training_targets, None, validate_pmat, validate_targets)
    '''