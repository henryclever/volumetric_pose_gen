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



def reprocess_synth_data():
    # fix_angles_in_dataset()

    import dart_skel_sim

    # gender = "f"
    # posture = "lay"
    # stiffness = "leftside"
    # num_resting_poses = 3722

    # all_data_names = [["f", "sit", "upperbody", 1000, 1302]]
    all_data_names = [  # ["f", "lay", "lowerbody", 2000, 2018],
        # ["f", "lay", "upperbody", 2000, 2051]
        # ["f", "lay", "leftside", 2000, 2034],
        # ["f", "lay", "rightside", 2000, 2052],
        # ["f", "lay", "none", 2000, 2029],
        # ["f", "sit", "lowerbody", 1000, 1126],
        ["f", "sit", "upperbody", 1000, 1168],
        ["f", "sit", "leftside", 1000, 1133]]
    # ["f", "sit", "rightside", 1000, 1121],
    # ["f", "sit", "none", 1000, 1113]]
    # ["m", "lay", "lowerbody", 2000, 2016],
    # ["m", "lay", "upperbody", 2000, 2024],
    # ["m", "lay", "leftside", 2000, 2015],
    # ["m", "lay", "rightside", 2000, 2016],
    # ["m", "lay", "none", 2000, 2008],
    # ["m", "sit", "lowerbody", 1000, 1166],
    # ["m", "sit", "upperbody", 1000, 1181],
    # ["m", "sit", "leftside", 1000, 1183],
    # ["m", "sit", "rightside", 1000, 1161],
    # ["m", "sit", "none", 1000, 1155]]

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

        # training_data_dict['v_template'] = []
        # training_data_dict['shapedirs'] = []

        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_' + gender + '_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)

        # m.betas = [0.29387146, -2.98438001, 2.66495728, -2.42446828, 1.01479387, -1.46582723, 0.14909998,
        #                0.94134736, -1.21646106, 0.69998872]
        # m.pose = [5.24451196e-01, 6.55976217e-03, 2.42505297e-02, -8.25246215e-01,
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

        # print m.betas
        # print m.pose
        # print "J x trans", m.J_transformed[:, 0]

        resting_pose_data_list = np.load('/media/henry/multimodal_data_2/data/resting_poses/side_up/resting_pose_'
                                         + gender + '_' + posture + '_' + str(num_resting_poses) + '_of_' + str(
            num_resting_poses_tried) + '_' + stiffness + '_stiff.npy',
                                         allow_pickle=True)
        training_database_pmat_height_list = np.load(
            '/media/henry/multimodal_data_2/data/pmat_height/side_up/pmat_height_'
            + gender + '_' + posture + '_' + str(num_resting_poses) + '_of_' + str(
                num_resting_poses_tried) + '_' + stiffness + '_stiff.npy',
            allow_pickle=True)

        print len(resting_pose_data_list), len(training_database_pmat_height_list[0])

        for resting_pose_data_ct in range(len(resting_pose_data_list)):
            resting_pose_data = resting_pose_data_list[resting_pose_data_ct]
            pmat = training_database_pmat_height_list[0, resting_pose_data_ct]
            height = training_database_pmat_height_list[1, resting_pose_data_ct]
            capsule_angles = resting_pose_data[0].tolist()
            root_joint_pos_list = resting_pose_data[1]
            body_shape_list = resting_pose_data[2]

            # print "shape", body_shape_list

            print np.shape(resting_pose_data), np.shape(pmat), np.shape(height), np.shape(capsule_angles), np.shape(
                root_joint_pos_list), np.shape(body_shape_list)

            for shape_param in range(10):
                m.betas[shape_param] = float(body_shape_list[shape_param])

            m.pose[:] = np.random.rand(m.pose.size) * 0.
            dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture=posture, stiffness=None,
                                            check_only_distal=True, filepath_prefix='/home/henry', add_floor=False)
            # print self.m.pose
            volumes = dss.getCapsuleVolumes(mm_resolution=1.)[2]
            dss.world.reset()
            dss.world.destroy()
            dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture=posture, stiffness=None,
                                            check_only_distal=True, filepath_prefix='/home/henry', add_floor=False,
                                            volume=volumes)
            # print self.m.pose
            training_data_dict['body_mass'].append(dss.body_mass)  # , "person's mass"
            dss.world.reset()
            dss.world.destroy()
            training_data_dict['body_height'].append(np.abs(np.min(m.r[:, 1]) - np.max(m.r[:, 1])))

            print training_data_dict['body_mass'][-1] * 2.20462, 'MASS, lbs'
            print training_data_dict['body_height'][-1] * 3.28084, 'HEIGHT, ft'

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
            # print "dict", training_data_dict['body_shape'][-1]

            # training_data_dict['v_template'].append(np.asarray(m.v_template))
            # training_data_dict['shapedirs'].append(np.asarray(m.shapedirs))

            # print np.sum(np.array(m.v_template))
            # print np.sum(np.array(m.shapedirs))
            # print np.sum(np.zeros((np.shape(np.array(m.J_regressor)))) + np.array(m.J_regressor))

            root_shift_x = root_joint_pos_list[0] + 0.374648 + 10 * INTER_SENSOR_DISTANCE
            root_shift_y = root_joint_pos_list[1] + 0.927099 + 10 * INTER_SENSOR_DISTANCE
            # root_shift_z = height
            root_shift_z = root_joint_pos_list[2] - 0.15
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
            xyz_positions = xyz_positions.reshape(xyz_positions_shape[0] * xyz_positions_shape[1])
            training_data_dict['markers_xyz_m'].append(xyz_positions)
            training_data_dict['root_xyz_shift'].append([root_shift_x, root_shift_y, root_shift_z])
            training_data_dict['images'].append(pmat.reshape(64 * 27))
            if posture == "sit":
                training_data_dict['bed_angle_deg'].append(60.)
            elif posture == "lay":
                training_data_dict['bed_angle_deg'].append(0.)

        print training_data_dict['markers_xyz_m'][0]

        print "RECHECKING!"
        for entry in range(len(training_data_dict['markers_xyz_m'])):
            print entry, training_data_dict['markers_xyz_m'][entry][0:2], training_data_dict['body_shape'][entry][0:2], \
            training_data_dict['joint_angles'][entry][0:2]

        pickle.dump(training_data_dict, open(os.path.join(
            '/home/henry/data/synth/train_' + gender + '_' + posture + '_' + str(num_resting_poses) + '_of_' + str(
                num_resting_poses_tried) + '_' + stiffness + '_stiff.p'), 'wb'))

    for item in training_data_dict:
        print "item name: ", item
        print np.shape(training_data_dict[item])

    test_database_file = load_pickle('/home/henry/data/real/trainval4_150rh1_sit120rh.p')
    # training_database_file.append(filepath_prefix_qt+'/trainval8_150rh1_sit120rh.p')

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

def reprocess_real_data_height_wt():
    #filepath_prefix = '/home/henry/'
    filepath_prefix = '/media/henry/multimodal_data_2/'

    in_to_m = 0.0254
    lb_to_kg = 0.453592
    subj_list = [1, 2, 3, 4, 5, 6, 7, 8,
                 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    height_list = [66.5, 62.0, 67.0, 69.5, 69.0, 68.5, 65.0, 65.5,
                   70.0, 66.0, 69.0, 63.0, 68.0, 72.0, 68.0, 63.0, 62.0, 65.0]
    weight_list = [132., 166., 140., 158., 113., 189., 166., 152.,
                   174., 106., 172., 117., 152., 208., 156., 123., 101., 120.]

    for idx in range(len(height_list)):
        height_list[idx] *= in_to_m
        weight_list[idx] *= lb_to_kg

    print height_list
    print weight_list


    #for design_subj_idx in [1, 2, 3, 4, 5, 6, 7]:
    #for design_subj_idx in [3]:
    for design_subj_idx in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
        subject = subj_list[design_subj_idx]
        height = height_list[design_subj_idx]
        weight = weight_list[design_subj_idx]

        print "working on subject ", subject

        #filename = 'data/real/s'+str(subject)+'_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
        #filename = 'data/real/trainval'+str(subject)+'_150rh1_sit120rh.p'
        #filename = 'data/real/subject_'+str(subject)+'/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p'
        filename = 'data/real/subject_'+str(subject)+'/p_files/trainval_sit175rlh_sit120rll.p'

        current_data = load_pickle(filepath_prefix+filename)
        current_data['body_mass'] = []
        current_data['body_height'] = []
        for i in range(len(current_data['images'])):
            current_data['body_mass'].append(weight)
            current_data['body_height'].append(height)


        for key in current_data:
            print key, len(current_data[key])


        pickle.dump(current_data, open(os.path.join(filepath_prefix+filename), 'wb'))


def get_contact_map_from_synth():
    all_data_names = [
        ["f", "lay", "lowerbody", 2000, 2018],
        #["f", "lay", "upperbody", 2000, 2051],
        # ["f", "lay", "leftside", 2000, 2034],
        # ["f", "lay", "rightside", 2000, 2052],
        ["f", "lay", "none", 2000, 2029]]#,
        # ["f", "sit", "lowerbody", 1000, 1126],
        #["f", "sit", "upperbody", 1000, 1168],
        #["f", "sit", "leftside", 1000, 1133]]
    from visualization_lib import VisualizationLib

    filler_taxels = []
    for i in range(27):
        for j in range(64):
            filler_taxels.append([i, j, 20000])
    filler_taxels = np.array(filler_taxels)


    for gpsn in all_data_names:
        gender = gpsn[0]
        posture = gpsn[1]
        stiffness = gpsn[2]
        num_resting_poses = gpsn[3]
        num_resting_poses_tried = gpsn[4]


        # training_data_dict['v_template'] = []
        # training_data_dict['shapedirs'] = []

        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_' + gender + '_lbs_10_207_0_v1.0.0.pkl'
        m = load_model(model_path)

        training_data_dict = load_pickle('/home/henry/data/synth/train_' + gender + '_' + posture + '_' + str(num_resting_poses) + '_of_' + str(
                                num_resting_poses_tried) + '_' + stiffness + '_stiff.p')

        betas = training_data_dict['body_shape']
        pose = training_data_dict['joint_angles']
        images = training_data_dict['images']
        training_data_dict['mesh_depth'] = []
        training_data_dict['mesh_contact'] = []
        root_xyz_shift = training_data_dict['root_xyz_shift']

        for index in range(len(betas)):
            for beta_idx in range(10):
                m.betas[beta_idx] = betas[index][beta_idx]
            for pose_idx in range(72):
                m.pose[pose_idx] = pose[index][pose_idx]

            training_data_dict['images'][index] = images[index].astype(int8) #convert the original pmat to an int to save space

            pmat = np.clip(images[index].reshape(64, 27)*5., 0, 100)
            curr_root_shift = np.array(root_xyz_shift[index])

            joints = np.array(m.J_transformed) + curr_root_shift + np.array([-0.286, 0.0, 0.0])
            vertices = np.array(m.r) + curr_root_shift + np.array([-0.286, 0.0, -0.075])

            joints_taxel = joints/0.0286
            vertices_taxel = vertices/0.0286
            vertices_taxel[:, 2] *= 1000
            vertices_taxel[:, 0] *= 1.04

            time_orig = time.time()

            #joints_taxel_int = (joints_taxel).astype(int)
            vertices_taxel_int = (vertices_taxel).astype(int)


            vertices_taxel_int = np.concatenate((filler_taxels, vertices_taxel_int), axis = 0)

            vertice_sorting_method = vertices_taxel_int[:, 0]*10000000 + vertices_taxel_int[:,1]*100000 + vertices_taxel_int[:,2]
            vertices_taxel_int = vertices_taxel_int[vertice_sorting_method.argsort()]

            vertice_sorting_method_2 = vertices_taxel_int[:, 0]*100 + vertices_taxel_int[:,1]

            unique_keys, indices = np.unique(vertice_sorting_method_2, return_index=True)

            vertices_taxel_int_unique = vertices_taxel_int[indices]


            #print vertices_taxel_int_unique.shape

            vertices_taxel_int_unique = vertices_taxel_int_unique[vertices_taxel_int_unique[:, 0] < 27, :]
            vertices_taxel_int_unique = vertices_taxel_int_unique[vertices_taxel_int_unique[:, 0] >= 0, :]
            vertices_taxel_int_unique = vertices_taxel_int_unique[vertices_taxel_int_unique[:, 1] < 64, :]
            vertices_taxel_int_unique = vertices_taxel_int_unique[vertices_taxel_int_unique[:, 1] >= 0, :]
            #print vertices_taxel_int_unique

            mesh_matrix = np.flipud(vertices_taxel_int_unique[:, 2].reshape(27, 64).T).astype(float)

            mesh_matrix[mesh_matrix == 20000] = 0
            mesh_matrix *= 0.0286


            #fix holes
            abc = np.zeros((66, 29, 4))
            abc[1:65, 1:28, 0] = np.copy(mesh_matrix)
            abc[1:65, 1:28, 0][abc[1:65, 1:28, 0] > 0] = 0
            abc[1:65, 1:28, 0] = abc[0:64, 0:27, 0] + abc[1:65, 0:27, 0] + abc[2:66, 0:27, 0] + \
                                 abc[0:64, 1:28, 0] + abc[2:66, 1:28, 0] + \
                                 abc[0:64, 2:29, 0] + abc[1:65, 2:29, 0] + abc[2:66, 2:29, 0]
            abc = abc[1:65, 1:28, :]
            abc[:, :, 0] /= 8
            abc[:, :, 1] = np.copy(mesh_matrix)
            abc[:, :, 1][abc[:, :, 1] < 0] = 0
            abc[:, :, 1][abc[:, :, 1] > 0] = 1
            abc[:, :, 2] = abc[:, :, 0]*abc[:, :, 1]
            abc[:, :, 3] = np.copy(abc[:, :, 2])
            abc[:, :, 3][abc[:, :, 3] != 0] = 1.
            abc[:, :, 3] = 1-abc[:, :, 3]
            mesh_matrix = mesh_matrix*abc[:, :, 3]
            mesh_matrix += abc[:, :, 2]
            #print np.min(mesh_matrix), np.max(mesh_matrix)
            mesh_matrix = mesh_matrix.astype(int32)
            #print np.min(mesh_matrix), np.max(mesh_matrix)

            #make a contact matrix
            contact_matrix = np.copy(mesh_matrix)
            contact_matrix[contact_matrix >= 0] = 0
            contact_matrix[contact_matrix >= 0] = 0
            contact_matrix[contact_matrix < 0] = 1
            contact_matrix = contact_matrix.astype(bool)

            print time.time() - time_orig

            training_data_dict['mesh_depth'].append(mesh_matrix)
            training_data_dict['mesh_contact'].append(contact_matrix)

            print training_data_dict['images'][index].dtype
            print training_data_dict['mesh_depth'][index].dtype
            print training_data_dict['mesh_contact'][index].dtype



            #print m.J_transformed

            #VisualizationLib().visualize_pressure_map(pmat, joints+np.array([0.286, 0.286, 0.0])+10., None, mesh_matrix, joints+np.array([0.286, 0.286, 0.0])+10.)
            #time.sleep(5)

            #break

        filename = '/home/henry/data/synth/train_' + gender + '_' + posture + '_' + str(num_resting_poses) + '_of_' + str(
                                num_resting_poses_tried) + '_' + stiffness + '_stiff_new.p'
        pickle.dump(training_data_dict, open(os.path.join(filename), 'wb'))




if __name__ == "__main__":
    get_contact_map_from_synth()
    #reprocess_synth_data()

