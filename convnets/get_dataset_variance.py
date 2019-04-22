#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *


#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
GPU = False

import chumpy as ch


import convnet as convnet
#import tf.transformations as tft

# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Pose Estimation Libraries
from visualization_lib import VisualizationLib
from preprocessing_lib import PreprocessingLib
from synthetic_lib import SyntheticLib


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
from sklearn.preprocessing import normalize
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor

np.set_printoptions(threshold=sys.maxsize)

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TRAIN = 24
NUMOFOUTPUTNODES_TEST = 10
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)

torch.set_num_threads(1)
#if torch.cuda.is_available():
if GPU == True:
    # Use for GPU
    dtype = torch.cuda.FloatTensor
    print '######################### CUDA is available! #############################'
else:
    # Use for CPU
    dtype = torch.FloatTensor
    print '############################## USING CPU #################################'

class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''
    def __init__(self, training_database_file_f, training_database_file_m, opt):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''


        #change this to 'direct' when you are doing baseline methods
        self.loss_vector_type = opt.losstype

        self.verbose = opt.verbose
        self.opt = opt
        self.batch_size = 128
        self.num_epochs = 300
        self.include_inter = True
        self.shuffle = True


        self.count = 0


        print self.num_epochs, 'NUM EPOCHS!'
        #Entire pressure dataset with coordinates in world frame

        self.save_name = '_' + opt.losstype+'_' +str(self.shuffle)+'s_' + str(self.batch_size) + 'b_' + str(self.num_epochs) + 'e'



        print 'appending to','train'+self.save_name
        self.train_val_losses = {}
        self.train_val_losses['train'+self.save_name] = []
        self.train_val_losses['val'+self.save_name] = []
        self.train_val_losses['epoch'+self.save_name] = []

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)

        dat_f_synth = self.load_files_to_database(training_database_file_f, 'synth', 'training f synth')
        dat_f_real = self.load_files_to_database(training_database_file_f, 'real', 'training f real')
        dat_m_synth = self.load_files_to_database(training_database_file_m, 'synth', 'training m synth')
        dat_m_real = self.load_files_to_database(training_database_file_m, 'real', 'training m real')

        self.train_x_flat = []  # Initialize the testing pressure mat list
        if dat_f_synth is not None:
            for entry in range(len(dat_f_synth['images'])):
                self.train_x_flat.append(dat_f_synth['images'][entry] * 3)
        if dat_f_real is not None:
            for entry in range(len(dat_f_real['images'])):
                self.train_x_flat.append(dat_f_real['images'][entry])
        if dat_m_synth is not None:
            for entry in range(len(dat_m_synth['images'])):
                self.train_x_flat.append(dat_m_synth['images'][entry] * 3)
        if dat_m_real is not None:
            for entry in range(len(dat_m_real['images'])):
                self.train_x_flat.append(dat_m_real['images'][entry])

        self.train_a_flat = []  # Initialize the testing pressure mat angle list
        if dat_f_synth is not None:
            for entry in range(len(dat_f_synth['images'])):
                self.train_a_flat.append(dat_f_synth['bed_angle_deg'][entry])
        if dat_f_real is not None:
            for entry in range(len(dat_f_real['images'])):
                self.train_a_flat.append(dat_f_real['bed_angle_deg'][entry])
        if dat_m_synth is not None:
            for entry in range(len(dat_m_synth['images'])):
                self.train_a_flat.append(dat_m_synth['bed_angle_deg'][entry])
        if dat_m_real is not None:
            for entry in range(len(dat_m_real['images'])):
                self.train_a_flat.append(dat_m_real['bed_angle_deg'][entry])

        train_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.train_x_flat,
                                                                                self.train_a_flat,
                                                                                self.include_inter, self.mat_size,
                                                                                self.verbose)
        train_xa = np.array(train_xa)
        self.train_y_flat = []  # Initialize the training ground truth list

        if dat_f_synth is not None:
            for entry in range(len(dat_f_synth['markers_xyz_m'])):
                if self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
                    # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                    c = np.concatenate((dat_f_synth['markers_xyz_m'][entry][0:72] * 1000,
                                        dat_f_synth['body_shape'][entry][0:10],
                                        dat_f_synth['joint_angles'][entry][0:72],
                                        dat_f_synth['root_xyz_shift'][entry][0:3],
                                        [1], [0], [1]), axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    self.train_y_flat.append(c)
                else:
                    # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                    c = np.concatenate((dat_f_synth['markers_xyz_m'][entry][0:72] * 1000,
                                        np.array(10 * [0]),
                                        np.array(72 * [0]),
                                        np.array(3 * [0]),
                                        [1], [0], [1]), axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    self.train_y_flat.append(c)

        
        if dat_f_real is not None:
            for entry in range(len(dat_f_real['markers_xyz_m'])):  ######FIX THIS!!!!######
                # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                # print np.shape(dat_f_real['markers_xyz_m'][entry])
                fixed_head_markers = dat_f_real['markers_xyz_m'][entry][0:3] * 1000 + \
                                     [0.0,
                                      -141.4*np.cos(np.deg2rad(dat_f_real['bed_angle_deg'][entry]+45)),
                                      -141.4*np.sin(np.deg2rad(dat_f_real['bed_angle_deg'][entry]+45))]
                #print dat_f_real['bed_angle_deg'][entry], dat_f_real['markers_xyz_m'][entry][0:3] * 1000, fixed_head_markers

                fixed_torso_markers = dat_f_real['markers_xyz_m'][entry][3:6] * 1000 + [0.0, 0.0, -100.0]

                c = np.concatenate((np.array(9 * [0]),
                                    #dat_f_real['markers_xyz_m'][entry][3:6] * 1000,  # TORSO
                                    fixed_torso_markers,  # TORSO
                                    dat_f_real['markers_xyz_m'][entry][21:24] * 1000,  # L KNEE
                                    dat_f_real['markers_xyz_m'][entry][18:21] * 1000,  # R KNEE
                                    np.array(3 * [0]),
                                    dat_f_real['markers_xyz_m'][entry][27:30] * 1000,  # L ANKLE
                                    dat_f_real['markers_xyz_m'][entry][24:27] * 1000,  # R ANKLE
                                    np.array(18 * [0]),
                                    #dat_f_real['markers_xyz_m'][entry][0:3] * 1000,  # HEAD
                                    fixed_head_markers,
                                    np.array(6 * [0]),
                                    dat_f_real['markers_xyz_m'][entry][9:12] * 1000,  # L ELBOW
                                    dat_f_real['markers_xyz_m'][entry][6:9] * 1000,  # R ELBOW
                                    dat_f_real['markers_xyz_m'][entry][15:18] * 1000,  # L WRIST
                                    dat_f_real['markers_xyz_m'][entry][12:15] * 1000,  # R WRIST
                                    np.array(6 * [0]),
                                    np.array(85 * [0]),
                                    [1], [0], [0]), axis=0)  # [x1], [x2], [x3]: female real: 1, 0, 0.
                self.train_y_flat.append(c)

        if dat_m_synth is not None:
            for entry in range(len(dat_m_synth['markers_xyz_m'])):
                if self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
                    # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                    c = np.concatenate((dat_m_synth['markers_xyz_m'][entry][0:72] * 1000,
                                        dat_m_synth['body_shape'][entry][0:10],
                                        dat_m_synth['joint_angles'][entry][0:72],
                                        dat_m_synth['root_xyz_shift'][entry][0:3],
                                        [0], [1], [1]), axis=0)  # [x1], [x2], [x3]: male synth: 0, 1, 1.
                    self.train_y_flat.append(c)
                else:
                    # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                    c = np.concatenate((dat_m_synth['markers_xyz_m'][entry][0:72] * 1000,
                                        np.array(10 * [0]),
                                        np.array(72 * [0]),
                                        np.array(3 * [0]),
                                        [1], [0], [1]), axis=0)  # [x1], [x2], [x3]: female synth: 1, 0, 1.
                    self.train_y_flat.append(c)

        if dat_m_real is not None:
            for entry in range(len(dat_m_real['markers_xyz_m'])):  ######FIX THIS!!!!######
                # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                fixed_head_markers = dat_m_real['markers_xyz_m'][entry][0:3] * 1000 + \
                                     [0.0,
                                      -141.4*np.cos(np.deg2rad(dat_m_real['bed_angle_deg'][entry]+45)),
                                      -141.4*np.sin(np.deg2rad(dat_m_real['bed_angle_deg'][entry]+45))]

                fixed_torso_markers = dat_m_real['markers_xyz_m'][entry][3:6] * 1000 + [0.0, 0.0, -100.0]

                c = np.concatenate((np.array(9 * [0]),
                                    #dat_m_real['markers_xyz_m'][entry][3:6] * 1000,  # TORSO
                                    fixed_torso_markers,
                                    dat_m_real['markers_xyz_m'][entry][21:24] * 1000,  # L KNEE
                                    dat_m_real['markers_xyz_m'][entry][18:21] * 1000,  # R KNEE
                                    np.array(3 * [0]),
                                    dat_m_real['markers_xyz_m'][entry][27:30] * 1000,  # L ANKLE
                                    dat_m_real['markers_xyz_m'][entry][24:27] * 1000,  # R ANKLE
                                    np.array(18 * [0]),
                                    #dat_m_real['markers_xyz_m'][entry][0:3] * 1000,  # HEAD
                                    fixed_head_markers,
                                    np.array(6 * [0]),
                                    dat_m_real['markers_xyz_m'][entry][9:12] * 1000,  # L ELBOW
                                    dat_m_real['markers_xyz_m'][entry][6:9] * 1000,  # R ELBOW
                                    dat_m_real['markers_xyz_m'][entry][15:18] * 1000,  # L WRIST
                                    dat_m_real['markers_xyz_m'][entry][12:15] * 1000,  # R WRIST
                                    np.array(6 * [0]),
                                    np.array(85 * [0]),
                                    [0], [1], [0]), axis=0)  # [x1], [x2], [x3]: male real: 0, 1, 0.
                self.train_y_flat.append(c)





    def load_files_to_database(self, database_file, creation_type, descriptor):
        # load in the training or testing files.  This may take a while.
        try:
            for some_subject in database_file:
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

            for key in dat:
                print descriptor, key, np.array(dat[key]).shape
        except:
            dat = None
        return dat




    def get_std_of_types(self):

        self.train_x_flat = np.array(self.train_x_flat)
        self.train_y_flat = np.array(self.train_y_flat)

        if self.verbose: print np.shape(self.train_x_flat), 'size of the training database'
        if self.verbose: print np.shape(self.train_y_flat), 'size of the training database output'


        joint_positions = self.train_y_flat[:, 0:72].reshape(-1, 24, 3)/1000
        betas = self.train_y_flat[:, 72:82]
        joint_angles = self.train_y_flat[:, 82:154]
        #print joint_positions.shape
        mean_joint_positions = np.mean(joint_positions, axis = 0)
        joint_positions_rel_mean = joint_positions - mean_joint_positions
        euclidean = np.linalg.norm(joint_positions_rel_mean, axis = 2)
        #print euclidean.shape


        mean_joint_angles = np.mean(joint_angles, axis = 0)
        joint_angles_rel_mean = joint_angles - mean_joint_angles



        #print joint_positions[0, :, :]
        #print joint_positions_rel_mean[0, :, :]
        #print euclidean[0, :]
        #print betas[0, :]

        #print joint_angles[0, :]
        #print mean_joint_angles

        std_of_euclidean = np.std(euclidean)
        std_of_betas = np.std(betas)
        std_of_angles = np.std(joint_angles_rel_mean)

        print std_of_euclidean #
        print std_of_betas
        print std_of_angles

        num_epochs = self.num_epochs
        hidden_dim = 12
        kernel_size = 10

        print "GOT HERE!!"

        #synthetic joints STD: 0.1282715100608753
        #synthetic betas STD: 1.7312621950698526
        #synthetic angles STD: 0.2130542427733348

    def get_bincount_ints_images(self):


        pass



if __name__ == "__main__":
    #Initialize trainer with a training database file
    import optparse
    p = optparse.OptionParser()
    p.add_option('--computer', action='store', type = 'string',
                 dest='computer', \
                 default='lab_harddrive', \
                 help='Set path to the training database on lab harddrive.')
    p.add_option('--gpu', action='store', type = 'string',
                 dest='gpu', \
                 default='0', \
                 help='Set the GPU you will use.')
    p.add_option('--losstype', action='store', type = 'string',
                 dest='losstype', \
                 default='anglesDC', \
                 help='Set if you want to do baseline ML or convnet.')
    p.add_option('--qt', action='store_true',
                 dest='quick_test', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')
    p.add_option('--viz', action='store_true',
                 dest='visualize', \
                 default=False, \
                 help='Visualize.')
    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=True, help='Printout everything (under construction).')

    p.add_option('--log_interval', type=int, default=5, metavar='N',
                 help='number of batches between logging train status')

    opt, args = p.parse_args()

    filepath_prefix_qt = '/home/henry/data'

    training_database_file_f = []
    training_database_file_m = []
    test_database_file_f = []
    test_database_file_m = []

    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3555_upperbody_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3681_rightside_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3722_leftside_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3808_lowerbody_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3829_none_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1508_upperbody_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1534_rightside_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1513_leftside_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1494_lowerbody_stiff.p')
    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1649_none_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_f.append(filepath_prefix_qt+'/real/s8_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_f.append(filepath_prefix_qt+'/real/trainval8_150rh1_sit120rh.p')

    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3573_upperbody_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3628_rightside_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3646_leftside_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3735_lowerbody_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3841_none_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1302_upperbody_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1259_rightside_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1302_leftside_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1275_lowerbody_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1414_none_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/s3_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/s5_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/s6_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/s7_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/trainval4_150rh1_sit120rh.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_95_rightside_stiff.p')


    p = PhysicalTrainer(training_database_file_f, training_database_file_m, opt)

    p.get_std_of_types()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
