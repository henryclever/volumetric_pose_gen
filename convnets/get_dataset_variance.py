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
from tensorprep_lib import TensorPrepLib
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

        # change this to 'direct' when you are doing baseline methods

        self.CTRL_PNL = {}
        self.CTRL_PNL['loss_vector_type'] = opt.losstype

        self.CTRL_PNL['verbose'] = opt.verbose
        self.opt = opt
        self.CTRL_PNL['batch_size'] = 128
        self.CTRL_PNL['num_epochs'] = 201
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = True
        self.CTRL_PNL['incl_ht_wt_channels'] = True
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 3
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        repeat_real_data_ct = 3
        self.CTRL_PNL['regr_angles'] = False
        self.CTRL_PNL['aws'] = False
        self.CTRL_PNL['depth_map_labels'] = True #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['depth_map_labels_test'] = True #can only be true is we have 100% synth for testing
        self.CTRL_PNL['depth_map_output'] = self.CTRL_PNL['depth_map_labels']
        self.CTRL_PNL['depth_map_input_est'] = True #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['depth_map_input_est'] #holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True
        self.CTRL_PNL['normalize_input'] = True
        self.CTRL_PNL['all_tanh_activ'] = False
        self.CTRL_PNL['L2_contact'] = False


        if opt.losstype == 'direct':
            self.CTRL_PNL['depth_map_labels'] = False
            self.CTRL_PNL['depth_map_output'] = False
        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['depth_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL['num_input_channels'] += 3
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.CTRL_PNL['num_input_channels'] += 2

        self.CTRL_PNL['filepath_prefix'] = '/home/henry/'

        if self.CTRL_PNL['depth_map_output'] == True: #we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

        print self.CTRL_PNL['num_epochs'], 'NUM EPOCHS!'
        # Entire pressure dataset with coordinates in world frame

        self.save_name = '_' + opt.losstype + \
                         '_synth_32000' + \
                         '_' + str(self.CTRL_PNL['batch_size']) + 'b' + \
                         '_' + str(self.CTRL_PNL['num_epochs']) + 'e'



        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



        #################################### PREP TRAINING DATA ##########################################
        #load training ysnth data
        dat_f_synth = TensorPrepLib().load_files_to_database(training_database_file_f, 'synth')
        dat_m_synth = TensorPrepLib().load_files_to_database(training_database_file_m, 'synth')
        dat_f_real = TensorPrepLib().load_files_to_database(training_database_file_f, 'real')
        dat_m_real = TensorPrepLib().load_files_to_database(training_database_file_m, 'real')


        self.train_x_flat = []  # Initialize the testing pressure mat list
        self.train_x_flat = TensorPrepLib().prep_images(self.train_x_flat, dat_f_synth, dat_m_synth, num_repeats = 1)
        self.train_x_flat = list(np.clip(np.array(self.train_x_flat) * 4.0, a_min=0, a_max=100))
        self.train_x_flat = TensorPrepLib().prep_images(self.train_x_flat, dat_f_real, dat_m_real, num_repeats = repeat_real_data_ct)
        self.train_x_flat = PreprocessingLib().preprocessing_blur_images(self.train_x_flat, self.mat_size, sigma=0.5)

        if len(self.train_x_flat) == 0: print("NO TRAINING DATA INCLUDED")

        self.train_a_flat = []  # Initialize the training pressure mat angle list
        self.train_a_flat = TensorPrepLib().prep_angles(self.train_a_flat, dat_f_synth, dat_m_synth, num_repeats = 1)
        self.train_a_flat = TensorPrepLib().prep_angles(self.train_a_flat, dat_f_real, dat_m_real, num_repeats = repeat_real_data_ct)

        if self.CTRL_PNL['depth_map_labels'] == True:
            self.depth_contact_maps = [] #Initialize the precomputed depth and contact maps. only synth has this label.
            self.depth_contact_maps = TensorPrepLib().prep_depth_contact(self.depth_contact_maps, dat_f_synth, dat_m_synth, num_repeats = 1)
        else:
            self.depth_contact_maps = None

        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.depth_contact_maps_input_est = [] #Initialize the precomputed depth and contact map input estimates
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(self.depth_contact_maps_input_est,
                                                                                             dat_f_synth, dat_m_synth, num_repeats = 1)
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(self.depth_contact_maps_input_est,
                                                                                             dat_f_real, dat_m_real, num_repeats = repeat_real_data_ct)
        else:
            self.depth_contact_maps_input_est = None

        #stack the bed height array on the pressure image as well as a sobel filtered image
        train_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.train_x_flat,
                                                                                self.train_a_flat,
                                                                                self.CTRL_PNL['incl_inter'], self.mat_size,
                                                                                self.CTRL_PNL['clip_sobel'],
                                                                                self.CTRL_PNL['verbose'])

        #stack the depth and contact mesh images (and possibly a pmat contact image) together
        train_xa = TensorPrepLib().append_input_depth_contact(np.array(train_xa),
                                                              include_pmat_contact = self.CTRL_PNL['incl_pmat_cntct_input'],
                                                              mesh_depth_contact_maps_input_est = self.depth_contact_maps_input_est,
                                                              include_mesh_depth_contact_input_est = self.CTRL_PNL['depth_map_input_est'],
                                                              mesh_depth_contact_maps = self.depth_contact_maps,
                                                              include_mesh_depth_contact = self.CTRL_PNL['depth_map_labels'])
        self.train_x = train_xa


        self.train_y_flat = []  # Initialize the training ground truth list
        self.train_y_flat = TensorPrepLib().prep_labels(self.train_y_flat, dat_f_synth, num_repeats = 1,
                                                        z_adj = -0.075, gender = "f", is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                        full_body_rot=self.CTRL_PNL['full_body_rot'])
        self.train_y_flat = TensorPrepLib().prep_labels(self.train_y_flat, dat_m_synth, num_repeats = 1,
                                                        z_adj = -0.075, gender = "m", is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                        full_body_rot=self.CTRL_PNL['full_body_rot'])

        self.train_y_flat = TensorPrepLib().prep_labels(self.train_y_flat, dat_f_real, num_repeats = repeat_real_data_ct,
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                        full_body_rot=self.CTRL_PNL['full_body_rot'])
        self.train_y_flat = TensorPrepLib().prep_labels(self.train_y_flat, dat_m_real, num_repeats = repeat_real_data_ct,
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                        full_body_rot=self.CTRL_PNL['full_body_rot'])



    def get_std_of_types(self):

        self.train_y_flat = np.array(self.train_y_flat)

        print(np.shape(self.train_y_flat), 'size of the training database output')


        joint_positions = self.train_y_flat[:, 0:72].reshape(-1, 24, 3)/1000
        #print joint_positions.shape
        mean_joint_positions = np.mean(joint_positions, axis = 0)
        joint_positions_rel_mean = joint_positions - mean_joint_positions
        euclidean = np.linalg.norm(joint_positions_rel_mean, axis = 2)
        #print euclidean.shape
        std_of_euclidean = np.std(euclidean)
        print std_of_euclidean, 'std of euclidean'

        #print joint_positions[0, :, :]
        #print joint_positions_rel_mean[0, :, :]
        #print euclidean[0, :]
        #print betas[0, :]

        #print joint_angles[0, :]
        #print mean_joint_angles

        betas = self.train_y_flat[:, 72:82]
        std_of_betas = np.std(betas)
        print std_of_betas, 'std of betas'


        #now get the std of the root joint in atan2
        joints = self.train_y_flat[:, 82:82+72]
        mean_joints = np.mean(joints, axis = 0)
        joints_rel_mean = joints - mean_joints
        std_of_joints = np.std(joints_rel_mean)
        print std_of_joints, 'std joints'


        #now get the std of the root joint in atan2
        joint_atan2x = np.cos(self.train_y_flat[:, 82:85])
        joint_atan2y = np.sin(self.train_y_flat[:, 82:85])
        joint_atan2 = np.concatenate((joint_atan2x, joint_atan2y), axis = 1)
        mean_joint_atan2 = np.mean(joint_atan2, axis = 0)
        joint_atan2_rel_mean = joint_atan2 - mean_joint_atan2
        std_of_joint_atan2 = np.std(joint_atan2_rel_mean)
        print std_of_joint_atan2, 'std of atan2 full body rot'

        #for i in range(72):
        #    print i, np.min(self.train_y_flat[:, 82+i]), np.max(self.train_y_flat[:, 82+i])


        #now get the std of the depth matrix
        depth_matrix_down = np.copy(self.train_x[:, 4])
        depth_matrix_down[depth_matrix_down > 0] = 0
        depth_matrix_down = np.abs(depth_matrix_down)
        mean_depth_mats = np.mean(depth_matrix_down, axis = 0)
        depth_matrix_rel_mean = depth_matrix_down - mean_depth_mats
        std_of_depth_matrix_down = np.std(depth_matrix_rel_mean)
        print std_of_depth_matrix_down, 'std of depth matrix'

        #now get the std of the contact matrix
        cntct_matrix_down = np.copy(self.train_x[:, 5])
        cntct_matrix_down = np.abs(cntct_matrix_down)
        mean_cntct_mats = np.mean(cntct_matrix_down, axis = 0)
        cntct_matrix_rel_mean = cntct_matrix_down - mean_cntct_mats
        std_of_cntct_matrix_down = np.std(cntct_matrix_rel_mean)
        print std_of_cntct_matrix_down, 'std of cntct matrix'


        #####################################################################
        ########DO NOT SUBTRACT THE MEAN##############
        print "now computing standard dev of the input"
        print np.shape(self.train_x)

        weight_matrix = np.repeat(self.train_y_flat[:, 160:161], 64*27, axis=1).reshape(np.shape(self.train_y_flat)[0], 1, 64, 27)
        height_matrix = np.repeat(self.train_y_flat[:, 161:162], 64*27, axis=1).reshape(np.shape(self.train_y_flat)[0], 1, 64, 27)

        print 'got here'
        print weight_matrix.shape, 'weight mat'
        print height_matrix.shape, 'height mat'
        print weight_matrix[0, :, 0:2, 0:2]
        print height_matrix[0, :, 0:2, 0:2]

        input_array = np.concatenate((self.train_x, weight_matrix, height_matrix), axis = 1)
        print input_array[0, 6, 0:2, 0:2]
        print input_array[0, 7, 0:2, 0:2]
        print input_array.shape

        if self.CTRL_PNL['depth_map_input_est'] == True:
            input_types = ['pmat_contact   ',
                           'mdm est pos    ', 'mdm est neg    ', 'cm est         ',
                           'pmat x5 clipped', 'pmat sobel     ', 'bed angle      ',
                           'depth output   ', 'contact output ',
                           'weight_matrix  ', 'height_matrix ']
        else:
            input_types = ['pmat_contact   ',
                           'pmat x5 clipped', 'pmat sobel     ', 'bed angle      ',
                           'depth output   ', 'contact output ',
                           'weight_matrix  ', 'height_matrix  ']


        for i in range(len(input_types)):
            some_layer = np.copy(input_array[:, i, :, :])
            mean_some_layer = np.mean(some_layer, axis = 0)
            some_layer_rel_mean = some_layer - mean_some_layer
            std_some_layer = np.std(some_layer_rel_mean, axis = 0)
            mean_some_layer = np.mean(mean_some_layer)
            std_some_layer = np.mean(std_some_layer)
            print i, input_types[i], '  mean is: %.3f  \t  std is: %.14f \t  min/max: %.3f, \t %.3f' %(mean_some_layer, std_some_layer, np.min(some_layer), np.max(some_layer))


        num_epochs = self.num_epochs
        hidden_dim = 12
        kernel_size = 10

        print "GOT HERE!!"

        #synthetic joints STD: 0.1282715100608753
        #synthetic betas STD: 1.7312621950698526
        #synthetic angles STD: 0.2130542427733348

    def get_bincount_ints_images(self):

        self.train_x_flat = np.array(self.train_x_flat).astype(int)

        print np.shape(self.train_x_flat), 'size of the training database'

        num_images = np.shape(self.train_x_flat)[0]

        self.train_x_flat = self.train_x_flat.flatten()
        self.train_x_flat[self.train_x_flat > 100] = 100

        #print self.train_x_flat[0:100]


        bin_count = []

        for i in range(0, 101):
            bin_count.append(np.count_nonzero(self.train_x_flat == i))

        bin_count = np.array(bin_count)
        print bin_count

        bin_count = bin_count/float(num_images)


        import matplotlib.pyplot as plt


        real = [9.32182891e+02, 2.41353293e+01, 2.22440547e+01, 2.01024808e+01,
                1.84882806e+01, 1.74491018e+01, 1.66834902e+01, 1.54379812e+01,
                1.45082121e+01, 1.36798118e+01, 1.33175364e+01, 1.25483319e+01,
                1.20975192e+01, 1.20109495e+01, 1.15692900e+01, 1.09368691e+01,
                1.09317365e+01, 1.05028229e+01, 1.02795552e+01, 9.92412318e+00,
                8.99520958e+00, 8.80025663e+00, 8.40556031e+00, 8.03772455e+00,
                7.69281437e+00, 7.30863986e+00, 6.95970915e+00, 6.62763045e+00,
                6.35337896e+00, 6.11274594e+00, 5.87134303e+00, 5.63250642e+00,
                5.39905902e+00, 5.11334474e+00, 4.94893071e+00, 4.92908469e+00,
                4.62728828e+00, 4.45457656e+00, 4.35637297e+00, 4.24901625e+00,
                3.91548332e+00, 3.83036784e+00, 3.63336185e+00, 3.66638152e+00,
                3.56971771e+00, 3.41120616e+00, 3.39136014e+00, 3.25355004e+00,
                3.23199316e+00, 3.17416595e+00, 3.11591104e+00, 3.09546621e+00,
                3.02053037e+00, 2.89461078e+00, 2.95816938e+00, 2.94200171e+00,
                2.85312233e+00, 2.82566296e+00, 2.68169376e+00, 2.67639008e+00,
                2.62412318e+00, 2.51035073e+00, 2.48083832e+00, 2.38879384e+00,
                2.39315654e+00, 2.35825492e+00, 2.32557742e+00, 2.23798118e+00,
                2.26236099e+00, 2.23028229e+00, 2.20923867e+00, 2.13840890e+00,
                2.14388366e+00 ,2.11608212e+00, 2.09324209e+00, 2.02095808e+00,
                1.92814371e+00, 1.98785287e+00, 1.91676647e+00, 1.79811805e+00,
                1.73370402e+00, 1.70384944e+00, 1.70940975e+00, 1.65089820e+00,
                1.55705731e+00, 1.47305389e+00, 1.47596236e+00, 1.40795552e+00,
                1.32814371e+00, 1.26595381e+00, 1.22309666e+00, 1.16133447e+00,
                1.06261762e+00, 9.98374679e-01, 9.68434559e-01, 9.25919589e-01,
                7.92044482e-01, 7.42857143e-01, 7.16595381e-01, 6.48759624e-01,
                2.61242857e+02]


        print bin_count

        plt.plot(np.arange(0, 99), real[1:100], 'r.', label='real')
        plt.plot(np.arange(0, 99), bin_count[1:100], 'b.', label='synth')
        plt.ylabel('Taxel Force Count \n (force rounded to nearest whole #)')
        plt.xlabel('Force (scale of 0 to 100)')
        plt.legend()
        plt.show()



    def plot_bincounts(self):

        real = [9.32182891e+02, 2.41353293e+01, 2.22440547e+01, 2.01024808e+01,
                1.84882806e+01, 1.74491018e+01, 1.66834902e+01, 1.54379812e+01,
                1.45082121e+01, 1.36798118e+01, 1.33175364e+01, 1.25483319e+01,
                1.20975192e+01, 1.20109495e+01, 1.15692900e+01, 1.09368691e+01,
                1.09317365e+01, 1.05028229e+01, 1.02795552e+01, 9.92412318e+00,
                8.99520958e+00, 8.80025663e+00, 8.40556031e+00, 8.03772455e+00,
                7.69281437e+00, 7.30863986e+00, 6.95970915e+00, 6.62763045e+00,
                6.35337896e+00, 6.11274594e+00, 5.87134303e+00, 5.63250642e+00,
                5.39905902e+00, 5.11334474e+00, 4.94893071e+00, 4.92908469e+00,
                4.62728828e+00, 4.45457656e+00, 4.35637297e+00, 4.24901625e+00,
                3.91548332e+00, 3.83036784e+00, 3.63336185e+00, 3.66638152e+00,
                3.56971771e+00, 3.41120616e+00, 3.39136014e+00, 3.25355004e+00,
                3.23199316e+00, 3.17416595e+00, 3.11591104e+00, 3.09546621e+00,
                3.02053037e+00, 2.89461078e+00, 2.95816938e+00, 2.94200171e+00,
                2.85312233e+00, 2.82566296e+00, 2.68169376e+00, 2.67639008e+00,
                2.62412318e+00, 2.51035073e+00, 2.48083832e+00, 2.38879384e+00,
                2.39315654e+00, 2.35825492e+00, 2.32557742e+00, 2.23798118e+00,
                2.26236099e+00, 2.23028229e+00, 2.20923867e+00, 2.13840890e+00,
                2.14388366e+00 ,2.11608212e+00, 2.09324209e+00, 2.02095808e+00,
                1.92814371e+00, 1.98785287e+00, 1.91676647e+00, 1.79811805e+00,
                1.73370402e+00, 1.70384944e+00, 1.70940975e+00, 1.65089820e+00,
                1.55705731e+00, 1.47305389e+00, 1.47596236e+00, 1.40795552e+00,
                1.32814371e+00, 1.26595381e+00, 1.22309666e+00, 1.16133447e+00,
                1.06261762e+00, 9.98374679e-01, 9.68434559e-01, 9.25919589e-01,
                7.92044482e-01, 7.42857143e-01, 7.16595381e-01, 6.48759624e-01,
                2.61242857e+02]

        synth = [1343.22232192,    4.07216977,    3.87526332,    3.73279629,    3.56068113,
                3.46481236,    3.41259655,    3.32685106,    3.2357806 ,    3.21613872,
                3.15151752,    3.10880081,    3.08726691,    3.03938129,    3.03341266,
                3.03470001,    2.98244519,    2.95613248,    2.90116642,    2.87887181,
                2.87982757,    2.85770851,    2.84108996,    2.80812593,    2.75975267,
                2.69887649,    2.64839666,    2.63983381,    2.59725365,    2.56867832,
                2.53407584,    2.52746353,    2.47682765,    2.46282281,    2.4540064,
                2.4299173 ,    2.41727783,    2.38047125,    2.38825388,    2.34370367,
                2.34918468,    2.31224155,    2.3065655 ,    2.30047983,    2.27030506,
                2.26591636,    2.25466178,    2.23172349,    2.22421393,    2.23084575,
                2.18783647,    2.17234922,    2.19435125,    2.15159554,    2.14320824,
                2.16341578,    2.12971054,    2.11356012,    2.12680424,    2.0678786,
                2.10441211,    2.05186471,    2.08133729,    2.06109074,    2.0262932,
                2.0886713 ,    1.99847858,    1.97450651,    1.97667161,    1.96077475,
                1.94228369,    1.92377311,    1.93130218,    1.91366935,    1.88341656,
                1.87058204,    1.86783179,    1.85429508,    1.83785207,    1.8299134,
                1.83151283,    1.81274869,    1.79724194,    1.7861434 ,    1.7733479,
                1.75464227,    1.74748381,    1.73125536,    1.74069595,    1.71299836,
                1.69971522,    1.68054147,    1.65883202,    1.65775923,    1.6430717,
                1.63571819,    1.60712335,    1.59345011,    1.59040727,    1.57058984,
                138.61369665]



        plt.plot(np.arange(0, 99), real[1:100], '-r', label='real')
        plt.plot(np.arange(0, 99), synth[1:100], '-b', label='synth')
        plt.ylabel('Taxel Force Count \n (force rounded to nearest whole #)')
        plt.xlabel('Force (scale of 0 to 100)')
        plt.legend()
        plt.show()




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

    #filepath_prefix = '/media/henry/multimodal_data_2/data/'
    filepath_prefix = '/home/henry/data/'

    training_database_file_f = []
    training_database_file_m = []


    training_database_file_f.append(filepath_prefix+'synth/random/train_roll0_f_lay_4000_none_stiff_output0p5.p')
    training_database_file_f.append(filepath_prefix+'synth/random/train_rollpi_f_lay_4000_none_stiff_output0p5.p')
    training_database_file_f.append(filepath_prefix+'synth/random/train_roll0_plo_f_lay_4000_none_stiff_output0p5.p')
    training_database_file_f.append(filepath_prefix+'synth/random/train_rollpi_plo_f_lay_4000_none_stiff_output0p5.p')
    training_database_file_m.append(filepath_prefix+'synth/random/train_roll0_m_lay_4000_none_stiff_output0p5.p')
    training_database_file_m.append(filepath_prefix+'synth/random/train_rollpi_m_lay_4000_none_stiff_output0p5.p')
    training_database_file_m.append(filepath_prefix+'synth/random/train_roll0_plo_m_lay_4000_none_stiff_output0p5.p')
    training_database_file_m.append(filepath_prefix+'synth/random/train_rollpi_plo_m_lay_4000_none_stiff_output0p5.p')
    training_database_file_f.append(filepath_prefix+'synth/random/test_roll0_f_lay_1000_none_stiff_output0p5.p')
    training_database_file_f.append(filepath_prefix+'synth/random/test_rollpi_f_lay_1000_none_stiff_output0p5.p')
    training_database_file_f.append(filepath_prefix+'synth/random/test_roll0_plo_f_lay_1000_none_stiff_output0p5.p')
    training_database_file_f.append(filepath_prefix+'synth/random/test_rollpi_plo_f_lay_1000_none_stiff_output0p5.p')
    training_database_file_m.append(filepath_prefix+'synth/random/test_roll0_m_lay_1000_none_stiff_output0p5.p')
    training_database_file_m.append(filepath_prefix+'synth/random/test_rollpi_m_lay_1000_none_stiff_output0p5.p')
    training_database_file_m.append(filepath_prefix+'synth/random/test_roll0_plo_m_lay_1000_none_stiff_output0p5.p')

    training_database_file_m.append(filepath_prefix+'synth/random/test_rollpi_plo_m_lay_1000_none_stiff_output0p5.p')


    #training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_lay_2000_of_2072_leftside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3555_upperbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3681_rightside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3722_leftside_st    iff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3808_lowerbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3829_none_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1508_upperbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1534_rightside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1513_leftside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1494_lowerbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1649_none_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_f.append(filepath_prefix_qt+'/real/s4_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_f.append(filepath_prefix_qt+'/real/s8_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_f.append(filepath_prefix_qt+'/real/trainval8_150rh1_sit120rh.p')

    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3573_upperbody_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3628_rightside_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3646_leftside_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3735_lowerbody_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3841_none_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1302_upperbody_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1259_rightside_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1302_leftside_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1275_lowerbody_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1414_none_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/s3_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/s5_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/s6_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/s7_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/trainval4_150rh1_sit120rh.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_95_rightside_stiff.p')


    p = PhysicalTrainer(training_database_file_f, training_database_file_m, opt)

    p.get_std_of_types()
    #p.get_bincount_ints_images()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
