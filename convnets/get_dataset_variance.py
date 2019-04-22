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


        self.train_x_flat = PreprocessingLib().preprocessing_blur_images(self.train_x_flat, self.mat_size, sigma = 1.0)


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

        self.train_y_flat = np.array(self.train_y_flat)

        print(np.shape(self.train_y_flat), 'size of the training database output')


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

    filepath_prefix_qt = '/home/henry/data'

    training_database_file_f = []
    training_database_file_m = []
    test_database_file_f = []
    test_database_file_m = []

    training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3555_upperbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3681_rightside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3722_leftside_stiff.p')
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

    #p.get_std_of_types()
    p.get_bincount_ints_images()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
