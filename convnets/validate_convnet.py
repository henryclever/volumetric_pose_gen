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
GPU = True

import chumpy as ch


import convnet as convnet
import tf.transformations as tft

# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Pose Estimation Libraries
from visualization_lib import VisualizationLib
from preprocessing_lib import PreprocessingLib


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

    def __init__(self, training_database_file_f, training_database_file_m, testing_database_file_f,
                 testing_database_file_m, opt):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''

        # change this to 'direct' when you are doing baseline methods
        self.loss_vector_type = opt.losstype

        self.verbose = opt.verbose
        self.opt = opt
        self.batch_size = 128
        self.num_epochs = 400
        self.include_inter = True
        self.shuffle = True

        self.count = 0

        print
        testing_database_file_f
        print
        self.num_epochs, 'NUM EPOCHS!'
        # Entire pressure dataset with coordinates in world frame

        self.save_name = '_' + opt.losstype + '_' + str(self.shuffle) + 's_' + str(self.batch_size) + 'b_' + str(
            self.num_epochs) + 'e'

        print
        'appending to', 'train' + self.save_name
        self.train_val_losses = {}
        self.train_val_losses['train' + self.save_name] = []
        self.train_val_losses['val' + self.save_name] = []
        self.train_val_losses['epoch' + self.save_name] = []

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
        self.train_x_tensor = torch.Tensor(train_xa)

        print self.train_x_tensor.shape, 'tensor shape'

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
                    self.train_y_flat.append(dat_f_synth['markers_xyz_m'][entry][0:72] * 1000)
        
        if dat_f_real is not None:
            for entry in range(len(dat_f_real['markers_xyz_m'])):  ######FIX THIS!!!!######
                if self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
                    # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                    # print np.shape(dat_f_real['markers_xyz_m'][entry])
                    c = np.concatenate((np.array(9 * [0]),
                                        dat_f_real['markers_xyz_m'][entry][3:6] * 1000,  # TORSO
                                        dat_f_real['markers_xyz_m'][entry][21:24] * 1000,  # L KNEE
                                        dat_f_real['markers_xyz_m'][entry][18:21] * 1000,  # R KNEE
                                        np.array(3 * [0]),
                                        dat_f_real['markers_xyz_m'][entry][27:30] * 1000,  # L ANKLE
                                        dat_f_real['markers_xyz_m'][entry][24:27] * 1000,  # R ANKLE
                                        np.array(18 * [0]),
                                        dat_f_real['markers_xyz_m'][entry][0:3] * 1000,  # HEAD
                                        np.array(6 * [0]),
                                        dat_f_real['markers_xyz_m'][entry][9:12] * 1000,  # L ELBOW
                                        dat_f_real['markers_xyz_m'][entry][6:9] * 1000,  # R ELBOW
                                        dat_f_real['markers_xyz_m'][entry][15:18] * 1000,  # L WRIST
                                        dat_f_real['markers_xyz_m'][entry][12:15] * 1000,  # R WRIST
                                        np.array(6 * [0]),
                                        np.array(85 * [0]),
                                        [1], [0], [0]), axis=0)  # [x1], [x2], [x3]: female real: 1, 0, 0.
                    self.train_y_flat.append(c)
                else:
                    self.train_y_flat.append(dat_f_real['markers_xyz_m'][entry][0:72] * 1000)

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
                    self.train_y_flat.append(dat_m_synth['markers_xyz_m'][entry][0:72] * 1000)

        if dat_m_real is not None:
            for entry in range(len(dat_m_real['markers_xyz_m'])):  ######FIX THIS!!!!######
                if self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
                    # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                    c = np.concatenate((np.array(9 * [0]),
                                        dat_m_real['markers_xyz_m'][entry][3:6] * 1000,  # TORSO
                                        dat_m_real['markers_xyz_m'][entry][21:24] * 1000,  # L KNEE
                                        dat_m_real['markers_xyz_m'][entry][18:21] * 1000,  # R KNEE
                                        np.array(3 * [0]),
                                        dat_m_real['markers_xyz_m'][entry][27:30] * 1000,  # L ANKLE
                                        dat_m_real['markers_xyz_m'][entry][24:27] * 1000,  # R ANKLE
                                        np.array(18 * [0]),
                                        dat_m_real['markers_xyz_m'][entry][0:3] * 1000,  # HEAD
                                        np.array(6 * [0]),
                                        dat_m_real['markers_xyz_m'][entry][9:12] * 1000,  # L ELBOW
                                        dat_m_real['markers_xyz_m'][entry][6:9] * 1000,  # R ELBOW
                                        dat_m_real['markers_xyz_m'][entry][15:18] * 1000,  # L WRIST
                                        dat_m_real['markers_xyz_m'][entry][12:15] * 1000,  # R WRIST
                                        np.array(6 * [0]),
                                        np.array(85 * [0]),
                                        [0], [1], [0]), axis=0)  # [x1], [x2], [x3]: male real: 0, 1, 0.
                    self.train_y_flat.append(c)
                else:
                    self.train_y_flat.append(dat_m_real['markers_xyz_m'][entry][0:72] * 1000)



        print np.shape(self.train_y_flat), 'shape flat !'
        self.train_y_tensor = torch.Tensor(self.train_y_flat)

        # load in the test file
        test_dat_f = self.load_files_to_database(testing_database_file_f, 'real', 'training f real')
        test_dat_m = self.load_files_to_database(testing_database_file_m, 'real', 'training f real')

        # create a tensor for our testing dataset.  First print out how many input/output sets we have and what data we have
        for key in test_dat_f:
            print
            'testing set: ', key, np.array(test_dat_f[key]).shape
        for key in test_dat_m:
            print
            'testing set: ', key, np.array(test_dat_m[key]).shape

        self.test_x_flat = []  # Initialize the testing pressure mat list
        for entry in range(len(test_dat_f['images'])):
            self.test_x_flat.append(test_dat_f['images'][entry])
        for entry in range(len(test_dat_m['images'])):
            self.test_x_flat.append(test_dat_m['images'][entry])

        self.test_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(test_dat_f['images'])):
            self.test_a_flat.append(test_dat_f['bed_angle_deg'][entry])
        for entry in range(len(test_dat_m['images'])):
            self.test_a_flat.append(test_dat_m['bed_angle_deg'][entry])
        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat, self.test_a_flat,
                                                                               self.include_inter, self.mat_size,
                                                                               self.verbose)
        test_xa = np.array(test_xa)
        self.test_x_tensor = torch.Tensor(test_xa)

        self.test_y_flat = []  # Initialize the ground truth list
        for entry in range(len(test_dat_f['markers_xyz_m'])):
            if self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
                # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                c = np.concatenate((test_dat_f['markers_xyz_m'][entry] * 1000,
                                    [1], [0]), axis=0)  # shapedirs (N, 6890, 3, 10)
                self.test_y_flat.append(c)
            else:
                self.test_y_flat.append(test_dat_f['markers_xyz_m'][entry] * 1000)

        for entry in range(len(test_dat_m['markers_xyz_m'])):
            if self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
                # print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                c = np.concatenate((test_dat_m['markers_xyz_m'][entry] * 1000,
                                    [0], [1]), axis=0)  # shapedirs (N, 6890, 3, 10)
                self.test_y_flat.append(c)
            else:
                self.test_y_flat.append(test_dat_m['markers_xyz_m'][entry] * 1000)

        self.test_y_tensor = torch.Tensor(self.test_y_flat)

        self.parents = np.array(
            [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(
            np.int32)

    def load_files_to_database(self, database_file, creation_type, descriptor):
        # load in the training or testing files.  This may take a while.
        try:
            for some_subject in database_file:
                if creation_type in some_subject:
                    dat_curr = load_pickle(some_subject)
                    print
                    some_subject, dat_curr['bed_angle_deg'][0]
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
        except: dat = None
        return dat



    def visualize_3d_data(self):


        p_mat_array = self.train_x_tensor.numpy()[:, 0, :, :]
        bedangle_array = self.train_x_tensor.numpy()[:, 2, 0, 0]
        joint_loc_array = self.train_y_tensor.numpy()[:, 0:72].reshape(-1, 24, 3)/1000
        root_loc_array = self.train_y_tensor.numpy()[:, 154:157]

        import rospy
        rospy.init_node('real_time_pose')


        print p_mat_array.shape
        print joint_loc_array.shape


        VizLib = VisualizationLib()
        for ct in range(p_mat_array.shape[0]):
            print joint_loc_array[0]

            VizLib.rviz_publish_input(p_mat_array[ct, :, :], bedangle_array[ct])
            VizLib.rviz_publish_output(joint_loc_array[ct])

            print bedangle_array[ct], joint_loc_array[ct, 0, :], root_loc_array[ct, :]

            time.sleep(1.0)


    def init_convnet_train(self):

        if self.verbose: print self.test_x_tensor.size(), 'length of the testing dataset'
        if self.verbose: print self.test_y_tensor.size(), 'size of the training database output'



        num_epochs = self.num_epochs
        hidden_dim = 12
        kernel_size = 10


        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.batch_size, shuffle=self.shuffle)



        if self.loss_vector_type == 'direct':
            fc_output_size = 72
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type, self.batch_size)

        elif self.loss_vector_type == 'anglesR':
            fc_output_size = 229# 10 + 3 + 24*3*3 --- betas, root shift, rotations
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type, self.batch_size)
            #self.model = torch.load('/home/ubuntu/Autobed_OFFICIAL_Trials' + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_'+str(self.loss_vector_type)+'_sTrue_128b_200e_' + str(self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)
            print 'LOADED!!!!!!!!!!!!!!!!!1'
            pp = 0
            for p in list(self.model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'

        elif self.loss_vector_type == 'anglesDC':
            fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type, self.batch_size)
            #self.model = torch.load('/home/ubuntu/Autobed_OFFICIAL_Trials' + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_'+str(self.loss_vector_type)+'_sTrue_128b_200e_' + str(self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)
            print 'LOADED!!!!!!!!!!!!!!!!!1'
            pp = 0
            for p in list(self.model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'

        elif self.loss_vector_type == 'anglesEU':
            fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type, self.batch_size)
            #self.model = torch.load('/home/ubuntu/Autobed_OFFICIAL_Trials' + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_'+str(self.loss_vector_type)+'_sTrue_128b_200e_' + str(self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)
            print 'LOADED!!!!!!!!!!!!!!!!!1'
            pp = 0
            for p in list(self.model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'



        # Run model on GPU if available
        #if torch.cuda.is_available():
        if GPU == True:
            self.model = self.model.cuda()
            self.model = torch.load('/home/henry/data/training/convnet_direct_Trues_128b_400e.pt')


        self.criterion = F.cross_entropy


        print 'done with epochs, now evaluating'
        self.validate_convnet('test')


    def validate_convnet(self, verbose=False, n_batches=None):

        self.model.eval()
        loss = 0.
        n_examples = 0
        for batch_i, batch in enumerate(self.test_loader):

            self.model.eval()

            if self.loss_vector_type == 'direct':


                images_up_non_tensor = np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, :, :], multiple = 2))
                images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)
                images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad=False), Variable(batch[1].type(dtype), requires_grad=False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1] / 3))).type(dtype), requires_grad=False)

                scores, targets_est, targets_est_reduced = self.model.forward_direct(images_up, targets, is_training = False)

                self.criterion = nn.L1Loss()

                loss = self.criterion(scores, scores_zeros)
                loss = loss.data.item()



            elif self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':

                #print batch[1].shape

                #get the direct joint locations
                batch.append(batch[1][:, 30:32])
                batch[1] = batch[1][:, 0:30]


                images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy(), multiple=2)))
                images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)
                images, targets = Variable(batch[0].type(dtype), volatile=True, requires_grad=False), Variable(batch[1].type(dtype), volatile=True,requires_grad=False),

                gender_switch = Variable(batch[2].type(dtype), volatile=True, requires_grad=False)


                self.optimizer.zero_grad()


                ground_truth = np.zeros((batch[0].numpy().shape[0], 30))
                ground_truth = Variable(torch.Tensor(ground_truth).type(dtype))
                ground_truth[:, 0:30] = targets[:, 0:30]/1000

                scores_zeros = np.zeros((batch[0].numpy().shape[0], 10))
                scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))

                if self.loss_vector_type == 'anglesR':
                    scores, targets_est, targets_est_reduced, betas_est = self.model.forward_kinematic_R(images_up, gender_switch, targets, is_training=False)  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.
                elif self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
                    scores, targets_est, targets_est_reduced, betas_est = self.model.forward_kinematic_angles(images_up, gender_switch, targets, is_training=False)  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.

                self.criterion = nn.L1Loss()
                loss = self.criterion(scores, scores_zeros)
                loss = loss.data.item()




            n_examples += self.batch_size
            #print n_examples

            if n_batches and (batch_i >= n_batches):
                break

        loss /= n_examples
        loss *= 100
        loss *= 1000

        if GPU == True:
            VisualizationLib().print_error_val(targets.data.cpu(), targets_est_reduced.cpu(), self.output_size_val, self.loss_vector_type, data='validate')
        else:
            VisualizationLib().print_error_val(targets.data, targets_est_reduced, self.output_size_val, self.loss_vector_type, data='validate')

        if self.loss_vector_type == 'anglesR':
            #print angles_est[0, :], 'validation angles'
            print betas_est[0, :], 'validation betas'


        NUM_IMAGES = images.data.size()[0]

        for image_ct in range(NUM_IMAGES):
            # #self.im_sampleval = self.im_sampleval[:,0,:,:]
            self.im_sampleval = images.data[image_ct, :].squeeze()
            self.tar_sampleval = targets.data[image_ct, :].squeeze() / 1000
            self.sc_sampleval = targets_est[image_ct, :].squeeze() / 1000
            self.sc_sampleval = self.sc_sampleval.view(24, 3)



            if self.opt.visualize == True:
                if GPU == True:
                    VisualizationLib().visualize_pressure_map(self.im_sampleval.cpu(), self.tar_sampleval.cpu(), self.sc_sampleval.cpu(), block=False)
                else:
                    VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,self.im_sampleval, self.tar_sampleval, self.sc_sampleval, block=False)
            time.sleep(1)


        return loss



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
                 default='direct', \
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

    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3552_upperbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3681_rightside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3722_leftside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3808_lowerbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_lay_3829_none_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1508_upperbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1534_rightside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1513_leftside_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1494_lowerbody_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/synth/train_f_sit_1649_none_stiff.p')
    #training_database_file_f.append(filepath_prefix_qt+'/real/trainval8_150rh1_sit120rh.p')
    #training_database_file_f.append(filepath_prefix_qt + '/real/s8_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')

    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3573_upperbody_stiff.p')
    training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3628_rightside_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3646_leftside_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3735_lowerbody_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_lay_3841_none_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1302_upperbody_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1259_rightside_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1302_leftside_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1275_lowerbody_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/synth/train_m_sit_1414_none_stiff.p')
    #training_database_file_m.append(filepath_prefix_qt+'/real/trainval4_150rh1_sit120rh.p')
    #training_database_file_m.append(filepath_prefix_qt + '/real/s4_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')

    test_database_file_f.append(filepath_prefix_qt+'/real/trainval4_150rh1_sit120rh.p')
    test_database_file_m.append(filepath_prefix_qt+'/real/trainval8_150rh1_sit120rh.p')
    #test_database_file_f.append(filepath_prefix_qt + '/real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + '/real/s3_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + '/real/s4_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + '/real/s5_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + '/real/s6_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + '/real/s7_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_f.append(filepath_prefix_qt + '/real/s8_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    p = PhysicalTrainer(training_database_file_f, training_database_file_m, test_database_file_f, test_database_file_m, opt)

    #p.init_convnet_train()
    p.visualize_3d_data()
        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
