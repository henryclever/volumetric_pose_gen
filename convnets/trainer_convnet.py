#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

import chumpy as ch

import convnet as convnet
# import tf.transformations as tft

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
from tensorprep_lib import TensorPrepLib
from unpack_batch_lib import UnpackBatchLib

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
# from skimage.feature import hog
# from skimage import data, color, exposure


from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor

np.set_printoptions(threshold=sys.maxsize)

MAT_WIDTH = 0.762  # metres
MAT_HEIGHT = 1.854  # metres
MAT_HALF_WIDTH = MAT_WIDTH / 2
NUMOFTAXELS_X = 64  # 73 #taxels
NUMOFTAXELS_Y = 27  # 30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TRAIN = 24
NUMOFOUTPUTNODES_TEST = 10
INTER_SENSOR_DISTANCE = 0.0286  # metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)
TEST_SUBJECT = 9

torch.set_num_threads(1)
if torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    print '######################### CUDA is available! #############################'
else:
    # Use for CPU
    GPU = False
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

        self.CTRL_PNL = {}
        self.CTRL_PNL['loss_vector_type'] = opt.losstype

        self.CTRL_PNL['verbose'] = opt.verbose
        self.opt = opt
        self.CTRL_PNL['batch_size'] = 128
        self.CTRL_PNL['num_epochs'] = 101
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = True
        self.CTRL_PNL['incl_ht_wt_channels'] = True
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 3
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        repeat_real_data_ct = 1
        self.CTRL_PNL['regr_angles'] = opt.reg_angles
        self.CTRL_PNL['aws'] = self.opt.aws
        self.CTRL_PNL['depth_map_labels'] = True #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['depth_map_output'] = self.CTRL_PNL['depth_map_labels']
        self.CTRL_PNL['depth_map_input_est'] = False #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = False#self.CTRL_PNL['depth_map_input_est'] #holds betas and root same as prior estimate

        self.weight_joints = self.opt.j_d_ratio*2
        self.weight_depth_planes = (1-self.opt.j_d_ratio)*2

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

        if self.opt.aws == True:
            self.CTRL_PNL['filepath_prefix'] = '/home/ubuntu/'
        else:
            self.CTRL_PNL['filepath_prefix'] = '/home/henry/'

        if self.CTRL_PNL['depth_map_output'] == True: #we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

        print self.CTRL_PNL['num_epochs'], 'NUM EPOCHS!'
        # Entire pressure dataset with coordinates in world frame

        self.save_name = '_' + opt.losstype + \
                         '_synth_s' + str(TEST_SUBJECT) + \
                         '_3xreal' + \
                         '_' + str(self.CTRL_PNL['batch_size']) + 'b' + \
                         '_' + str(self.CTRL_PNL['num_epochs']) + 'e'


        if self.CTRL_PNL['depth_map_labels'] == True:
            self.save_name += '_' + str(self.opt.j_d_ratio) + 'rtojtdpth'
        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.save_name += '_pmatcntin'
        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.save_name += '_depthestin'
        if self.CTRL_PNL['adjust_ang_from_est'] == True:
            self.save_name += '_angleadj'

        # self.save_name = '_' + opt.losstype+'_real_s9_alltest_' + str(self.CTRL_PNL['batch_size']) + 'b_'# + str(self.CTRL_PNL['num_epochs']) + 'e'

        print 'appending to', 'train' + self.save_name
        self.train_val_losses = {}
        self.train_val_losses['train' + self.save_name] = []
        self.train_val_losses['val' + self.save_name] = []
        self.train_val_losses['epoch' + self.save_name] = []

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
        self.train_x_flat = list(np.clip(np.array(self.train_x_flat) * 5.0, a_min=0, a_max=100))

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
                                                                                self.CTRL_PNL['verbose'])

        #stack the depth and contact mesh images (and possibly a pmat contact image) together
        train_xa = TensorPrepLib().append_input_depth_contact(np.array(train_xa),
                                                              include_pmat_contact = self.CTRL_PNL['incl_pmat_cntct_input'],
                                                              mesh_depth_contact_maps_input_est = self.depth_contact_maps_input_est,
                                                              include_mesh_depth_contact_input_est = self.CTRL_PNL['depth_map_input_est'],
                                                              mesh_depth_contact_maps = self.depth_contact_maps,
                                                              include_mesh_depth_contact = self.CTRL_PNL['depth_map_labels'])

        self.train_x_tensor = torch.Tensor(train_xa)

        self.train_y_flat = []  # Initialize the training ground truth list
        self.train_y_flat = TensorPrepLib().prep_labels(self.train_y_flat, dat_f_synth, num_repeats = 1,
                                                        z_adj = -0.075, gender = "f", is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        self.train_y_flat = TensorPrepLib().prep_labels(self.train_y_flat, dat_m_synth, num_repeats = 1,
                                                        z_adj = -0.075, gender = "m", is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])

        self.train_y_flat = TensorPrepLib().prep_labels(self.train_y_flat, dat_f_real, num_repeats = repeat_real_data_ct,
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        self.train_y_flat = TensorPrepLib().prep_labels(self.train_y_flat, dat_m_real, num_repeats = repeat_real_data_ct,
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        self.train_y_tensor = torch.Tensor(self.train_y_flat)

        print self.train_x_tensor.shape, 'Input training tensor shape'
        print self.train_y_tensor.shape, 'Output training tensor shape'




        #################################### PREP TESTING DATA ##########################################
        # load in the test file
        test_dat_f = TensorPrepLib().load_files_to_database(testing_database_file_f, 'real')
        test_dat_m = TensorPrepLib().load_files_to_database(testing_database_file_m, 'real')

        self.test_x_flat = []  # Initialize the testing pressure mat list
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f, test_dat_m, num_repeats = 1)
        self.test_x_flat = PreprocessingLib().preprocessing_blur_images(self.test_x_flat, self.mat_size, sigma=0.5)

        if len(self.test_x_flat) == 0: print("NO TESTING DATA INCLUDED")

        self.test_a_flat = []  # Initialize the testing pressure mat angle listhave
        self.test_a_flat = TensorPrepLib().prep_angles(self.test_a_flat, test_dat_f, test_dat_m, num_repeats = 1)

        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.depth_contact_maps_input_est = [] #Initialize the precomputed depth and contact map input estimates
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(self.depth_contact_maps_input_est,
                                                                                             test_dat_f, test_dat_m, num_repeats = 1)
        else:
            self.depth_contact_maps_input_est = None

        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat,
                                                                               self.test_a_flat,
                                                                               self.CTRL_PNL['incl_inter'], self.mat_size,
                                                                               self.CTRL_PNL['verbose'])

        test_xa = TensorPrepLib().append_input_depth_contact(np.array(test_xa),
                                                              include_pmat_contact = self.CTRL_PNL['incl_pmat_cntct_input'],
                                                              mesh_depth_contact_maps_input_est = self.depth_contact_maps_input_est,
                                                              include_mesh_depth_contact_input_est = self.CTRL_PNL['depth_map_input_est'],
                                                              mesh_depth_contact_maps = None,
                                                              include_mesh_depth_contact = False)

        self.test_x_tensor = torch.Tensor(test_xa)

        self.test_y_flat = []  # Initialize the ground truth listhave
        self.test_y_flat = TensorPrepLib().prep_labels(self.test_y_flat, test_dat_f, num_repeats = 1,
                                                        z_adj = 0.0, gender = "f", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        self.test_y_flat = TensorPrepLib().prep_labels(self.test_y_flat, test_dat_m, num_repeats = 1,
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        self.test_y_tensor = torch.Tensor(self.test_y_flat)


        print self.test_x_tensor.shape, 'Input testing tensor shape'
        print self.test_y_tensor.shape, 'Output testing tensor shape'




    def init_convnet_train(self):

        #self.train_x_tensor = self.train_x_tensor.unsqueeze(1)
        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])

        print "Loading convnet model................................"
        if self.CTRL_PNL['loss_vector_type'] == 'direct':
            fc_output_size = 30
            self.model = convnet.CNN(fc_output_size, self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'],
                                     verts_list = self.verts_list, filepath=self.CTRL_PNL['filepath_prefix'],
                                     in_channels=self.CTRL_PNL['num_input_channels'])

        elif self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':
            fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations
            self.model = convnet.CNN(fc_output_size, self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'],
                                     verts_list = self.verts_list, filepath=self.CTRL_PNL['filepath_prefix'], in_channels=self.CTRL_PNL['num_input_channels'])

            #self.model = torch.load(self.CTRL_PNL['filepath_prefix']+'data/convnets/convnet_anglesEU_synthreal_tanh_s4ang_sig0p5_5xreal_voloff_128b_200e.pt', map_location='cpu')

            #self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'], filepath=filepath_prefix)
            #self.model = torch.load('/home/ubuntu/Autobed_OFFICIAL_Trials' + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_'+str(self.CTRL_PNL['loss_vector_type'])+'_sTrue_128b_200e_' + str(self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)


        pp = 0
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print 'LOADED. num params: ', pp


        # Run model on GPU if available
        #if torch.cuda.is_available():
        if GPU == True:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.0005) #start with .00005

        # train the model one epoch at a time
        for epoch in range(1, self.CTRL_PNL['num_epochs'] + 1):
            self.t1 = time.time()
            self.train_convnet(epoch)

            try:
                self.t2 = time.time() - self.t1
            except:
                self.t2 = 0
            print 'Time taken by epoch',epoch,':',self.t2,' seconds'

            break
            #if epoch == 25 or epoch == 50 or epoch == 100 or epoch == 200 or epoch == 300:
            #    torch.save(self.model, filepath_prefix+'synth/convnet'+self.save_name+'_'+str(epoch)+'e.pt')
            #    pkl.dump(self.train_val_losses,open(filepath_prefix+'synth/convnet_losses'+self.save_name+'_'+str(epoch)+'e.p', 'wb'))


        print 'done with epochs, now evaluating'
        #self.validate_convnet('test')

        print self.train_val_losses, 'trainval'
        # Save the model (architecture and weights)




    def train_convnet(self, epoch):
        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.
        self.model.train()
        with torch.autograd.set_detect_anomaly(True):

            # This will loop a total = training_images/batch_size times
            for batch_idx, batch in enumerate(self.train_loader):

                if self.CTRL_PNL['loss_vector_type'] == 'direct':

                    self.optimizer.zero_grad()
                    scores, INPUT_DICT, OUTPUT_DICT = \
                        UnpackBatchLib().unpackage_batch_dir_pass(batch, is_training=True, model=self.model, CTRL_PNL = self.CTRL_PNL)

                    self.criterion = nn.L1Loss()
                    scores_zeros = np.zeros((batch[0].numpy().shape[0], 10))  # 24 is joint euclidean errors
                    scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))

                    loss = self.criterion(scores, scores_zeros)



                elif self.CTRL_PNL['loss_vector_type'] == 'anglesR' or self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':

                    self.optimizer.zero_grad()
                    scores, INPUT_DICT, OUTPUT_DICT = \
                        UnpackBatchLib().unpackage_batch_kin_pass(batch, is_training=True, model = self.model, CTRL_PNL=self.CTRL_PNL)

                    self.criterion = nn.L1Loss()
                    scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype),
                                            requires_grad=True)

                    loss_betas = self.criterion(scores[:, 0:10], scores_zeros[:, 0:10])*self.weight_joints
                    loss_eucl = self.criterion(scores[:, 10:34], scores_zeros[:, 10:34])*self.weight_joints*2

                    if self.CTRL_PNL['regr_angles'] == True:
                        loss_angs = self.criterion(scores[:, 34:106], scores_zeros[:, 34:106])*self.weight_joints
                        loss = (loss_betas + loss_eucl + loss_angs)
                    else:
                        loss = (loss_betas + loss_eucl)


                    if self.CTRL_PNL['depth_map_labels'] == True:
                        INPUT_DICT['batch_mdm'][INPUT_DICT['batch_mdm'] > 0] = 0
                        OUTPUT_DICT['batch_mdm_est'][OUTPUT_DICT['batch_mdm_est'] > 0] = 0
                        loss_mesh_depth = self.criterion(INPUT_DICT['batch_mdm'], OUTPUT_DICT['batch_mdm_est'])*self.weight_depth_planes / 10
                        loss_mesh_contact = self.criterion(INPUT_DICT['batch_cm'], OUTPUT_DICT['batch_cm_est'])*self.weight_depth_planes * 2.0
                        loss += loss_mesh_depth
                        loss += loss_mesh_contact


                loss.backward()
                self.optimizer.step()
                loss *= 1000

                if batch_idx % opt.log_interval == 0:
                    val_n_batches = 1
                    print "evaluating on ", val_n_batches

                    if GPU == True:
                        VisualizationLib().print_error_train(INPUT_DICT['batch_targets'].cpu(), OUTPUT_DICT['batch_targets_est'].cpu(),
                                                             self.output_size_train, self.CTRL_PNL['loss_vector_type'],
                                                             data='train')
                    else:
                        VisualizationLib().print_error_train(INPUT_DICT['batch_targets'], OUTPUT_DICT['batch_targets_est'],
                                                             self.output_size_train, self.CTRL_PNL['loss_vector_type'],
                                                             data='train')

                    print INPUT_DICT['batch_images'].shape
                    self.im_sample = INPUT_DICT['batch_images'][0, 1:, :].squeeze()
                    self.im_sample_ext = INPUT_DICT['batch_mdm'][0, :, :].squeeze().unsqueeze(0)*-1
                    self.im_sample_ext2 = OUTPUT_DICT['batch_mdm_est'][0, :, :].squeeze().unsqueeze(0)*-1
                    self.tar_sample = INPUT_DICT['batch_targets']
                    self.tar_sample = self.tar_sample[0, :].squeeze() / 1000
                    self.sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
                    self.sc_sample = self.sc_sample[0, :].squeeze() / 1000
                    self.sc_sample = self.sc_sample.view(self.output_size_train)

                    val_loss = self.validate_convnet(n_batches=val_n_batches)
                    train_loss = loss.data.item()
                    examples_this_epoch = batch_idx * len(INPUT_DICT['batch_images'])
                    epoch_progress = 100. * batch_idx / len(self.train_loader)

                    print_text_list = [ 'Train Epoch: {} ',
                                        '[{}',
                                        '/{} ',
                                        '({:.0f}%)]\t']
                    print_vals_list = [epoch,
                                      examples_this_epoch,
                                      len(self.train_loader.dataset),
                                      epoch_progress]
                    if self.CTRL_PNL['loss_vector_type'] == 'anglesR' or self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':
                        print_text_list.append('Train Loss Joints: {:.2f}')
                        print_vals_list.append(1000*loss_eucl.data)
                        print_text_list.append('\n\t\t\t\t\t\t   Betas Loss: {:.2f}')
                        print_vals_list.append(1000*loss_betas.data)
                        if self.CTRL_PNL['regr_angles'] == True:
                            print_text_list.append('\n\t\t\t\t\t\t  Angles Loss: {:.2f}')
                            print_vals_list.append(1000*loss_angs.data)
                        if self.CTRL_PNL['depth_map_labels'] == True:
                            print_text_list.append('\n\t\t\t\t\t\t   Mesh Depth: {:.2f}')
                            print_vals_list.append(1000*loss_mesh_depth.data)
                            print_text_list.append('\n\t\t\t\t\t\t Mesh Contact: {:.2f}')
                            print_vals_list.append(1000*loss_mesh_contact.data)

                    print_text_list.append('\n\t\t\t\t\t\t   Total Loss: {:.2f}')
                    print_vals_list.append(train_loss)
                    print_text_list.append('\n\t\t\t\t   Val Loss Total: {:.2f}')
                    print_vals_list.append(val_loss)

                    print_text = ''
                    for item in print_text_list:
                        print_text += item
                    print(print_text.format(*print_vals_list))


                    print 'appending to alldata losses'
                    self.train_val_losses['train' + self.save_name].append(train_loss)
                    self.train_val_losses['val' + self.save_name].append(val_loss)
                    self.train_val_losses['epoch' + self.save_name].append(epoch)


    def validate_convnet(self, verbose=False, n_batches=None):

        self.model.eval()
        loss = 0.
        n_examples = 0
        batch_ct = 0
        for batch_i, batch in enumerate(self.test_loader):

            self.model.eval()

            if self.CTRL_PNL['loss_vector_type'] == 'direct':
                scores, INPUT_DICT_VAL, OUTPUT_DICT_VAL = \
                    UnpackBatchLib().unpackage_batch_dir_pass(batch, is_training=False, model=self.model, CTRL_PNL = self.CTRL_PNL)
                self.criterion = nn.L1Loss()
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[1].shape[0], scores.size()[1]))).type(dtype),
                                        requires_grad=False)
                loss += self.criterion(scores, scores_zeros).data.item()

            elif self.CTRL_PNL['loss_vector_type'] == 'anglesR' or self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':
                scores, INPUT_DICT_VAL, OUTPUT_DICT_VAL = \
                    UnpackBatchLib().unpackage_batch_kin_pass(batch, is_training=False, model=self.model, CTRL_PNL=self.CTRL_PNL)
                self.criterion = nn.L1Loss()
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype),
                                        requires_grad=False)
                loss += self.criterion(scores[:, 10:34], scores_zeros[:, 10:34]).data.item() / 10.

            n_examples += self.CTRL_PNL['batch_size']

            if n_batches and (batch_i >= n_batches):
                break

            batch_ct += 1

        loss /= batch_ct
        loss *= 1000
        loss *= 10. / 34

        if GPU == True:
            VisualizationLib().print_error_val(INPUT_DICT_VAL['batch_targets'].cpu(), OUTPUT_DICT_VAL['batch_targets_est'].cpu(), self.output_size_val,
                                               self.CTRL_PNL['loss_vector_type'], data='validate')
        else:
            VisualizationLib().print_error_val(INPUT_DICT_VAL['batch_targets'], OUTPUT_DICT_VAL['batch_targets_est'], self.output_size_val,
                                               self.CTRL_PNL['loss_vector_type'], data='validate')


        self.im_sample_val = INPUT_DICT_VAL['batch_images']
        self.im_sample_val = self.im_sample_val[0, 1:, :].squeeze()
        self.tar_sample_val = INPUT_DICT_VAL['batch_targets']  # this is just 10 x 3
        self.tar_sample_val = self.tar_sample_val[0, :].squeeze() / 1000
        self.sc_sample_val = OUTPUT_DICT_VAL['batch_targets_est']  # score space is larger is 72 x3
        self.sc_sample_val = self.sc_sample_val[0, :].squeeze() / 1000
        self.sc_sample_val = self.sc_sample_val.view(24, 3)

        print self.im_sample.shape, self.im_sample_val.shape, self.im_sample_ext.shape, self.im_sample_ext2.shape

        if self.opt.visualize == True:
            if GPU == True:
                VisualizationLib().visualize_pressure_map(self.im_sample.cpu(), self.tar_sample.cpu(), self.sc_sample.cpu(),
                                                          self.im_sample_val.cpu(), self.tar_sample.cpu(), self.sc_sample.cpu(),
                                                          self.im_sample_ext.cpu(),
                                                          block=False)
            else:
                VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,
                                                          self.im_sample_ext, None, None,
                                                          self.im_sample_ext2, None, None,
                                                          self.im_sample_val, self.tar_sample_val, self.sc_sample_val,
                                                          block=False)

        return loss



if __name__ == "__main__":
    #Initialize trainer with a training database file

    #import rospy

    #rospy.init_node('pose_trainer')

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
    p.add_option('--j_d_ratio', action='store', type = 'float',
                 dest='j_d_ratio', \
                 default=0.5, \
                 help='Set the loss mix: joints to depth planes.')
    p.add_option('--qt', action='store_true',
                 dest='quick_test', \
                 default=False, \
                 help='Do a quick test.')
    p.add_option('--viz', action='store_true',
                 dest='visualize', \
                 default=False, \
                 help='Visualize.')
    p.add_option('--aws', action='store_true',
                 dest='aws', \
                 default=False, \
                 help='Use ubuntu user dir instead of henry.')
    p.add_option('--rgangs', action='store_true',
                 dest='reg_angles', \
                 default=False, \
                 help='Regress the angles as well as betas and joint pos.')
    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=True, help='Printout everything (under construction).')

    p.add_option('--log_interval', type=int, default=10, metavar='N',
                 help='number of batches between logging train status')

    opt, args = p.parse_args()


    if opt.aws == True:
        filepath_prefix = '/home/ubuntu/data/'
        filepath_suffix = ''
    else:
        filepath_prefix = '/home/henry/data/'
        filepath_suffix = ''

    filepath_prefix = '/media/henry/multimodal_data_2/data/'
    filepath_suffix = '_outputB'

    training_database_file_f = []
    training_database_file_m = []
    test_database_file_f = []
    test_database_file_m = [] #141 total training loss at epoch 9



    if opt.quick_test == True:
        training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_lay_2000_of_2047_lowerbody_stiff'+filepath_suffix+'.p')
        #training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_sit_1000_of_1121_upperbody_stiff'+filepath_suffix+'.p')
        #training_database_file_f.append(filepath_prefix+'real/trainval4_150rh1_sit120rh'+filepath_suffix+'.p')
        #training_database_file_m.append(filepath_prefix+'real/trainval4_150rh1_sit120rh'+filepath_suffix+'.p')
        #training_database_file_f.append(filepath_prefix + 'real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
        #training_database_file_f.append(filepath_prefix + 'real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
        #training_database_file_m.append(filepath_prefix+'real/s3_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
        #training_database_file_m.append(filepath_prefix+'real/s5_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
        test_database_file_m.append(filepath_prefix+'real/s4_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
        #training_database_file_f.append(filepath_prefix+'real/trainval4_150rh1_sit120rh'+filepath_suffix+'.p')
        #test_database_file_m.append(filepath_prefix+'real/trainval4_150rh1_sit120rh'+filepath_suffix+'.p')
    else:
        network_design = True
        if network_design == True:
            incl_synth = True
            incl_real = True
            if incl_synth == True:
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_lay_2000_of_2103_upperbody_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_lay_2000_of_2086_rightside_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_lay_2000_of_2072_leftside_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_lay_2000_of_2047_lowerbody_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_lay_2000_of_2067_none_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_sit_1000_of_1121_upperbody_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_sit_1000_of_1087_rightside_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_sit_1000_of_1102_leftside_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_sit_1000_of_1106_lowerbody_stiff'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'synth/side_up_fw/train_f_sit_1000_of_1096_none_stiff'+filepath_suffix+'.p')
            if incl_real == True:
                training_database_file_f.append(filepath_prefix+'real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
                training_database_file_f.append(filepath_prefix+'real/s8_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
                    
            #training_database_file_f.append(filepath_prefix+'real/trainval8_150rh1_sit120rh'+filepath_suffix+'.p')
            if incl_synth == True:
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_lay_2000_of_2031_upperbody_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_lay_2000_of_2016_rightside_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_lay_2000_of_2016_leftside_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_lay_2000_of_2012_lowerbody_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_lay_2000_of_2006_none_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_sit_1000_of_1147_upperbody_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_sit_1000_of_1132_rightside_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_sit_1000_of_1152_leftside_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_sit_1000_of_1144_lowerbody_stiff'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'synth/side_up_fw/train_m_sit_1000_of_1126_none_stiff'+filepath_suffix+'.p')
            if incl_real == True:
                training_database_file_m.append(filepath_prefix+'real/s3_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
                #training_database_file_m.append(filepath_prefix+'real/s4_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'real/s5_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'real/s6_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
                training_database_file_m.append(filepath_prefix+'real/s7_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')

            #training_database_file_m.append(filepath_prefix+'real/trainval4_150rh1_sit120rh'+filepath_suffix+'.p')
            #training_database_file_m.append(filepath_prefix+'synth/train_m_sit_95_rightside_stiff'+filepath_suffix+'.p')

            #test_database_file_f.append(filepath_prefix+'real/trainval4_150rh1_sit120rh'+filepath_suffix+'.p')
            #test_database_file_m.append(filepath_prefix+'real/trainval8_150rh1_sit120rh'+filepath_suffix+'.p')
            #test_database_file_f.append(filepath_prefix + 'real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
            #test_database_file_m.append(filepath_prefix + 'real/s3_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
            test_database_file_m.append(filepath_prefix + 'real/s4_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
            #test_database_file_m.append(filepath_prefix + 'real/s5_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
            #test_database_file_m.append(filepath_prefix + 'real/s6_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
            #test_database_file_m.append(filepath_prefix + 'real/s7_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
            #test_database_file_f.append(filepath_prefix + 'real/s8_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll'+filepath_suffix+'.p')
        else:
            
            training_database_file_f.append(filepath_prefix+'real/subject_12/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_f.append(filepath_prefix+'real/subject_12/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')
            training_database_file_f.append(filepath_prefix+'real/subject_16/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_f.append(filepath_prefix+'real/subject_16/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')
            training_database_file_f.append(filepath_prefix+'real/subject_17/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_f.append(filepath_prefix+'real/subject_17/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')
            training_database_file_f.append(filepath_prefix+'real/subject_18/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_f.append(filepath_prefix+'real/subject_18/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')

            #training_database_file_m.append(filepath_prefix+'real/subject_9/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            #training_database_file_m.append(filepath_prefix+'real/subject_9/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_10/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_10/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_11/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_11/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_13/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_13/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_14/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_14/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_15/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            training_database_file_m.append(filepath_prefix+'real/subject_15/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')

            
            test_database_file_m.append(filepath_prefix+'real/subject_9/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair'+filepath_suffix+'.p')
            test_database_file_m.append(filepath_prefix+'real/subject_9/p_files/trainval_sit175rlh_sit120rll'+filepath_suffix+'.p')


    p = PhysicalTrainer(training_database_file_f, training_database_file_m, test_database_file_f, test_database_file_m, opt)

    p.init_convnet_train()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
