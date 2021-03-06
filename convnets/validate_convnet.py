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
from tensorprep_lib import TensorPrepLib
from unpack_batch_lib import UnpackBatchLib
from gauss_fit_lib import GaussFitLib


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

DROPOUT = False

torch.set_num_threads(1)
if torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    print'######################### CUDA is available! #############################'
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    print'############################## USING CPU #################################'


class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''


    def __init__(self, testing_database_file_f, testing_database_file_m, opt):

        '''Opens the specified pickle files to get the combined dataset:
                This dataset is a dictionary of pressure maps with the corresponding
                3d position and orientation of the markers associated with it.'''

        # change this to 'direct' when you are doing baseline methods

        self.CTRL_PNL = {}
        self.CTRL_PNL['loss_vector_type'] = opt.losstype

        self.CTRL_PNL['verbose'] = opt.verbose
        self.opt = opt
        self.CTRL_PNL['batch_size'] = 1
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
        self.CTRL_PNL['depth_map_labels'] = True  # can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['depth_map_labels_test'] = True  # can only be true is we have 100% synth for testing
        self.CTRL_PNL['depth_map_output'] = self.CTRL_PNL['depth_map_labels']
        self.CTRL_PNL['depth_map_input_est'] = False  # do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL[
            'depth_map_input_est']  # holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True
        self.CTRL_PNL['normalize_input'] = True
        self.CTRL_PNL['all_tanh_activ'] = True
        self.CTRL_PNL['L2_contact'] = True
        self.CTRL_PNL['pmat_mult'] = int(5)
        self.CTRL_PNL['cal_noise'] = False


        j_d_ratio = 0.5
        self.weight_joints = j_d_ratio*2
        self.weight_depth_planes = (1-j_d_ratio)*2


        if opt.losstype == 'direct':
            self.CTRL_PNL['depth_map_labels'] = False
            self.CTRL_PNL['depth_map_output'] = False

        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['pmat_mult'] = int(1)
            self.CTRL_PNL['incl_pmat_cntct_input'] = False  # if there's calibration noise we need to recompute this every batch
            self.CTRL_PNL['clip_sobel'] = False

        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['depth_map_input_est'] == True:  # for a two part regression
            self.CTRL_PNL['num_input_channels'] += 3
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.CTRL_PNL['num_input_channels'] += 2
        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['num_input_channels'] += 1

        self.CTRL_PNL['filepath_prefix'] = '/home/henry/'
            # self.CTRL_PNL['filepath_prefix'] = '/media/henry/multimodal_data_2/'

        if self.CTRL_PNL['depth_map_output'] == True:  # we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

        print self.CTRL_PNL['num_epochs'], 'NUM EPOCHS!'
        # Entire pressure dataset with coordinates in world frame

        self.save_name = '_' + opt.losstype + \
                         '_synth_32000' + \
                         '_' + str(self.CTRL_PNL['batch_size']) + 'b' + \
                         '_' + str(self.CTRL_PNL['num_epochs']) + 'e' + \
                         '_x' + str(self.CTRL_PNL['pmat_mult']) + 'pmult'

        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.save_name += '_depthestin'
        if self.CTRL_PNL['adjust_ang_from_est'] == True:
            self.save_name += '_angleadj'
        if self.CTRL_PNL['all_tanh_activ'] == True:
            self.save_name += '_alltanh'
        if self.CTRL_PNL['L2_contact'] == True:
            self.save_name += '_l2cnt'
        if self.CTRL_PNL['cal_noise'] == True:
            self.save_name += '_calnoise'

        # self.save_name = '_' + opt.losstype+'_real_s9_alltest_' + str(self.CTRL_PNL['batch_size']) + 'b_'# + str(self.CTRL_PNL['num_epochs']) + 'e'

        print 'appending to', 'train' + self.save_name
        self.train_val_losses = {}
        self.train_val_losses['train' + self.save_name] = []
        self.train_val_losses['val' + self.save_name] = []
        self.train_val_losses['epoch' + self.save_name] = []

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array(
            [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)


        #################################### PREP TESTING DATA ##########################################
        # load in the test file
        test_dat_f_synth = TensorPrepLib().load_files_to_database(testing_database_file_f, 'synth')
        test_dat_m_synth = TensorPrepLib().load_files_to_database(testing_database_file_m, 'synth')
        test_dat_f_real = TensorPrepLib().load_files_to_database(testing_database_file_f, 'real')
        test_dat_m_real = TensorPrepLib().load_files_to_database(testing_database_file_m, 'real')

        self.test_x_flat = []  # Initialize the testing pressure mat list
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f_synth, test_dat_m_synth,
                                                       num_repeats=1)
        self.test_x_flat = list(
            np.clip(np.array(self.test_x_flat) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100))
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f_real, test_dat_m_real,
                                                       num_repeats=1)

        if self.CTRL_PNL['cal_noise'] == False:
            self.test_x_flat = PreprocessingLib().preprocessing_blur_images(self.test_x_flat, self.mat_size, sigma=0.5)

        if len(self.test_x_flat) == 0: print("NO TESTING DATA INCLUDED")

        self.test_a_flat = []  # Initialize the testing pressure mat angle listhave
        self.test_a_flat = TensorPrepLib().prep_angles(self.test_a_flat, test_dat_f_synth, test_dat_m_synth,
                                                       num_repeats=1)
        self.test_a_flat = TensorPrepLib().prep_angles(self.test_a_flat, test_dat_f_real, test_dat_m_real,
                                                       num_repeats=1)

        if self.CTRL_PNL['depth_map_labels_test'] == True:
            self.depth_contact_maps = []  # Initialize the precomputed depth and contact maps. only synth has this label.
            self.depth_contact_maps = TensorPrepLib().prep_depth_contact(self.depth_contact_maps, test_dat_f_synth,
                                                                         test_dat_m_synth, num_repeats=1)
        else:
            self.depth_contact_maps = None

        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.depth_contact_maps_input_est = []  # Initialize the precomputed depth and contact map input estimates
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(
                self.depth_contact_maps_input_est,
                test_dat_f_synth, test_dat_m_synth, num_repeats=1)
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(
                self.depth_contact_maps_input_est,
                test_dat_f_real, test_dat_m_real, num_repeats=1)
        else:
            self.depth_contact_maps_input_est = None

        print np.shape(self.test_x_flat), np.shape(self.test_a_flat)

        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat,
                                                                               self.test_a_flat,
                                                                               self.mat_size,
                                                                               self.CTRL_PNL)

        test_xa = TensorPrepLib().append_input_depth_contact(np.array(test_xa),
                                                             CTRL_PNL=self.CTRL_PNL,
                                                             mesh_depth_contact_maps_input_est=self.depth_contact_maps_input_est,
                                                             mesh_depth_contact_maps=self.depth_contact_maps)

        # normalize the input
        if self.CTRL_PNL['normalize_input'] == True:
            test_xa = TensorPrepLib().normalize_network_input(test_xa, self.CTRL_PNL)

        self.test_x_tensor = torch.Tensor(test_xa)

        test_y_flat = []  # Initialize the ground truth listhave

        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_f_synth, num_repeats=1,
                                                  z_adj=-0.075, gender="f", is_synth=True,
                                                  loss_vector_type=self.CTRL_PNL['loss_vector_type'],
                                                  initial_angle_est=self.CTRL_PNL['adjust_ang_from_est'])
        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_m_synth, num_repeats=1,
                                                  z_adj=-0.075, gender="m", is_synth=True,
                                                  loss_vector_type=self.CTRL_PNL['loss_vector_type'],
                                                  initial_angle_est=self.CTRL_PNL['adjust_ang_from_est'])

        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_f_real, num_repeats=1,
                                                  z_adj=0.0, gender="f", is_synth=False,
                                                  loss_vector_type=self.CTRL_PNL['loss_vector_type'],
                                                  initial_angle_est=self.CTRL_PNL['adjust_ang_from_est'])
        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_m_real, num_repeats=1,
                                                  z_adj=0.0, gender="m", is_synth=False,
                                                  loss_vector_type=self.CTRL_PNL['loss_vector_type'],
                                                  initial_angle_est=self.CTRL_PNL['adjust_ang_from_est'])

        if self.CTRL_PNL['normalize_input'] == True:
            test_y_flat = TensorPrepLib().normalize_wt_ht(test_y_flat)

        self.test_y_tensor = torch.Tensor(test_y_flat)

        print self.test_x_tensor.shape, 'Input testing tensor shape'
        print self.test_y_tensor.shape, 'Output testing tensor shape'


    def visualize_3d_data(self):

        p_mat_array = self.test_x_tensor.numpy()[:, 0, :, :]
        bedangle_array = self.test_x_tensor.numpy()[:, 2, 0, 0]

        print self.test_y_tensor.size(), self.test_y_tensor.numpy()[:, 0:72].shape

        joint_loc_array = self.test_y_tensor.numpy()[:, 0:72].reshape(-1, 24, 3)/1000
        root_loc_array = self.test_y_tensor.numpy()[:, 154:157]

        import rospy
        rospy.init_node('real_time_pose')

        print p_mat_array.shape
        print joint_loc_array.shape

        VizLib = VisualizationLib()
        for ct in range(p_mat_array.shape[0]):


            VizLib.rviz_publish_input(p_mat_array[ct, :, :], bedangle_array[ct])
            VizLib.rviz_publish_output(joint_loc_array[ct])

            #print bedangle_array[ct], joint_loc_array[ct, 0, :], root_loc_array[ct, :]

            time.sleep(1.0)

    def init_smpl(self, batch_size):

        from smpl.smpl_webuser.serialization import load_model

        model_path_f = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        human_f = load_model(model_path_f)
        self.v_template_f = torch.Tensor(np.array(human_f.v_template)).type(dtype)
        self.shapedirs_f = torch.Tensor(np.array(human_f.shapedirs)).permute(0, 2, 1).type(dtype)
        self.J_regressor_f = np.zeros((human_f.J_regressor.shape)) + human_f.J_regressor
        self.J_regressor_f = torch.Tensor(np.array(self.J_regressor_f).astype(float)).permute(1, 0).type(dtype)
        self.posedirs_f = torch.Tensor(np.array(human_f.posedirs))
        self.weights_f = torch.Tensor(np.array(human_f.weights))

        model_path_m = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        human_m = load_model(model_path_m)
        self.v_template_m = torch.Tensor(np.array(human_m.v_template)).type(dtype)
        self.shapedirs_m = torch.Tensor(np.array(human_m.shapedirs)).permute(0, 2, 1).type(dtype)
        self.J_regressor_m = np.zeros((human_m.J_regressor.shape)) + human_m.J_regressor
        self.J_regressor_m = torch.Tensor(np.array(self.J_regressor_m).astype(float)).permute(1, 0).type(dtype)
        self.posedirs_m = torch.Tensor(np.array(human_m.posedirs))
        self.weights_m = torch.Tensor(np.array(human_m.weights))

        print self.posedirs_m.size()

        self.parents = np.array(
            [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)

        # print batch_size
        self.N = batch_size
        self.shapedirs_repeat_f = self.shapedirs_f.unsqueeze(0).repeat(self.N, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
        self.shapedirs_repeat_m = self.shapedirs_m.unsqueeze(0).repeat(self.N, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
        self.shapedirs_repeat = torch.cat((self.shapedirs_repeat_f, self.shapedirs_repeat_m), 0)  # this is 2 x N x B x R x D
        self.B = self.shapedirs_repeat.size()[2]  # this is 10
        self.R = self.shapedirs_repeat.size()[3]  # this is 6890, or num of verts
        self.D = self.shapedirs_repeat.size()[4]  # this is 3, or num dimensions
        self.shapedirs_repeat = self.shapedirs_repeat.permute(1, 0, 2, 3, 4).view(self.N, 2, self.B * self.R * self.D)

        self.v_template_repeat_f = self.v_template_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.v_template_repeat_m = self.v_template_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.v_template_repeat = torch.cat((self.v_template_repeat_f, self.v_template_repeat_m), 0)  # this is 2 x N x R x D
        self.v_template_repeat = self.v_template_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R * self.D)

        self.J_regressor_repeat_f = self.J_regressor_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.J_regressor_repeat_m = self.J_regressor_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.J_regressor_repeat = torch.cat((self.J_regressor_repeat_f, self.J_regressor_repeat_m), 0)  # this is 2 x N x R x 24
        self.J_regressor_repeat = self.J_regressor_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R * 24)

        self.posedirs_repeat_f = self.posedirs_f.unsqueeze(0).repeat(self.N, 1, 1, 1).unsqueeze(0)
        self.posedirs_repeat_m = self.posedirs_m.unsqueeze(0).repeat(self.N, 1, 1, 1).unsqueeze(0)
        self.posedirs_repeat = torch.cat((self.posedirs_repeat_f, self.posedirs_repeat_m), 0)
        #self.posedirs_repeat = self.posedirs_repeat.permute(1, 0, 2, 3, 4).view(self.N, 2, self.R*self.D*207)
        self.posedirs_repeat = self.posedirs_repeat.permute(1, 0, 2, 3, 4).view(self.N, 2, self.R, self.D*207)
        self.posedirs_repeat = torch.stack([self.posedirs_repeat[:, :, 1325, :],
                                            self.posedirs_repeat[:, :, 336, :],
                                            self.posedirs_repeat[:, :, 1032, :],
                                            self.posedirs_repeat[:, :, 4515, :],
                                            self.posedirs_repeat[:, :, 1374, :],
                                            self.posedirs_repeat[:, :, 4848, :],
                                            self.posedirs_repeat[:, :, 1739, :],
                                            self.posedirs_repeat[:, :, 5209, :],
                                            self.posedirs_repeat[:, :, 1960, :],
                                            self.posedirs_repeat[:, :, 5423, :]])


        self.posedirs_repeat = self.posedirs_repeat.permute(1, 2, 0, 3).contiguous().view(self.N, 2, 10*self.D*207)

        self.weights_repeat_f = self.weights_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.weights_repeat_m = self.weights_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.weights_repeat = torch.cat((self.weights_repeat_f, self.weights_repeat_m), 0)
        #self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R * 24)
        self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R, 24)
        self.weights_repeat = torch.stack([self.weights_repeat[:, :, 1325, :],
                                            self.weights_repeat[:, :, 336, :],
                                            self.weights_repeat[:, :, 1032, :],
                                            self.weights_repeat[:, :, 4515, :],
                                            self.weights_repeat[:, :, 1374, :],
                                            self.weights_repeat[:, :, 4848, :],
                                            self.weights_repeat[:, :, 1739, :],
                                            self.weights_repeat[:, :, 5209, :],
                                            self.weights_repeat[:, :, 1960, :],
                                            self.weights_repeat[:, :, 5423, :]])
        self.weights_repeat = self.weights_repeat.permute(1, 2, 0, 3).contiguous().view(self.N, 2, 10*24)



    def visualize_offset_3d_data(self):

        from kinematics_lib import KinematicsLib

        p_mat_array = self.test_x_tensor.numpy()[:, 0, :, :]
        bedangle_array = self.test_x_tensor.numpy()[:, 2, 0, 0]

        self.init_smpl(self.CTRL_PNL['batch_size'])

        print self.test_y_tensor.size(), self.test_y_tensor.numpy()[:, 0:72].shape

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])




        for batch_idx, batch in enumerate(self.test_loader):

            p_mat_array = batch[0].numpy()[:, 0, :, :]
            joint_loc_array = batch[1][:, 0:72].numpy().reshape(-1, 24, 3) / 1000
            betas_array = batch[1][:, 72:82]
            #betas_array[:, 5] = 2.0
            dir_cos_angles_array = batch[1][:, 82:154]

            root_loc_array = batch[1][:, 154:157]
            gender_switch = batch[1][:, 157:159]



            gender_switch = gender_switch.unsqueeze(1)
            batch_size = gender_switch.size()[0]


            print self.shapedirs_repeat.size(), self.posedirs_repeat.size(), batch_size

            shapedirs = torch.bmm(gender_switch, self.shapedirs_repeat[0:batch_size, :, :])\
                             .view(batch_size, self.B, self.R*self.D)

            betas_shapedirs_mult = torch.bmm(betas_array.unsqueeze(1), shapedirs)\
                                        .squeeze(1)\
                                        .view(batch_size, self.R, self.D)

            v_template = torch.bmm(gender_switch, self.v_template_repeat[0:batch_size, :, :])\
                              .view(batch_size, self.R, self.D)

            v_shaped = betas_shapedirs_mult + v_template

            J_regressor_repeat = torch.bmm(gender_switch, self.J_regressor_repeat[0:batch_size, :, :])\
                                      .view(batch_size, self.R, 24)

            Jx = torch.bmm(v_shaped[:, :, 0].unsqueeze(1), J_regressor_repeat).squeeze(1)
            Jy = torch.bmm(v_shaped[:, :, 1].unsqueeze(1), J_regressor_repeat).squeeze(1)
            Jz = torch.bmm(v_shaped[:, :, 2].unsqueeze(1), J_regressor_repeat).squeeze(1)


            J_est = torch.stack([Jx, Jy, Jz], dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
            #J_est = J_est - J_est[:, 0:1, :] + root_loc_array.unsqueeze(1)

            Rs_est = KinematicsLib().batch_rodrigues(dir_cos_angles_array.view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

            targets_est, A_est = KinematicsLib().batch_global_rigid_transformation(Rs_est, J_est, self.parents, self.GPU, rotate_base=False)
            targets_est = targets_est - J_est[:, 0:1, :] + root_loc_array.unsqueeze(1)

            '''
            #now we assemble the transformed mesh
            pose_feature = (Rs_est[:, 1:, :, :]).sub(1.0, torch.eye(3).float()).view(-1, 207)
            posedirs_repeat = torch.bmm(gender_switch, self.posedirs_repeat[0:batch_size, :, :])\
                                        .view(batch_size, self.R*self.D, 207)\
                                        .permute(0, 2, 1)
            v_posed = torch.bmm(pose_feature.unsqueeze(1), posedirs_repeat).view(-1, self.R, self.D) + v_shaped
            weights_repeat = torch.bmm(gender_switch, self.weights_repeat[0:batch_size, :, :])\
                                    .squeeze(1)\
                                    .view(batch_size, self.R, 24)
            T = torch.bmm(weights_repeat, A_est.view(batch_size, 24, 16)).view(batch_size, -1, 4, 4)
            v_posed_homo = torch.cat([v_posed, torch.ones(batch_size, v_posed.shape[1], 1)], dim=2)
            v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
            verts = v_homo[:, :, :3, 0] - J_est[:, 0:1, :] + root_loc_array.unsqueeze(1)
            '''

            #assemble a reduced form of the transformed mesh
            v_shaped_red = torch.stack([v_shaped[:, 1325, :],
                                        v_shaped[:, 336, :], #head
                                        v_shaped[:, 1032, :], #l knee
                                        v_shaped[:, 4515, :], #r knee
                                        v_shaped[:, 1374, :], #l ankle
                                        v_shaped[:, 4848, :], #r ankle
                                        v_shaped[:, 1739, :], #l elbow
                                        v_shaped[:, 5209, :], #r elbow
                                        v_shaped[:, 1960, :], #l wrist
                                        v_shaped[:, 5423, :]]).permute(1, 0, 2) #r wrist
            #now we assemble the transformed mesh
            pose_feature = (Rs_est[:, 1:, :, :]).sub(1.0, torch.eye(3).float()).view(-1, 207)
            posedirs_repeat = torch.bmm(gender_switch, self.posedirs_repeat[0:batch_size, :, :])\
                                        .view(batch_size, 10*self.D, 207)\
                                        .permute(0, 2, 1)
            v_posed = torch.bmm(pose_feature.unsqueeze(1), posedirs_repeat).view(-1, 10, self.D)
            v_posed = v_posed.clone() + v_shaped_red
            weights_repeat = torch.bmm(gender_switch, self.weights_repeat[0:batch_size, :, :])\
                                    .squeeze(1)\
                                    .view(batch_size, 10, 24)
            T = torch.bmm(weights_repeat, A_est.view(batch_size, 24, 16)).view(batch_size, -1, 4, 4)
            v_posed_homo = torch.cat([v_posed, torch.ones(batch_size, v_posed.shape[1], 1)], dim=2)
            v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
            verts = v_homo[:, :, :3, 0] - J_est[:, 0:1, :] + root_loc_array.unsqueeze(1)


            targets_est = targets_est.numpy()
            import rospy
            rospy.init_node('real_time_pose')


            VizLib = VisualizationLib()
            for ct in range(p_mat_array.shape[0]):

                print joint_loc_array[ct], 'all joints'
                print targets_est[ct, :]

                #VizLib.rviz_publish_input(p_mat_array[ct, :, :], bedangle_array[ct])
                #VizLib.rviz_publish_output_mesh(verts.numpy()[ct], verts.numpy()[ct], offset_targets)
                VizLib.rviz_publish_output_mesh(verts.numpy()[ct], verts.numpy()[ct])
                VizLib.rviz_publish_output(joint_loc_array[ct])

                #print bedangle_array[ct], joint_loc_array[ct, 0, :], root_loc_array[ct, :]

                time.sleep(5.0)


    def init_convnet_test(self):

        if self.CTRL_PNL['verbose']: print self.test_x_tensor.size(), 'length of the testing dataset'
        if self.CTRL_PNL['verbose']: print self.test_y_tensor.size(), 'size of the testing database output'


        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])



        if self.CTRL_PNL['loss_vector_type'] == 'direct':
            fc_output_size = 72
            if GPU == True:
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/1.5xsize/convnet_direct_real_s5_128b_300e.pt')
                #self.model = torch.load('/home/henry/data/convnets/convnet_direct_real_s2_128b_300e.pt')
                self.model = torch.load('/home/henry/data/synth/convnet_direct_real_s9_alltest_128b_300e.pt')
                self.model = self.model.cuda()
            else:
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/1.5xsize/convnet_direct_real_s5_128b_300e.pt', map_location='cpu')
                self.model = torch.load('/home/henry/data/convnets/convnet_direct_real_s9_alltest_128b_300e_noscale.pt', map_location='cpu')


        elif self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':
            fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations
            if GPU == True:
                #self.model = torch.load('/home/henry/data/synth/convnet_anglesEU_synth_planesreg_128b_100e.pt')
                #self.model = torch.load('/home/henry/data/convnets/epochs_set_3/convnet_anglesEU_synthreal_s12_3xreal_128b_101e_300e.pt')
                #self.model = torch.load('/home/henry/data/synth/convnet_anglesEU_synthreal_s4_3xreal_128b_200e.pt')
                #self.model = torch.load('/home/henry/data/synth/convnet_anglesEU_synthreal_s4_3xreal_128b_101e_htwt_legacy.pt')
                #self.model = torch.load('/home/henry/data/synth/convnet_anglesEU_synth_s9_3xreal_128b_101e_0.5rtojtdpth_pmatcntin_100e.pt')
                self.model = torch.load('/home/henry/data/convnets/planesreg/convnet_anglesEU_synth_s9_3xreal_128b_0.5rtojtdpth_pmatcntin_200e_00001lr.pt')
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/1.5xsize/convnet_anglesEU_synthreal_tanh_s4ang_sig0p5_5xreal_voloff_128b_300e.pt')
                self.model = self.model.cuda()
            else:
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/planesreg/convnet_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_100e_00001lr.pt', map_location = 'cpu')
                self.model = torch.load('/media/henry/multimodal_data_1/convnets/convnet_anglesDC_synth_32000_128b_201e_0.5rtojtdpth_alltanh_l2cnt_100e.pt', map_location = 'cpu')


            print 'LOADED anglesEU !!!!!!!!!!!!!!!!!'
            pp = 0
            for p in list(self.model.parameters()):
                nnn = 1
                for s in list(p.size()):
                    nnn = nnn * s
                pp += nnn
            print pp, 'num params'


        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()


        print 'done with epochs, now evaluating'
        self.validate_convnet('test')


    def validate_convnet(self, verbose=False, n_batches=None):

        if DROPOUT == True:
            self.model.train()
        else:
            self.model.eval()
        loss = 0.
        n_examples = 0

        for batch_i, batch in enumerate(self.test_loader):
            #if batch_i == 10: break
            if DROPOUT == True:
                batch[0] = batch[0].repeat(25, 1, 1, 1)
                batch[1] = batch[1].repeat(25, 1)
            #self.model.train()

            if self.CTRL_PNL['loss_vector_type'] == 'direct':
                scores, INPUT_DICT, OUTPUT_DICT = \
                    UnpackBatchLib().unpackage_batch_dir_pass(batch, is_training=False, model=self.model,
                                                              CTRL_PNL=self.CTRL_PNL)
                self.criterion = nn.L1Loss()
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[1].shape[0], scores.size()[1]))).type(dtype),
                                        requires_grad=False)
                loss += self.criterion(scores, scores_zeros).data.item()



            elif self.CTRL_PNL['loss_vector_type'] == 'anglesR' or self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':

                if self.CTRL_PNL['GPU'] == True: print('GPU memory val:', torch.cuda.max_memory_allocated())
                scores, INPUT_DICT_VAL, OUTPUT_DICT_VAL = \
                    UnpackBatchLib().unpackage_batch_kin_pass(batch, is_training=False, model=self.model, CTRL_PNL=self.CTRL_PNL)
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype),
                                        requires_grad=False)

                if self.CTRL_PNL['full_body_rot'] == True:
                    OSA = 6
                    loss_bodyrot = float(self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * self.weight_joints)
                    if self.CTRL_PNL['adjust_ang_from_est'] == True:
                        loss_bodyrot *= 0
                else: OSA = 0

                loss_eucl = float(self.criterion(scores[:, 10+OSA:34+OSA], scores_zeros[:,  10+OSA:34+OSA]) * self.weight_joints)
                loss_betas = float(self.criterion(scores[:, 0:10], scores_zeros[:, 0:10]) * self.weight_joints * 0.5)


                if self.CTRL_PNL['regr_angles'] == True:
                    loss_angs = float(self.criterion(scores[:, 34+OSA:106+OSA], scores_zeros[:, 34+OSA:106+OSA]) * self.weight_joints)
                    loss_to_add = (loss_betas + loss_eucl + loss_bodyrot + loss_angs)
                else:
                    loss_to_add = (loss_betas + loss_eucl + loss_bodyrot)

                # print INPUT_DICT_VAL['batch_mdm'].size(), OUTPUT_DICT_VAL['batch_mdm_est'].size()

                if self.CTRL_PNL['depth_map_labels'] == True:
                    INPUT_DICT_VAL['batch_mdm'][INPUT_DICT_VAL['batch_mdm'] > 0] = 0
                    if self.CTRL_PNL['mesh_bottom_dist'] == True:
                        OUTPUT_DICT_VAL['batch_mdm_est'][OUTPUT_DICT_VAL['batch_mdm_est'] > 0] = 0

                    if self.CTRL_PNL['L2_contact'] == True:
                        loss_mesh_depth = float(self.criterion2(INPUT_DICT_VAL['batch_mdm'],OUTPUT_DICT_VAL['batch_mdm_est']) * self.weight_depth_planes * (1. / 44.46155340000357) * (1. / 44.46155340000357))
                        loss_mesh_contact = float(self.criterion(INPUT_DICT_VAL['batch_cm'],OUTPUT_DICT_VAL['batch_cm_est']) * self.weight_depth_planes * (1. / 0.4428100696329912))
                    else:
                        loss_mesh_depth = float(self.criterion(INPUT_DICT_VAL['batch_mdm'],OUTPUT_DICT_VAL['batch_mdm_est']) * self.weight_depth_planes * (1. / 44.46155340000357))
                        loss_mesh_contact = float(self.criterion(INPUT_DICT_VAL['batch_cm'],OUTPUT_DICT_VAL['batch_cm_est']) * self.weight_depth_planes * (1. / 0.4428100696329912))
                    loss_to_add += loss_mesh_depth
                    loss_to_add += loss_mesh_contact




            n_examples += self.CTRL_PNL['batch_size']
            #print n_examples

            if n_batches and (batch_i >= n_batches):
                break


            try:
                targets_print = torch.cat([targets_print, torch.mean(INPUT_DICT_VAL['batch_targets'], dim = 0).unsqueeze(0)], dim=0)
                targets_est_print = torch.cat([targets_est_print, torch.mean(OUTPUT_DICT_VAL['batch_targets_est'], dim = 0).unsqueeze(0)], dim=0)
            except:

                targets_print = torch.mean(INPUT_DICT_VAL['batch_targets'], dim = 0).unsqueeze(0)
                targets_est_print = torch.mean(OUTPUT_DICT_VAL['batch_targets_est'], dim = 0).unsqueeze(0)


            print targets_print.shape, INPUT_DICT_VAL['batch_targets'].shape
            print targets_est_print.shape, OUTPUT_DICT_VAL['batch_targets_est'].shape


            if GPU == True:
                error_norm, error_avg, _ = VisualizationLib().print_error_val(targets_print[-2:-1,:].cpu(),
                                                                                   targets_est_print[-2:-1,:].cpu(),
                                                                                   self.output_size_val,
                                                                                   self.CTRL_PNL['loss_vector_type'],
                                                                                   data='validate')
            else:
                error_norm, error_avg, _ = VisualizationLib().print_error_val(targets_print[-1:,:],
                                                                              targets_est_print[-1:,:],
                                                                                   self.output_size_val,
                                                                                   self.CTRL_PNL['loss_vector_type'],
                                                                                   data='validate')


            if self.opt.visualize == True:
                loss /= n_examples
                loss *= 100
                loss *= 1000

                print INPUT_DICT_VAL['batch_images'].size()
                NUM_IMAGES = INPUT_DICT_VAL['batch_images'].size()[0]

                for image_ct in range(NUM_IMAGES):
                    # #self.im_sampleval = self.im_sampleval[:,0,:,:]
                    self.im_sampleval = INPUT_DICT_VAL['batch_images'][image_ct, 1:].squeeze()*26.
                    self.im_sample_ext = INPUT_DICT_VAL['batch_mdm'][0, :, :].squeeze().cpu().unsqueeze(0)*-1

                    #print self.im_sample_ext.size()
                    self.im_sample_ext2 = OUTPUT_DICT_VAL['batch_mdm_est'][0, :, :].squeeze().cpu().unsqueeze(0)*-1

                    self.tar_sampleval = INPUT_DICT_VAL['batch_targets'][image_ct, :].squeeze() / 1000
                    self.sc_sampleval = OUTPUT_DICT_VAL['batch_targets_est'][image_ct, :].squeeze() / 1000
                    self.sc_sampleval = self.sc_sampleval.view(24, 3)



                    #self.im_sample2 = GaussFitLib().get_pressure_under_legs(self.im_sampleval[0, :, :], np.copy(self.sc_sampleval))


                    #self.im_sample2 = OUTPUT_DICT_VAL['batch_mdm_est'].data[image_ct, :].squeeze()*-1


                    if GPU == True:
                        VisualizationLib().visualize_pressure_map(self.im_sampleval.cpu(), self.tar_sampleval.cpu(), self.sc_sampleval.cpu(),
                                                                    self.im_sampleval.cpu(), None, None, 
                                                                    self.im_sample_ext2, None, None, 
                                                                    #self.im_sample_ext, None, None,
                                                                     block=False)
                    else:
                        VisualizationLib().visualize_pressure_map(self.im_sampleval, self.tar_sampleval, self.sc_sampleval,
                                                                  self.im_sample_ext, self.tar_sampleval, self.sc_sampleval,
                                                                  self.im_sample_ext2, self.tar_sampleval, self.sc_sampleval,
                                                                  block=False)
                    time.sleep(1)

        if GPU == True:
            error_norm, error_avg, _ = VisualizationLib().print_error_iros2018(targets_print.cpu(), targets_est_print.cpu(), self.output_size_val, self.CTRL_PNL['loss_vector_type'], data='validate')
        else:
            error_norm, error_avg, _ = VisualizationLib().print_error_iros2018(targets_print, targets_est_print, self.output_size_val, self.CTRL_PNL['loss_vector_type'], data='validate')

        print "MEAN IS: ", np.mean(error_norm, axis=0)
        print error_avg, np.mean(error_avg)*10


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

    #filepath_prefix_qt = '/home/henry/'
    filepath_prefix_qt = '/media/henry/multimodal_data_1/'


    test_database_file_f = []
    test_database_file_m = []

    network_design = True

    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_lay_2000_of_2103_upperbody_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_lay_2000_of_2086_rightside_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_lay_2000_of_2072_leftside_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_lay_2000_of_2047_lowerbody_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_lay_2000_of_2067_none_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_sit_1000_of_1121_upperbody_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_sit_1000_of_1087_rightside_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_sit_1000_of_1102_leftside_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_sit_1000_of_1106_lowerbody_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_sit_1000_of_1096_none_stiff.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_f.append(filepath_prefix_qt+'data/real/s8_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')

    test_database_file_f.append(filepath_prefix_qt+'convnets/synth/test_rollpi_f_lay_1000_none_stiff.p')

    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_lay_2000_of_2031_upperbody_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_lay_2000_of_2016_rightside_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_lay_2000_of_2016_leftside_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_lay_2000_of_2012_lowerbody_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_lay_2000_of_2006_none_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_sit_1000_of_1147_upperbody_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_sit_1000_of_1132_rightside_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_sit_1000_of_1152_leftside_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_sit_1000_of_1144_lowerbody_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/synth/side_up_fw/train_m_sit_1000_of_1126_none_stiff.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/s3_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt+'data/real/s4_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/s5_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/s6_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/s7_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')

    # test_database_file_m.append(filepath_prefix_qt+'data/real/trainval4_150rh1_sit120rh.p')
    # test_database_file_m.append(filepath_prefix_qt+'data/synth/train_m_sit_95_rightside_stiff.p')

    # test_database_file_f.append(filepath_prefix_qt+'data/real/trainval4_150rh1_sit120rh.p')
    # test_database_file_m.append(filepath_prefix_qt+'data/real/trainval8_150rh1_sit120rh.p')
    # test_database_file_f.append(filepath_prefix_qt + 'data/real/s2_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    # test_database_file_m.append(filepath_prefix_qt + 'data/real/s3_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/s4_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    # test_database_file_m.append(filepath_prefix_qt + 'data/real/s5_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    # test_database_file_m.append(filepath_prefix_qt + 'data/real/s6_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    # test_database_file_m.append(filepath_prefix_qt + 'data/real/s7_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')
    # test_database_file_f.append(filepath_prefix_qt + 'data/real/s8_trainval_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p')

    #test_database_file_m.append(filepath_prefix_qt + 'data/real/trainval4_150rh1_sit120rh.p')
    #test_database_file_f.append(filepath_prefix_qt + 'data/real/subject_12/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_f.append(filepath_prefix_qt + 'data/real/subject_12/p_files/trainval_sit175rlh_sit120rll.p')
    #test_database_file_f.append(filepath_prefix_qt + 'data/real/subject_16/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_f.append(filepath_prefix_qt + 'data/real/subject_16/p_files/trainval_sit175rlh_sit120rll.p')
    #test_database_file_f.append(filepath_prefix_qt + 'data/real/subject_17/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_f.append(filepath_prefix_qt + 'data/real/subject_17/p_files/trainval_sit175rlh_sit120rll.p')
    #test_database_file_f.append(filepath_prefix_qt + 'data/real/subject_18/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_f.append(filepath_prefix_qt + 'data/real/subject_18/p_files/trainval_sit175rlh_sit120rll.p')

    # test_database_file_m.append(filepath_prefix_qt+'data/real/subject_9/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    # test_database_file_m.append(filepath_prefix_qt+'data/real/subject_9/p_files/trainval_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_10/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_10/p_files/trainval_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_11/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_11/p_files/trainval_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_13/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_13/p_files/trainval_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_14/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_14/p_files/trainval_sit175rlh_sit120rll.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_15/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_15/p_files/trainval_sit175rlh_sit120rll.p')

    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_9/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_17/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair.p')
    #test_database_file_m.append(filepath_prefix_qt + 'data/real/subject_17/p_files/trainval_sit175rlh_sit120rll.p')

    p = PhysicalTrainer(test_database_file_f, test_database_file_m, opt)

    print "GOT HERE!"
    p.init_convnet_test()
    #p.visualize_3d_data()

