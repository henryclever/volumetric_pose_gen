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
#from torchvision import transforms
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
        self.loss_vector_type = opt.losstype

        self.GPU = GPU

        self.verbose = opt.verbose
        self.opt = opt
        self.batch_size = 1
        self.num_epochs = 300
        self.include_inter = True
        self.shuffle = True

        self.count = 0

        #print testing_database_file_f
        print self.num_epochs, 'NUM EPOCHS!'
        # Entire pressure dataset with coordinates in world frame


        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)

        #print training_database_file_f, training_database_file_m


        # load in the test file
        test_dat_f = self.load_files_to_database(testing_database_file_f, 'unlabeled_pmat_data', 'training f real')
        test_dat_m = self.load_files_to_database(testing_database_file_m, 'unlabeled_pmat_data', 'training f real')

        # create a tensor for our testing dataset.  First print out how many input/output sets we have and what data we have
        if test_dat_f is not None:
            for key in test_dat_f:
                print 'testing set: ', key, np.array(test_dat_f[key]).shape
        if test_dat_m is not None:
            for key in test_dat_m:
                print 'testing set: ', key, np.array(test_dat_m[key]).shape

        self.test_x_flat = []  # Initialize the testing pressure mat listhave
        if test_dat_f is not None:
            for entry in range(len(test_dat_f['images'])):
                self.test_x_flat.append(np.clip(np.array(test_dat_f['images'][entry])*5.0, a_min=0, a_max=100))
        if test_dat_m is not None:
            for entry in range(len(test_dat_m['images'])):
                self.test_x_flat.append(np.clip(np.array(test_dat_m['images'][entry])*5.0, a_min=0, a_max=100))


        self.test_x_flat = PreprocessingLib().preprocessing_blur_images(self.test_x_flat, self.mat_size, sigma=0.5)

        self.test_a_flat = []  # Initialize the testing pressure mat angle listhave
        if test_dat_f is not None:
            for entry in range(len(test_dat_f['images'])):
                self.test_a_flat.append(test_dat_f['bed_angle_deg'][entry])
        if test_dat_m is not None:
            for entry in range(len(test_dat_m['images'])):
                self.test_a_flat.append(test_dat_m['bed_angle_deg'][entry])

        if len(self.test_x_flat) == 0: print("NO TESTING DATA INCLUDED")

        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat, self.test_a_flat,
                                                                               self.include_inter, self.mat_size,
                                                                               self.verbose)
        test_xa = np.array(test_xa)
        self.test_x_tensor = torch.Tensor(test_xa)

        self.test_y_flat = []  # Initialize the ground truth listhave
        if test_dat_f is not None:
            for entry in range(len(test_dat_f['images'])):
                c = np.concatenate(([1], [0]), axis=0)  # shapedirs (N, 6890, 3, 10)
                self.test_y_flat.append(c)

        if test_dat_m is not None:
            for entry in range(len(test_dat_m['images'])):
                c = np.concatenate(([0], [1]), axis=0)  # shapedirs (N, 6890, 3, 10)
                self.test_y_flat.append(c)

        self.test_y_tensor = torch.Tensor(self.test_y_flat)

        self.parents = np.array(
            [4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)

    def load_files_to_database(self, database_file, creation_type, descriptor):
        # load in the training or testing files.  This may take a while.
        try:
            for some_subject in database_file:
                print "got here"
                print some_subject
                if creation_type in some_subject:
                    print "some creation type"
                    dat_curr = load_pickle(some_subject)
                    print some_subject, dat_curr['bed_angle_deg'][0]
                    for key in dat_curr:
                        if np.array(dat_curr[key]).shape[0] != 0:
                            for inputgoalset in np.arange(len(dat_curr['images'])):
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
            print "COULD NOT LOAD"
            dat = None
        return dat

    def visualize_3d_data(self):


        p_mat_array = self.train_x_tensor.numpy()[:, 0, :, :]
        bedangle_array = self.train_x_tensor.numpy()[:, 2, 0, 0]

        print self.train_y_tensor.size(), self.train_y_tensor.numpy()[:, 0:72].shape

        joint_loc_array = self.train_y_tensor.numpy()[:, 0:72].reshape(-1, 24, 3)/1000
        root_loc_array = self.train_y_tensor.numpy()[:, 154:157]





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
                                            self.posedirs_repeat[:, :, 1046, :],
                                            self.posedirs_repeat[:, :, 4530, :],
                                            self.posedirs_repeat[:, :, 3333, :],
                                            self.posedirs_repeat[:, :, 6732, :],
                                            self.posedirs_repeat[:, :, 1664, :],
                                            self.posedirs_repeat[:, :, 5121, :],
                                            self.posedirs_repeat[:, :, 2208, :],
                                            self.posedirs_repeat[:, :, 5669, :]])
        self.posedirs_repeat = self.posedirs_repeat.permute(1, 2, 0, 3).contiguous().view(self.N, 2, 10*self.D*207)

        self.weights_repeat_f = self.weights_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.weights_repeat_m = self.weights_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
        self.weights_repeat = torch.cat((self.weights_repeat_f, self.weights_repeat_m), 0)
        #self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R * 24)
        self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R, 24)
        self.weights_repeat = torch.stack([self.weights_repeat[:, :, 1325, :],
                                            self.weights_repeat[:, :, 336, :],
                                            self.weights_repeat[:, :, 1046, :],
                                            self.weights_repeat[:, :, 4530, :],
                                            self.weights_repeat[:, :, 3333, :],
                                            self.weights_repeat[:, :, 6732, :],
                                            self.weights_repeat[:, :, 1664, :],
                                            self.weights_repeat[:, :, 5121, :],
                                            self.weights_repeat[:, :, 2208, :],
                                            self.weights_repeat[:, :, 5669, :]])
        self.weights_repeat = self.weights_repeat.permute(1, 2, 0, 3).contiguous().view(self.N, 2, 10*24)



    def visualize_offset_3d_data(self):

        from kinematics_lib import KinematicsLib

        self.init_smpl(self.batch_size)


        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.batch_size, shuffle=self.shuffle)



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
                                        v_shaped[:, 1046, :], #l knee
                                        v_shaped[:, 4530, :], #r knee
                                        v_shaped[:, 3333, :], #l ankle
                                        v_shaped[:, 6732, :], #r ankle
                                        v_shaped[:, 1664, :], #l elbow
                                        v_shaped[:, 5121, :], #r elbow
                                        v_shaped[:, 2208, :], #l wrist
                                        v_shaped[:, 5669, :]]).permute(1, 0, 2) #r wrist
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
                #offset_targets = np.stack([verts[ct, 1325, :], #torso
                #                          verts[ct, 336, :], #head
                #                          verts[ct, 1046, :], #l knee
                #                          verts[ct, 4530, :], #r knee
                #                          verts[ct, 3333, :], #l ankle
                ##                          verts[ct, 6732, :], #r ankle
                 #                         verts[ct, 1664, :], #l elbow
                 #                         verts[ct, 5121, :], #r elbow
                 #                         verts[ct, 2208, :], #l wrist
                 #                         verts[ct, 5669, :]]) #r wrist

                #print offset_targets


                #VizLib.rviz_publish_input(p_mat_array[ct, :, :], bedangle_array[ct])
                #VizLib.rviz_publish_output_mesh(verts.numpy()[ct], verts.numpy()[ct], offset_targets)
                VizLib.rviz_publish_output_mesh(verts.numpy()[ct], verts.numpy()[ct])
                VizLib.rviz_publish_output(joint_loc_array[ct])

                #print bedangle_array[ct], joint_loc_array[ct, 0, :], root_loc_array[ct, :]

                time.sleep(5.0)


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
            if GPU == True:
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/1.5xsize/convnet_direct_real_s5_128b_300e.pt')
                #self.model = torch.load('/home/henry/data/convnets/convnet_direct_real_s2_128b_300e.pt')
                self.model = torch.load('/home/henry/data/synth/convnet_direct_real_s9_alltest_128b_300e.pt')
                self.model = self.model.cuda()
            else:
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/1.5xsize/convnet_direct_real_s5_128b_300e.pt', map_location='cpu')
                self.model = torch.load('/home/henry/data/convnets/convnet_direct_real_s9_alltest_128b_300e_noscale.pt', map_location='cpu')


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

        elif self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
            fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations
            if GPU == True:
                #self.model = torch.load('/home/henry/data/convnets/convnet_anglesEU_synthreal_tanh_s7ang_sig0p5_5xreal_voloff_128b_200e.pt')
                self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/1.5xsize/convnet_anglesEU_synthreal_tanh_s4ang_sig0p5_5xreal_voloff_128b_300e.pt')
                self.model = self.model.cuda()
            else:
                #self.model = torch.load('/home/henry/data/convnets/convnet_anglesEU_synthreal_tanh_s4ang_sig0p5_5xreal_voloff_128b_200e.pt', map_location='cpu')
                self.model = torch.load('/home/henry/data/convnets/convnet_anglesEU_synthreal_s4_3xreal_4xsize_detach_128b_200e.pt', map_location='cpu')
                #pass
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/2.0xsize/convnet_anglesEU_synthreal_tanh_s8ang_sig0p5_5xreal_voloff_128b_300e.pt', map_location='cpu')

            print 'LOADED!!!!!!!!!!!!!!!!!1'
            pp = 0
            for p in list(self.model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'


        self.criterion = F.cross_entropy


        print 'done with epochs, now evaluating'
        self.validate_convnet('test')


    def validate_convnet(self, verbose=False, n_batches=None):

        if DROPOUT == True:
            self.model.train()
        else:
            self.model.eval()
        loss = 0.
        n_examples = 0

        error_list = []

        for batch_i, batch in enumerate(self.test_loader):

            if DROPOUT == True:
                batch[0] = batch[0].repeat(25, 1, 1, 1)
                batch[1] = batch[1].repeat(25, 1)
            #self.model.train()

            if self.loss_vector_type == 'direct':


                images_up_non_tensor = np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, :, :], multiple = 2))
                images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)
                images = Variable(batch[0].type(dtype), requires_grad=False)

                scores, targets_est, targets_est_reduced = self.model.forward_direct(images_up, 0, targets, is_training = False)

                self.criterion = nn.L1Loss()

                loss = self.criterion(scores, scores_zeros)
                loss = loss.data.item()




            elif self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':



                images_up_non_tensor = np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy(), multiple=2))
                images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)
                images = Variable(batch[0].type(dtype), requires_grad=False)

                gender_switch = Variable(batch[1].type(dtype), requires_grad=False)



                if self.loss_vector_type == 'anglesR':
                    scores, targets_est, targets_est_reduced, betas_est = self.model.forward_kinematic_R(images_up, gender_switch,
                                                                                                         0, targets=None, is_training=False)  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.
                elif self.loss_vector_type == 'anglesDC' or self.loss_vector_type == 'anglesEU':
                    scores, targets_est, targets_est_reduced, betas_est = self.model.forward_kinematic_angles(images_up, gender_switch,
                                                                                                              0, targets=None, is_training=False,
                                                                                                              reg_angles = self.opt.reg_angles)  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.

                #print("VAL SCORES",
                #      np.sum(np.abs(scores.data.numpy())),
                #      np.sum(np.abs(scores.data.numpy())),
                #      np.sum(np.abs(scores.data.numpy()))/(scores.size()[0] * scores.size()[1]),
                #      np.sum(np.abs(scores.data.numpy()))/(scores.size()[0] * 24))




            if self.loss_vector_type == 'anglesR':
                #print angles_est[0, :], 'validation angles'
                print betas_est[0, :], 'validation betas'


            NUM_IMAGES = images.data.size()[0]

            for image_ct in range(NUM_IMAGES):
                # #self.im_sampleval = self.im_sampleval[:,0,:,:]
                self.im_sampleval = images.data[image_ct, :].squeeze()
                self.sc_sampleval = targets_est[image_ct, :].squeeze() / 1000
                self.sc_sampleval = self.sc_sampleval.view(24, 3)


                if GPU == True:
                    VisualizationLib().visualize_pressure_map(self.im_sampleval.cpu(), self.sc_sampleval.cpu(), block=False)
                else:
                    VisualizationLib().visualize_pressure_map(self.im_sampleval, self.sc_sampleval, block=False)
                time.sleep(1)




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
    p.add_option('--rgangs', action='store_true',
                 dest='reg_angles', \
                 default=False, \
                 help='Regress the angles as well as betas and joint pos.')
    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=True, help='Printout everything (under construction).')

    p.add_option('--log_interval', type=int, default=5, metavar='N',
                 help='number of batches between logging train status')

    opt, args = p.parse_args()

    filepath_prefix_qt = '/home/henry/'

    training_database_file_f = []
    training_database_file_m = []
    test_database_file_f = []
    test_database_file_m = []

    #test_database_file_f.append(filepath_prefix_qt+'data/real/trainval4_150rh1_sit120rh.p')
    test_database_file_m.append('/home/henry/data/unlabeled_pmat_data/henryc_on_bed_05102019.p')

    #test_database_file_m.append('/home/henry/data/unlabeled_pmat_data/henrye_on_bed_09102019.p')

    p = PhysicalTrainer(test_database_file_f, test_database_file_m, opt)

    print "GOT HERE!"
    p.init_convnet_train()
    #p.visualize_3d_data()
    #p.visualize_3d_data()
        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
