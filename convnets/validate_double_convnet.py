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
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['loss_vector_type'] = opt.losstype
        self.CTRL_PNL['verbose'] = opt.verbose
        self.opt = opt
        self.CTRL_PNL['num_epochs'] = 101
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = False
        self.CTRL_PNL['incl_pmat_cntct_input'] = False
        self.CTRL_PNL['num_input_channels'] = 3
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['repeat_real_data_ct'] = 1
        self.CTRL_PNL['regr_angles'] = 1
        self.CTRL_PNL['depth_map_labels'] = False
        self.CTRL_PNL['depth_map_output'] = True
        self.CTRL_PNL['depth_map_input_est'] = False #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['depth_map_input_est'] #holds betas and root same as prior estimate
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
        self.CTRL_PNL['aws'] = False
        self.CTRL_PNL['lock_root'] = False



        # change this to 'direct' when you are doing baseline methods
        self.CTRL_PNL_COR = self.CTRL_PNL.copy()
        self.CTRL_PNL_COR['depth_map_output'] = False
        self.CTRL_PNL_COR['depth_map_input_est'] = True
        self.CTRL_PNL_COR['adjust_ang_from_est'] = True
        self.CTRL_PNL_COR['incl_ht_wt_channels'] = True
        self.CTRL_PNL_COR['incl_pmat_cntct_input'] = True
        self.CTRL_PNL_COR['num_input_channels'] = 3
        if self.CTRL_PNL_COR['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL_COR['num_input_channels'] += 1
        if self.CTRL_PNL_COR['depth_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL_COR['num_input_channels'] += 3
        self.CTRL_PNL_COR['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        if self.CTRL_PNL_COR['incl_ht_wt_channels'] == True:
            self.CTRL_PNL_COR['num_input_channels'] += 2


        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



        #################################### PREP TESTING ##########################################
        #load testing synth data
        dat_f_synth = TensorPrepLib().load_files_to_database(testing_database_file_f, 'synth')
        dat_m_synth = TensorPrepLib().load_files_to_database(testing_database_file_m, 'synth')
        dat_f_real = TensorPrepLib().load_files_to_database(testing_database_file_f, 'real')
        dat_m_real = TensorPrepLib().load_files_to_database(testing_database_file_m, 'real')

        self.test_x_flat = []  # Initialize the testing pressure mat list
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, dat_f_synth, dat_m_synth, num_repeats = 1)
        self.test_x_flat = list(np.clip(np.array(self.test_x_flat) * 5.0, a_min=0, a_max=100))

        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, dat_f_real, dat_m_real, num_repeats = self.CTRL_PNL['repeat_real_data_ct'])

        self.test_x_flat = PreprocessingLib().preprocessing_blur_images(self.test_x_flat, self.mat_size, sigma=0.5)

        if len(self.test_x_flat) == 0: print("NO testing DATA INCLUDED")

        self.test_a_flat = []  # Initialize the testing pressure mat angle list
        self.test_a_flat = TensorPrepLib().prep_angles(self.test_a_flat, dat_f_synth, dat_m_real, num_repeats = 1)
        self.test_a_flat = TensorPrepLib().prep_angles(self.test_a_flat, dat_f_real, dat_m_real, num_repeats = self.CTRL_PNL['repeat_real_data_ct'])

        if self.CTRL_PNL['depth_map_labels'] == True:
            self.depth_contact_maps = [] #Initialize the precomputed depth and contact maps
            self.depth_contact_maps = TensorPrepLib().prep_depth_contact(self.depth_contact_maps, dat_f_synth, dat_m_synth, num_repeats = 1)
        else:
            self.depth_contact_maps = None

        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat,
                                                                                self.test_a_flat,
                                                                                self.CTRL_PNL['incl_inter'], self.mat_size,
                                                                                self.CTRL_PNL['verbose'])

        test_xa = TensorPrepLib().append_input_depth_contact(np.array(test_xa),
                                                              include_pmat_contact = self.CTRL_PNL_COR['incl_pmat_cntct_input'],
                                                              mesh_depth_contact_maps = self.depth_contact_maps,
                                                              include_mesh_depth_contact = self.CTRL_PNL['depth_map_labels'])

        self.test_x_tensor = torch.Tensor(test_xa)

        self.test_y_flat = []  # Initialize the testing ground truth list
        self.test_y_flat = TensorPrepLib().prep_labels(self.test_y_flat, dat_f_synth, num_repeats = 1,
                                                        z_adj = -0.075, gender = "f", is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        self.test_y_flat = TensorPrepLib().prep_labels(self.test_y_flat, dat_m_synth, num_repeats = 1,
                                                        z_adj = -0.075, gender = "m", is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])

        self.test_y_flat = TensorPrepLib().prep_labels(self.test_y_flat, dat_f_real, num_repeats = self.CTRL_PNL['repeat_real_data_ct'],
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        self.test_y_flat = TensorPrepLib().prep_labels(self.test_y_flat, dat_m_real, num_repeats = self.CTRL_PNL['repeat_real_data_ct'],
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        self.test_y_tensor = torch.Tensor(self.test_y_flat)

        print self.test_x_tensor.shape, 'Input testing tensor shape'
        print self.test_y_tensor.shape, 'Output testing tensor shape'


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
                self.model = torch.load('/home/henry/data/synth/convnet_anglesEU_synthreal_s4_3xreal_128b_200e.pt')
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/1.5xsize/convnet_anglesEU_synthreal_tanh_s4ang_sig0p5_5xreal_voloff_128b_300e.pt')
                self.model = self.model.cuda()
            else:
                #self.model = torch.load('/home/henry/data/convnets/convnet_anglesEU_synthreal_tanh_s6ang_sig0p5_5xreal_voloff_128b_200e.pt', map_location='cpu')

                #self.model = torch.load('/home/henry/data/synth/convnet_anglesEU_synth_planesreg_128b_100e.pt', map_location='cpu')
                #self.model = torch.load('/home/henry/data/convnets/convnet_anglesEU_synthreal_tanh_s4ang_sig0p5_5xreal_voloff_128b_200e.pt', map_location='cpu')

                self.model = torch.load('/home/henry/data/convnets/convnet_anglesEU_synthreal_s4_3xreal_128b_200e.pt', map_location = 'cpu')
                self.model_cor = torch.load('/home/henry/data/convnets/convnet_anglesEU_synth_s9_3xreal_128b_101e_pmatcntin_depthestin_angleadj_25e.pt', map_location = 'cpu')
                self.model_cor2 = torch.load('/home/henry/data/convnets/convnet_anglesEU_synth_s9_3xreal_128b_101e_pmatcntin_angleadj_25e.pt', map_location = 'cpu')
                #self.model = torch.load('/home/henry/data/synth/convnet_anglesEU_synthreal_s4_3xreal_4xsize_128b_200e.pt', map_location = 'cpu')
                #self.model = torch.load('/home/henry/data/convnets/convnet_anglesEU_synthreal_tanh_s4ang_sig0p5_5xreal_voloff_128b_200e.pt', map_location='cpu')
                #self.model = torch.load('/media/henry/multimodal_data_2/data/convnets/2.0xsize/convnet_anglesEU_synthreal_tanh_s8ang_sig0p5_5xreal_voloff_128b_300e.pt', map_location='cpu')

            print 'LOADED anglesEU !!!!!!!!!!!!!!!!!1'
            pp = 0
            for p in list(self.model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'

            for p in list(self.model_cor.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params correction net'


        print 'done with epochs, now evaluating'
        self.validate_convnet('test')


    def validate_convnet(self, verbose=False, n_batches=None):

        self.model.eval()
        self.model_cor.eval()
        loss = 0.
        n_examples = 0

        for batch_i, batch in enumerate(self.test_loader):

            if self.CTRL_PNL['loss_vector_type'] == 'direct':
                scores, INPUT_DICT, OUTPUT_DICT = \
                    UnpackBatchLib().unpackage_batch_dir_pass(batch, is_training=False, model=self.model,
                                                              CTRL_PNL=self.CTRL_PNL)
                self.criterion = nn.L1Loss()
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[1].shape[0], scores.size()[1]))).type(dtype),
                                        requires_grad=False)
                loss += self.criterion(scores, scores_zeros).data.item()



            elif self.CTRL_PNL['loss_vector_type'] == 'anglesR' or self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':

                batch_cor = []
                batch_cor.append(batch[0])
                batch_cor.append(batch[1])
                if self.CTRL_PNL_COR['incl_pmat_cntct_input'] == True: batch[0] = batch[0][:, 1:, :, :]
                print batch[0].size(), 'batch 0 size'
                print batch[1].size(), 'batch 1 size'
                print batch_cor[0].size(), 'batch cor 0 size'
                print batch_cor[1].size(), 'batch cor 1 size'

                scores, INPUT_DICT, OUTPUT_DICT = \
                    UnpackBatchLib().unpackage_batch_kin_pass(batch, is_training=False, model=self.model,
                                                              CTRL_PNL=self.CTRL_PNL)


                self.criterion = nn.L1Loss()
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype),
                                        requires_grad=False)

                loss_curr = self.criterion(scores[:, 10:34], scores_zeros[:, 10:34]).data.item() / 10.

                print loss_curr, 'loss'

                mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)
                mdm_est_neg = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)
                mdm_est_pos[mdm_est_pos < 0] = 0
                mdm_est_neg[mdm_est_neg > 0] = 0
                mdm_est_neg *= -1
                cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100
                # depth_contact_input_est_list.append([dat['mdm_est'][entry], dat['cm_est'][entry], ])

                if self.CTRL_PNL_COR['incl_pmat_cntct_input'] == True:
                    batch_cor[0] = torch.cat((batch_cor[0][:, 0:1, :, :],
                                                mdm_est_pos,
                                                mdm_est_neg,
                                                cm_est,
                                                batch_cor[0][:, 1:, :, :]), dim = 1)
                else:
                    batch_cor[0] = torch.cat((mdm_est_pos,
                                                mdm_est_neg,
                                                cm_est,
                                                batch_cor[0]), dim = 1)

                batch_cor[1] = torch.cat((batch_cor[1],
                                          OUTPUT_DICT['batch_betas_est'].clone(),
                                          OUTPUT_DICT['batch_angles_est'].clone(),
                                          OUTPUT_DICT['batch_root_xyz_est'].clone()), dim = 1)

                print batch_cor[0].size(), 'batch cor 0 size'
                print batch_cor[1].size(), 'batch cor 1 size'

                batch_cor2 = []
                batch_cor2.append(torch.cat((batch_cor[0][:, 0:1, :, :],
                                                batch_cor[0][:, 4:, :, :]), dim=1))
                batch_cor2.append(batch_cor[1].clone())


                scores, INPUT_DICT_COR, OUTPUT_DICT_COR = \
                    UnpackBatchLib().unpackage_batch_kin_pass(batch_cor, is_training=False, model=self.model_cor,
                                                              CTRL_PNL=self.CTRL_PNL_COR)


                print batch_cor2[0].size(), 'batch cor 0 size'
                print batch_cor2[1].size(), 'batch cor 1 size'

                scores2, INPUT_DICT_COR2, OUTPUT_DICT_COR2 = \
                    UnpackBatchLib().unpackage_batch_kin_pass(batch_cor2, is_training=False, model=self.model_cor2,
                                                              CTRL_PNL=self.CTRL_PNL_COR)

                loss += loss_curr



            n_examples += self.CTRL_PNL['batch_size']
            #print n_examples

            if n_batches and (batch_i >= n_batches):
                break


            try:
                targets_print = torch.cat([targets_print, torch.mean(INPUT_DICT['batch_targets'], dim = 0).unsqueeze(0)], dim=0)
                targets_est_print = torch.cat([targets_est_print, torch.mean(OUTPUT_DICT['batch_targets_est'], dim = 0).unsqueeze(0)], dim=0)
            except:

                targets_print = torch.mean(INPUT_DICT['batch_targets'], dim = 0).unsqueeze(0)
                targets_est_print = torch.mean(OUTPUT_DICT['batch_targets_est'], dim = 0).unsqueeze(0)


            print targets_print.shape, INPUT_DICT['batch_targets'].shape
            print targets_est_print.shape, OUTPUT_DICT['batch_targets_est'].shape


            if GPU == True:
                error_norm, error_avg, _ = VisualizationLib().print_error_val(targets_print[-2:-1,:].cpu(),
                                                                                   targets_est_print[-2:-1,:].cpu(),
                                                                                   self.output_size_val,
                                                                                   self.CTRL_PNL['loss_vector_type'],
                                                                                   data='validate')
            else:
                error_norm, error_avg, _ = VisualizationLib().print_error_val(targets_print[-2:-1,:],
                                                                              targets_est_print[-2:-1,:],
                                                                                   self.output_size_val,
                                                                                   self.CTRL_PNL['loss_vector_type'],
                                                                                   data='validate')

            if self.opt.visualize == True:
                loss /= n_examples
                loss *= 100
                loss *= 1000

                print INPUT_DICT['batch_images'].size()
                NUM_IMAGES = INPUT_DICT['batch_images'].size()[0]

                for image_ct in range(NUM_IMAGES):
                    # #self.im_sample = self.im_sample[:,0,:,:]
                    self.im_sample = INPUT_DICT['batch_images'][image_ct, :].squeeze()
                    self.tar_sample = INPUT_DICT['batch_targets'][image_ct, :].squeeze() / 1000
                    self.sc_sample = OUTPUT_DICT['batch_targets_est'][image_ct, :].squeeze() / 1000
                    self.im_sample_cor = INPUT_DICT['batch_images'][image_ct, :].squeeze()
                    self.sc_sample_cor = OUTPUT_DICT_COR['batch_targets_est'][image_ct, :].squeeze() / 1000
                    self.im_sample_cor2 = INPUT_DICT['batch_images'][image_ct, :].squeeze()
                    self.sc_sample_cor2 = OUTPUT_DICT_COR2['batch_targets_est'][image_ct, :].squeeze() / 1000

                    #self.im_sample2 = GaussFitLib().get_pressure_under_legs(self.im_sample[0, :, :], np.copy(self.sc_sample))


                    #self.im_sample2 = OUTPUT_DICT['batch_mdm_est'].data[image_ct, :].squeeze()*-1


                    if GPU == True:
                        VisualizationLib().visualize_pressure_map(self.im_sample.cpu(), self.tar_sample.cpu(), self.sc_sample.cpu(), block=False)
                    else:
                        VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,
                                                                  self.im_sample_cor2, self.tar_sample, self.sc_sample_cor2,
                                                                  self.im_sample_cor, self.tar_sample, self.sc_sample_cor,
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

    filepath_prefix_qt = '/home/henry/'

    test_database_file_f = []
    test_database_file_m = []

    network_design = True

    test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_lay_2000_of_2103_upperbody_stiff.p')
   # test_database_file_f.append(filepath_prefix_qt+'data/synth/side_up_fw/train_f_lay_2000_of_2086_rightside_stiff.p')
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

    #test_database_file_f.append(filepath_prefix_qt+'data/real/trainval8_150rh1_sit120rh.p')

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

