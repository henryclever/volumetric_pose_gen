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


import convnet as convnet
import convnet_cascade as convnet_cascade
import tf.transformations as tft

# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Pose Estimation Libraries
from create_dataset_lib import CreateDatasetLib
from synthetic_lib import SyntheticLib
from visualization_lib import VisualizationLib
from kinematics_lib import KinematicsLib
from cascade_lib import CascadeLib
from preprocessing_lib import PreprocessingLib


import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
from skimage.feature import hog
from skimage import data, color, exposure


from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.multioutput import MultiOutputRegressor

np.set_printoptions(threshold='nan')

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES = 10
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)

torch.set_num_threads(1)
if False:#torch.cuda.is_available():
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
    def __init__(self, training_database_file, test_file, opt):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''


        #change this to 'direct' when you are doing baseline methods
        self.loss_vector_type = opt.losstype

        self.verbose = opt.verbose
        self.opt = opt
        self.batch_size = 128

        #self.model = torch.load(
        #    '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials' + '/subject_' + str(
        #        self.opt.leave_out) + '/convnets/convnet_9to18_anglesSTVL_sTrue_128b_200e_' + str(
        #        self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)
        #print 'LOADED!!!!!!!!!!!!!!!!!1'

        self.num_epochs = 200
        self.include_inter = True
        self.shuffle = True

        if opt.mltype == 'convnet':
            self.tensor = True
        else:
            self.tensor = False

        self.count = 0


        print test_file
        print self.num_epochs, 'NUM EPOCHS!'
        #Entire pressure dataset with coordinates in world frame

        self.save_name = '_9to18_' + opt.losstype+'_s' +str(self.shuffle)+'_' + str(self.batch_size) + 'b_' + str(self.num_epochs) + 'e_'+str(self.opt.leave_out)



        print 'appending to','train'+self.save_name+str(self.opt.leave_out)
        self.train_val_losses = {}
        self.train_val_losses['train'+self.save_name] = []
        self.train_val_losses['val'+self.save_name] = []
        self.train_val_losses['epoch'+self.save_name] = []

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size = (NUMOFOUTPUTNODES, NUMOFOUTPUTDIMS)


        if self.tensor == True:
            #load in the training files.  This may take a while.
            for some_subject in training_database_file:
                print some_subject
                dat_curr = load_pickle(some_subject)
                for key in dat_curr:
                    if np.array(dat_curr[key]).shape[0] != 0:
                        for inputgoalset in np.arange(len(dat_curr['markers_xyz_m'])):
                            try:
                                dat[key].append(dat_curr[key][inputgoalset])
                            except:
                                try:
                                    dat[key] = []
                                    dat[key].append(dat_curr[key][inputgoalset])
                                except:
                                    dat = {}
                                    dat[key] = []
                                    dat[key].append(dat_curr[key][inputgoalset])




            #create a tensor for our training dataset.  First print out how many input/output sets we have and what data we have
            for key in dat:
                print 'training set: ', key, np.array(dat[key]).shape

            if self.opt.vgg == False:
                self.train_x_flat = []  # Initialize the testing pressure mat list
                for entry in range(len(dat['images'])):
                    self.train_x_flat.append(dat['images'][entry])

                self.train_a_flat = []  # Initialize the testing pressure mat angle list
                for entry in range(len(dat['images'])):
                    self.train_a_flat.append(dat['bed_angle_deg'][entry])
                train_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.train_x_flat, self.train_a_flat, self.include_inter, self.mat_size, self.verbose)
                train_xa = np.array(train_xa)

                self.train_x_tensor = torch.Tensor(train_xa)
            else:
                train_f = []  # Initialize the testing pressure mat list
                for entry in range(len(dat['features'])):
                    train_f.append(dat['features'][entry])
                train_f = np.array(train_f)
                self.train_x_tensor = torch.Tensor(train_f)


            print self.train_x_tensor.shape, 'tensor shape'

            self.train_y_flat = [] #Initialize the training ground truth list
            for entry in range(len(dat['markers_xyz_m'])):
                if self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL'  or self.loss_vector_type == 'anglesSTVL':
                    c = np.concatenate((dat['markers_xyz_m'][entry][0:30] * 1000,
                                        dat['joint_lengths_U_m'][entry][0:9] * 100,
                                        dat['joint_angles_U_deg'][entry][0:10],
                                        dat['joint_lengths_L_m'][entry][0:8] * 100,
                                        dat['joint_angles_L_deg'][entry][0:8]), axis=0)
                    self.train_y_flat.append(c)
                else:
                    self.train_y_flat.append(dat['markers_xyz_m'][entry] * 1000)

            print np.shape(self.train_y_flat), 'shape flat!'
            self.train_y_tensor = torch.Tensor(self.train_y_flat)


            #load in the test file
            for some_subject in test_file:
                print some_subject
                dat_curr = load_pickle(some_subject)
                for key in dat_curr:
                    if np.array(dat_curr[key]).shape[0] != 0:
                        for inputgoalset in np.arange(len(dat_curr['markers_xyz_m'])):
                            try:
                                test_dat[key].append(dat_curr[key][inputgoalset])
                            except:
                                try:
                                    test_dat[key] = []
                                    test_dat[key].append(dat_curr[key][inputgoalset])
                                except:
                                    test_dat = {}
                                    test_dat[key] = []
                                    test_dat[key].append(dat_curr[key][inputgoalset])



            # create a tensor for our testing dataset.  First print out how many input/output sets we have and what data we have
            for key in test_dat:
                print 'testing set: ', key, np.array(test_dat[key]).shape


            if self.opt.vgg == False:
                self.test_x_flat = []  # Initialize the testing pressure mat list
                for entry in range(len(test_dat['images'])):
                    self.test_x_flat.append(test_dat['images'][entry])

                self.test_a_flat = []  # Initialize the testing pressure mat angle list
                for entry in range(len(test_dat['images'])):
                    self.test_a_flat.append(test_dat['bed_angle_deg'][entry])
                test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat, self.test_a_flat, self.include_inter, self.mat_size, self.verbose)
                test_xa = np.array(test_xa)
                self.test_x_tensor = torch.Tensor(test_xa)
            else:
                test_f = []  # Initialize the testing pressure mat list
                for entry in range(len(test_dat['features'])):
                    test_f.append(test_dat['features'][entry])
                test_f = np.array(test_f)
                self.test_x_tensor = torch.Tensor(test_f)


            self.test_y_flat = []  # Initialize the ground truth list
            for entry in range(len(test_dat['markers_xyz_m'])):
                if self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL':
                    c = np.concatenate((test_dat['markers_xyz_m'][entry][0:30] * 1000,
                                        test_dat['joint_lengths_U_m'][entry][0:9] * 100,
                                        test_dat['joint_angles_U_deg'][entry][0:10],
                                        test_dat['joint_lengths_L_m'][entry][0:8] * 100,
                                        test_dat['joint_angles_L_deg'][entry][0:8]), axis=0)
                    self.test_y_flat.append(c)
                elif self.loss_vector_type == 'direct' or self.loss_vector_type == 'confidence':
                    self.test_y_flat.append(test_dat['markers_xyz_m'][entry] * 1000)
                else:
                    print "ERROR! SPECIFY A VALID LOSS VECTOR TYPE."
            self.test_y_flat = np.array(self.test_y_flat)
            self.test_y_tensor = torch.Tensor(self.test_y_flat)


        else:
            #load in the training files.  This may take a while.
            for some_subject in training_database_file:
                print some_subject
                images_train, targets_train = load_pickle(some_subject)
                images_train = np.array(images_train)
                targets_train = np.array(targets_train)
                try:
                    images_train_aggregated = np.concatenate((images_train_aggregated, images_train), axis=0)
                except:
                    images_train_aggregated = np.array(images_train)
                try:
                    targets_train_aggregated = np.concatenate((targets_train_aggregated, targets_train), axis=0)
                except:
                    targets_train_aggregated = np.array(targets_train)
            self.images_train_aggregated = images_train_aggregated
            self.targets_train_aggregated = targets_train_aggregated
            print self.images_train_aggregated.shape, self.targets_train_aggregated.shape

            # load in the test file
            for some_subject in test_file:
                images_test, targets_test = load_pickle(some_subject)
                images_test = np.array(images_test)
                targets_test = np.array(targets_test)
                print some_subject

                try:
                    images_test_aggregated = np.concatenate((images_test_aggregated, images_test), axis=0)
                except:
                    images_test_aggregated = np.array(images_test)
                try:
                    targets_test_aggregated = np.concatenate((targets_test_aggregated, targets_test), axis=0)
                except:
                    targets_test_aggregated = np.array(targets_test)
            self.images_test_aggregated = images_test_aggregated
            self.targets_test_aggregated = targets_test_aggregated
            print self.images_test_aggregated.shape, self.targets_test_aggregated.shape


    def baseline_createHOGset(self, opt):
        # for knn we don't really care about the variable function in pytorch, but it's a nice utility for shuffling the data.
        self.batch_size = self.train_y_tensor.numpy().shape[0]
        self.batchtest_size = 11  # self.test_y_tensor.numpy().shape[0]

        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=True)


        for batch_idx, batch in enumerate(self.train_loader):

            # get the whole body x y z
            batch[1] = batch[1][:, 0:30]

            batch[0], batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True,
                                                                    shift=True, scale=False,
                                                                    bedangle=True,
                                                                    include_inter=self.include_inter,
                                                                    loss_vector_type=self.loss_vector_type)

            images = batch[0].numpy()[:, 0, 10:74, 10:37]
            targets = batch[1].numpy()

            print images.shape

            # upsample the images
            images_up = PreprocessingLib().preprocessing_pressure_map_upsample(images)
            # targets = list(targets)
            # print images[0].shape

            print np.shape(images_up), 'IMAGES UP SHAPE upsample'


            # Compute HoG of the current(training) pressure map dataset
            images_up = PreprocessingLib().compute_HoG(images_up)

            print np.shape(images_up)
            print np.shape(targets), 'target shape'

            # images_up = np.reshape(images, (images.shape[0], images.shape[1]*images.shape[2]))

            # images_up = [[0], [1], [2], [3]]
            # targets = [0, 0, 1, 1]

            print np.shape(images_up), 'IMAGES UP SHAPE'
            print np.shape(targets)
            pkl.dump([images_up, targets], open(os.path.join('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'+str(opt.leave_out)+'/p_files/trainval_200rlh1_115rlh2_75rlh3_175rllair_HOGshift.p'), 'wb'))  #shiftscale_ #sit175rlh_sit120rll_HOG

    def baseline_train(self, baseline):
        n_neighbors = 5
        cv_fold = 3

        images_up = self.images_train_aggregated
        targets = self.targets_train_aggregated
        #
        print 'fitting KNN'
        baseline = 'KNN'
        #
        #if baseline == 'KNN':
        regr = neighbors.KNeighborsRegressor(10, weights='distance', algorithm = 'auto', p=2, metric = 'minkowski')
        regr.fit(images_up, targets)
        #
        # print 'done fitting KNN'
        #
        if self.opt.computer == 'lab_harddrive':
            print 'saving to ', '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                self.opt.leave_out) + '/p_files/HoGshift_' + baseline + '_p' + str(self.opt.leave_out) + '.p'
            pkl.dump(regr, open(
                '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leave_out) + '/p_files/HoGshift_' + baseline + '_p' + str(self.opt.leave_out) + '.p',
                'wb'))
            print 'saved successfully'
        elif self.opt.computer == 'aws':
            pkl.dump(regr, open('/home/ubuntu/Autobed_OFFICIAL_Trials/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'
        elif self.opt.computer == 'baymax':
            pkl.dump(regr, open('/home/henryclever/IROS_Data/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'
        elif self.opt.computer == 'gorilla':
            pkl.dump(regr, open('/home/henry/IROS_Data/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'
        #
        #
        print 'fitting Ridge'
        baseline = 'Ridge'
        #elif baseline == 'Ridge':
        regr = linear_model.Ridge(alpha=0.7) # complexity parameter that controls the amount of shrinkage:
        # the larger the value of \alpha, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity
        # a zero value for alpha is the same as doing OLS (ordinary least squares. More complex model in this case and more prone to overfitting
        # The shrinkage is a regularization term. coefficients in the regularization correspond to those in the least squares part
        #OLS can overfit to data with high variance.  the alpha term helps to keep this from happpening

        regr.fit(images_up, targets)

        print 'done fitting Ridge'

        if self.opt.computer == 'lab_harddrive':
            print 'saving to ', '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                self.opt.leave_out) + '/p_files/HoGshift0.7_' + baseline + '_p' + str(self.opt.leave_out) + '.p'
            pkl.dump(regr, open(
                '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leave_out) + '/p_files/HoGshift0.7_' + baseline + '_p' + str(self.opt.leave_out) + '.p',
                'wb'))
            print 'saved successfully'
        elif self.opt.computer == 'aws':
            pkl.dump(regr, open('/home/ubuntu/Autobed_OFFICIAL_Trials/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'
        elif self.opt.computer == 'baymax':
            pkl.dump(regr, open('/home/henryclever/IROS_Data/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'
        elif self.opt.computer == 'gorilla':
            pkl.dump(regr, open('/home/henry/IROS_Data/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'



        #
        print 'fitting KRidge'
        baseline = 'KRidge'
        #elif baseline == 'KRidge':
        regr = kernel_ridge.KernelRidge(alpha=0.4, kernel='rbf')
        #regr = kernel_ridge.KernelRidge(kernel='polynomial')
        regr.fit(images_up, targets)
        #
        print 'done fitting KRidge'
        #
        if self.opt.computer == 'lab_harddrive':
            print 'saving to ', '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                self.opt.leave_out) + '/p_files/HoGshift0.4_' + baseline + '_p' + str(self.opt.leave_out) + '.p'
            pkl.dump(regr, open(
                '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_' + str(
                    self.opt.leave_out) + '/p_files/HoGshift0.4_' + baseline + '_p' + str(self.opt.leave_out) + '.p','wb'))
            print 'saved successfully'
        elif self.opt.computer == 'aws':
            pkl.dump(regr, open('/home/ubuntu/Autobed_OFFICIAL_Trials/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'
        elif self.opt.computer == 'baymax':
            pkl.dump(regr, open('/home/henryclever/IROS_Data/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'
        elif self.opt.computer == 'gorilla':
            pkl.dump(regr, open('/home/henry/IROS_Data/subject_' + str(
                self.opt.leave_out) + '/HoG_' + baseline + '_p' + str(self.opt.leave_out) + '.p', 'wb'))
            print 'saved successfully'



        if baseline == 'SVM':
            regr = MultiOutputRegressor(estimator=svm.SVR(C=1.0, kernel='rbf', verbose = True))
            regr.fit(images_up, targets)
            #SVR(C=1.0, kernel='rbf', verbose = True)
            #SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
            #                    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

        elif baseline == 'kmeans_SVM':
            k_means = KMeans(n_clusters=10, n_init=4)
            k_means.fit(images_up)
            labels = k_means.labels_
            print labels.shape, 'label shape'
            print targets.shape, 'target shape'
            print 'done fitting kmeans'
            svm_classifier = svm.SVC(kernel='rbf', verbose = True)
            svm_classifier.fit(labels, targets)
            print 'done fitting svm'
            regr = linear_model.LinearRegression()
            regr.fit(labels, targets)
            print 'done fitting linear model'

        elif baseline == 'Linear':
            regr = linear_model.LinearRegression()
            regr.fit(images_up, targets)



        #validation
        for test_data_length in range(self.targets_test_aggregated.shape[0]):
            scores = regr.predict(self.images_test_aggregated[test_data_length, :])
            targets = self.targets_test_aggregated[test_data_length, :]
            #print scores.shape
            #print targets.shape
            #print scores[0]
            #print targets[0]

            #print regr.predict(images_up_test[0]) - targets[0]
            VisualizationLib().print_error(scores, targets, self.output_size, loss_vector_type=self.loss_vector_type, data='test', printerror=True)

            self.im_sample = np.squeeze(images_test[0, :])
            #print self.im_sample.shape

            self.tar_sample = np.squeeze(targets[0, :]) / 1000
            self.sc_sample = np.copy(scores)
            self.sc_sample = np.squeeze(self.sc_sample[0, :]) / 1000
            self.sc_sample = np.reshape(self.sc_sample, self.output_size)
            if self.opt.visualize == True:
                VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample, block=True)

        print len(scores)
        print scores[0].shape
        print scores.shape
        print targets.shape



    def init_convnet_train(self):
        #indices = torch.LongTensor([0])
        #self.train_y_tensor = torch.index_select(self.train_y_tensor, 1, indices)

        if self.verbose: print self.train_x_tensor.size(), 'size of the training database'
        if self.verbose: print self.train_y_tensor.size(), 'size of the training database output'
        print self.train_y_tensor
        if self.verbose: print self.test_x_tensor.size(), 'length of the testing dataset'
        if self.verbose: print self.test_y_tensor.size(), 'size of the training database output'



        num_epochs = self.num_epochs
        hidden_dim = 12
        kernel_size = 10



        #self.train_x_tensor = self.train_x_tensor.unsqueeze(1)
        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=True)

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.batch_size, shuffle=True)


        output_size = self.output_size[0]*self.output_size[1]

        if self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL':
            fc_output_size = 40#38 #18 angles for body, 17 lengths for body, 3 torso coordinates
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)
            #self.model = torch.load('/home/ubuntu/Autobed_OFFICIAL_Trials' + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_'+str(self.loss_vector_type)+'_sTrue_128b_200e_' + str(self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)
            print 'LOADED!!!!!!!!!!!!!!!!!1'
            pp = 0
            for p in list(self.model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'
        elif self.loss_vector_type == 'direct' or self.loss_vector_type == 'confidence':
            fc_output_size = 30
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)

        # Run model on GPU if available
        if False:#torch.cuda.is_available():
            self.model = self.model.cuda()


        self.criterion = F.cross_entropy



        if self.loss_vector_type == None:
            #previously the learning rates were 0.00002, 0.00002, 0.000002.
            #today I was running 0.00005, 0.00005, 0.000005.
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=0.0005)
        elif self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL' or self.loss_vector_type == 'direct':
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=0.0005)  #0.000002 does not converge even after 100 epochs on subjects 2-8 kin cons. use .00001
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.000001, momentum=0.7, weight_decay=0.0005)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.000002, weight_decay=0.0005) #start with .00005


        # train the model one epoch at a time
        for epoch in range(1, num_epochs + 1):
            self.t1 = time.time()

            self.train_convnet(epoch)

            if epoch > 5: self.optimizer = self.optimizer2

            try:
                self.t2 = time.time() - self.t1
            except:
                self.t2 = 0
            print 'Time taken by epoch',epoch,':',self.t2,' seconds'



        print 'done with epochs, now evaluating'
        self.validate_convnet('test')

        print self.train_val_losses, 'trainval'
        # Save the model (architecture and weights)

        if self.opt.computer == 'lab_harddrive':
            torch.save(self.model, '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'+str(self.opt.leave_out)+'/p_files/convnet'+self.save_name+'.pt')
            pkl.dump(self.train_val_losses,open('/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/subject_'+str(self.opt.leave_out)+'/p_files/losses'+self.save_name+'.p', 'wb'))

        elif self.opt.computer == 'aws':
            torch.save(self.model, '/home/ubuntu/Autobed_OFFICIAL_Trials/subject_'+str(self.opt.leave_out)+'/convnet'+self.save_name+'.pt')
            pkl.dump(self.train_val_losses,open('/home/ubuntu/Autobed_OFFICIAL_Trials/subject_'+str(self.opt.leave_out)+'/losses'+self.save_name+'.p', 'wb'))

        elif self.opt.computer == 'baymax':
            torch.save(self.model, '/home/henryclever/IROS_Data/subject_'+str(self.opt.leave_out)+'/convnet'+self.save_name+'.pt')
            pkl.dump(self.train_val_losses,open('/home/henryclever/IROS_Data/subject_'+str(self.opt.leave_out)+'/losses'+self.save_name+'.p', 'wb'))

        elif self.opt.computer == 'gorilla':
            torch.save(self.model, '/home/henry/IROS_Data/subject_'+str(self.opt.leave_out)+'/convnet'+self.save_name+'.pt')
            pkl.dump(self.train_val_losses,open('/home/henry/IROS_Data/subject_'+str(self.opt.leave_out)+'/losses'+self.save_name+'.p', 'wb'))


    def train_convnet(self, epoch):
        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.
        self.model.train()
        scores = 0


        #This will loop a total = training_images/batch_size times
        for batch_idx, batch in enumerate(self.train_loader):

            if self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL':

                # append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:,39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim = 1))


                #get the whole body x y z
                batch[1] = batch[1][:, 0:30]

                batch[0], batch[1], batch[2]= SyntheticLib().synthetic_master(batch[0], batch[1], batch[2], flip=True, shift=True, scale=True, bedangle=True, include_inter=self.include_inter, loss_vector_type=self.loss_vector_type)

                #images_up = Variable(torch.Tensor(PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37])))).type(dtype), requires_grad=False)
                images_up = Variable(torch.Tensor(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37]))).type(dtype), requires_grad=False)

                images, targets, constraints = Variable(batch[0].type(dtype), requires_grad = False), Variable(batch[1].type(dtype), requires_grad = False), Variable(batch[2].type(dtype), requires_grad = False)


                # targets_2D = CascadeLib().get_2D_projection(images.data.numpy(), np.reshape(targets.data.numpy(), (targets.size()[0], 10, 3)))
                #targets_2D = CascadeLib().get_2D_projection(images.data, targets.data.view(targets.size()[0], 10, 3))

                #image_coords = np.round(targets_2D[:, :, 0:2] / 28.6, 0)
                #print image_coords[0, :, :]

                self.optimizer.zero_grad()

                ground_truth = np.zeros((batch[0].numpy().shape[0], 47)) #47 is 17 joint lengths and 30 joint locations for x y z
                ground_truth = Variable(torch.Tensor(ground_truth).type(dtype))
                ground_truth[:, 0:17] = constraints[:, 18:35]/100
                ground_truth[:, 17:47] = targets[:, 0:30]/1000



                scores_zeros = np.zeros((batch[0].numpy().shape[0], 27)) #27 is  10 euclidean errors and 17 joint lengths
                scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))
                scores_zeros[:, 10:27] = constraints[:, 18:35]/100 #divide by 100 for direct output. divide by 10 if you multiply the estimate length by 10.

                scores, targets_est, angles_est, lengths_est, _ = self.model.forward_kinematic_jacobian(images_up, targets, constraints, forward_only = False, subject = self.opt.leave_out, loss_vector_type=self.loss_vector_type) # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.
                #print lengths_est[0,0:10], 'lengths est'
                #print batch[0][0,2,10,10], 'angle'

                #print scores_zeros[0, :]

                self.criterion = nn.L1Loss()

                if self.loss_vector_type == 'anglesCL':
                    #if epoch < 4:
                    #    loss = self.criterion(scores, scores_zeros) #train like its variable lengths for the first 3 epochs to get things converging
                    #else:
                    loss = self.criterion(scores[0:10], scores_zeros[0:10])
                elif self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL':
                    #print scores_zeros[0, :]
                    #print scores[0,:]

                    loss = self.criterion(scores, scores_zeros)


            elif self.loss_vector_type == 'direct':


                if self.opt.vgg == True:
                    vgg16_image_features = Variable(batch[0].type(dtype), requires_grad=False)
                    images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad=False),  Variable(batch[1].type(dtype), requires_grad=False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1] / 3))).type(dtype), requires_grad=False)

                    self.optimizer.zero_grad()
                    scores, targets_est = self.model.forward_direct_vgg(vgg16_image_features, targets)

                else:
                    batch[0], batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True, shift=True,
                                                                            scale=True, bedangle=True,
                                                                            include_inter=self.include_inter,
                                                                            loss_vector_type=self.loss_vector_type)

                    images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37])))

                    images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)
                    images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad = False), Variable(batch[1].type(dtype), requires_grad = False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1]/3))).type(dtype), requires_grad = False)

                    self.optimizer.zero_grad()
                    scores, targets_est = self.model.forward_direct(images_up, targets)

                self.criterion = nn.L1Loss()
                loss = self.criterion(scores, scores_zeros)

            #print loss.data.numpy() * 1000, 'loss'

            loss.backward()
            self.optimizer.step()
            loss *= 1000


            if batch_idx % opt.log_interval == 0:
                if self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL':
                    print targets.data.size()
                    print targets_est.shape

                    VisualizationLib().print_error(targets.data, targets_est, self.output_size, self.loss_vector_type, data = 'train')
                    print angles_est[0, :], 'angles'
                    print batch[0][0,2,10,10], 'bed angle'

                elif self.loss_vector_type == 'direct':
                    pass

                VisualizationLib().print_error(targets.data, targets_est, self.output_size, self.loss_vector_type, data='train')

                if self.opt.vgg == False:
                    self.im_sample = images.data
                    #self.im_sample = self.im_sample[:,1, :, :]
                    self.im_sample = self.im_sample[0, :].squeeze()
                    self.tar_sample = targets.data
                    self.tar_sample = self.tar_sample[0, :].squeeze()/1000
                    self.sc_sample = targets_est.clone()
                    self.sc_sample = self.sc_sample[0, :].squeeze() / 1000
                    self.sc_sample = self.sc_sample.view(self.output_size)

                val_loss = self.validate_convnet(n_batches=4)
                train_loss = loss.data[0]
                examples_this_epoch = batch_idx * len(images)
                epoch_progress = 100. * batch_idx / len(self.train_loader)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                      'Train Loss: {:.6f}\tVal Loss: {:.6f}'.format(
                    epoch, examples_this_epoch, len(self.train_loader.dataset),
                    epoch_progress, train_loss, val_loss))


                print 'appending to alldata losses'
                self.train_val_losses['train'+self.save_name].append(train_loss)
                self.train_val_losses['val'+self.save_name].append(val_loss)
                self.train_val_losses['epoch'+self.save_name].append(epoch)




    def validate_convnet(self, verbose=False, n_batches=None):

        self.model.eval()
        loss = 0.
        n_examples = 0
        for batch_i, batch in enumerate(self.test_loader):

            self.model.eval()


            if self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL':

                #append upper joint angles, lower joint angles, upper joint lengths, lower joint lengths, in that order
                batch.append(torch.cat((batch[1][:,39:49], batch[1][:, 57:65], batch[1][:, 30:39], batch[1][:, 49:57]), dim = 1))

                #get the direct joint locations
                batch[1] = batch[1][:, 0:30]

                images_up = Variable(torch.Tensor(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37]))).type(dtype), requires_grad=False)
                images, targets, constraints = Variable(batch[0].type(dtype), volatile = True, requires_grad=False), Variable(batch[1].type(dtype),volatile = True, requires_grad=False), Variable(batch[2].type(dtype), volatile = True, requires_grad=False)

                self.optimizer.zero_grad()


                ground_truth = np.zeros((batch[0].numpy().shape[0], 47))
                ground_truth = Variable(torch.Tensor(ground_truth).type(dtype))
                ground_truth[:, 0:17] = constraints[:, 18:35]/100
                ground_truth[:, 17:47] = targets[:, 0:30]/1000

                scores_zeros = np.zeros((batch[0].numpy().shape[0], 27))
                scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))
                scores_zeros[:, 10:27] = constraints[:, 18:35]/100

                scores, targets_est, angles_est, lengths_est, _ = self.model.forward_kinematic_jacobian(images_up, targets, forward_only = False, subject = self.opt.leave_out, loss_vector_type=self.loss_vector_type)


                self.criterion = nn.L1Loss()
                loss = self.criterion(scores[:, 0:10], scores_zeros[:, 0:10])
                loss = loss.data[0]



            elif self.loss_vector_type == 'direct':

                if self.opt.vgg == True:
                    vgg16_image_features = Variable(batch[0].type(dtype), requires_grad=False)
                    images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad=False), Variable(batch[1].type(dtype), requires_grad=False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1] / 3))).type(dtype), requires_grad=False)
                    scores, targets_est = self.model.forward_direct_vgg(vgg16_image_features, targets)

                else:
                    images_up_non_tensor = np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, 10:74, 10:37]))
                    images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)
                    images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad=False), Variable(batch[1].type(dtype), requires_grad=False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1] / 3))).type(dtype), requires_grad=False)

                    scores, targets_est = self.model.forward_direct(images_up, targets)

                self.criterion = nn.L1Loss()

                loss = self.criterion(scores, scores_zeros)
                loss = loss.data[0]


            n_examples += self.batch_size
            #print n_examples

            if n_batches and (batch_i >= n_batches):
                break

        loss /= n_examples
        loss *= 100
        loss *= 1000


        VisualizationLib().print_error(targets.data, targets_est, self.output_size, self.loss_vector_type, data='validate')

        if self.loss_vector_type == 'anglesCL' or self.loss_vector_type == 'anglesVL' or self.loss_vector_type == 'anglesSTVL':
            print angles_est[0, :], 'validation angles'
            print lengths_est[0, :], 'validation lengths'

        if self.opt.vgg == False:
            print batch[0][0,2,10,10], 'validation bed angle'
            self.im_sampleval = images.data
            # #self.im_sampleval = self.im_sampleval[:,0,:,:]
            self.im_sampleval = self.im_sampleval[0, :].squeeze()
            self.tar_sampleval = targets.data
            self.tar_sampleval = self.tar_sampleval[0, :].squeeze() / 1000
            self.sc_sampleval = targets_est.clone()
            self.sc_sampleval = self.sc_sampleval[0, :].squeeze() / 1000
            self.sc_sampleval = self.sc_sampleval.view(self.output_size)

            if self.opt.visualize == True:
                VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,self.im_sampleval, self.tar_sampleval, self.sc_sampleval, block=False)


        return loss



if __name__ == "__main__":
    #Initialize trainer with a training database file
    import optparse
    p = optparse.OptionParser()
    p.add_option('--training_dataset', '--train_dataset',  action='store', type='string', \
                 dest='trainPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/basic_train_dataset.p', \
                 help='Specify path to the training database.')
    p.add_option('--leave_out', action='store', type=int, \
                 dest='leave_out', \
                 help='Specify which subject to leave out for validation')
    p.add_option('--only_test','--t',  action='store_true', dest='only_test',
                 default=False, help='Whether you want only testing of previously stored model')
    p.add_option('--training_model', '--model',  action='store', type='string', \
                 dest='modelPath',\
                 default = '/home/henryclever/hrl_file_server/Autobed/pose_estimation_data', \
                 help='Specify path to the trained model')
    p.add_option('--testing_dataset', '--test_dataset',  action='store', type='string', \
                 dest='testPath',\
                 default='/home/henryclever/hrl_file_server/Autobed/pose_estimation_data/basic_test_dataset.p', \
                 help='Specify path to the training database.')
    p.add_option('--computer', action='store', type = 'string',
                 dest='computer', \
                 default='lab_harddrive', \
                 help='Set path to the training database on lab harddrive.')
    p.add_option('--gpu', action='store', type = 'string',
                 dest='gpu', \
                 default='0', \
                 help='Set the GPU you will use.')
    p.add_option('--mltype', action='store', type = 'string',
                 dest='mltype', \
                 default='convnet', \
                 help='Set if you want to do baseline ML or convnet.')
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
    p.add_option('--vgg', action='store_true',
                 dest='vgg', \
                 default=False, \
                 help='Train on VGG features.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=False, help='Printout everything (under construction).')
    p.add_option('--log_interval', type=int, default=10, metavar='N',
                        help='number of batches between logging train status')

    opt, args = p.parse_args()

    if opt.mltype == 'convnet': filetag = ''
    else: filetag = '_HOGshift'

    if opt.computer == 'lab_harddrive':
        filepath_prefix = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'
        filepath_prefix_qt = '/home/henryclever/test'

    elif opt.computer == 'aws':
        filepath_prefix = '/home/ubuntu/Autobed_OFFICIAL_Trials'
        filepath_prefix_qt = '/home/ubuntu/test'

    elif opt.computer == 'baymax':
        filepath_prefix = '/home/henryclever/IROS_Data'
        filepath_prefix_qt = '/home/henryclever/test'

    elif opt.computer == 'gorilla':
        filepath_prefix = '/home/henry/IROS_Data'
        filepath_prefix_qt = '/home/henry/test'

    if opt.vgg == True:
        name_prefix_train = 'trainfeat4xup'
        name_prefix_test = 'testfeat4xup'
    else:
        name_prefix_train = 'train_val'
        name_prefix_test = 'train_val'


    opt.subject2Path = filepath_prefix+'/subject_2/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject3Path = filepath_prefix+'/subject_3/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject4Path = filepath_prefix+'/subject_4/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject5Path = filepath_prefix+'/subject_5/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject6Path = filepath_prefix+'/subject_6/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject7Path = filepath_prefix+'/subject_7/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject8Path = filepath_prefix+'/subject_8/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject9Path = filepath_prefix+'/subject_9/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject10Path = filepath_prefix+'/subject_10/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject11Path = filepath_prefix+'/subject_11/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject12Path = filepath_prefix+'/subject_12/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject13Path = filepath_prefix+'/subject_13/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject14Path = filepath_prefix+'/subject_14/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject15Path = filepath_prefix+'/subject_15/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject16Path = filepath_prefix+'/subject_16/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject17Path = filepath_prefix+'/subject_17/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject18Path = filepath_prefix+'/subject_18/p_files/'+name_prefix_train+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject9PathB = filepath_prefix+'/subject_9/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject10PathB = filepath_prefix+'/subject_10/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject11PathB = filepath_prefix+'/subject_11/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject12PathB = filepath_prefix+'/subject_12/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject13PathB = filepath_prefix+'/subject_13/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject14PathB = filepath_prefix+'/subject_14/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject15PathB = filepath_prefix+'/subject_15/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject16PathB = filepath_prefix+'/subject_16/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject17PathB = filepath_prefix+'/subject_17/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject18PathB = filepath_prefix+'/subject_18/p_files/'+name_prefix_train+'_sit175rlh_sit120rll'+filetag+'.p'

    opt.subject2PathTest = filepath_prefix+'/subject_2/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject3PathTest = filepath_prefix+'/subject_3/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject4PathTest = filepath_prefix+'/subject_4/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject5PathTest = filepath_prefix+'/subject_5/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject6PathTest = filepath_prefix+'/subject_6/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject7PathTest = filepath_prefix+'/subject_7/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject8PathTest = filepath_prefix+'/subject_8/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_150rll_sit175rlh_sit120rll.p'
    opt.subject9PathTest = filepath_prefix+'/subject_9/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject10PathTest = filepath_prefix+'/subject_10/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject11PathTest = filepath_prefix+'/subject_11/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject12PathTest = filepath_prefix+'/subject_12/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject13PathTest = filepath_prefix+'/subject_13/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject14PathTest = filepath_prefix+'/subject_14/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject15PathTest = filepath_prefix+'/subject_15/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject16PathTest = filepath_prefix+'/subject_16/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject17PathTest = filepath_prefix+'/subject_17/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject18PathTest = filepath_prefix+'/subject_18/p_files/'+name_prefix_test+'_200rlh1_115rlh2_75rlh3_175rllair'+filetag+'.p'
    opt.subject9PathBTest = filepath_prefix+'/subject_9/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject10PathBTest = filepath_prefix+'/subject_10/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject11PathBTest = filepath_prefix+'/subject_11/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject12PathBTest = filepath_prefix+'/subject_12/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject13PathBTest = filepath_prefix+'/subject_13/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject14PathBTest = filepath_prefix+'/subject_14/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject15PathBTest = filepath_prefix+'/subject_15/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject16PathBTest = filepath_prefix+'/subject_16/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject17PathBTest = filepath_prefix+'/subject_17/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'
    opt.subject18PathBTest = filepath_prefix+'/subject_18/p_files/'+name_prefix_test+'_sit175rlh_sit120rll'+filetag+'.p'

    #shortcut:



    test_database_file = []
    training_database_file = []
    if opt.leave_out == 4:
        test_database_file.append(opt.subject4PathTest)
        training_database_file.append(opt.subject2Path)
        training_database_file.append(opt.subject3Path)
        training_database_file.append(opt.subject5Path)
        training_database_file.append(opt.subject6Path)
        training_database_file.append(opt.subject7Path)
        training_database_file.append(opt.subject8Path)

    elif opt.leave_out == 2:
        test_database_file.append(opt.subject2PathTest)
        training_database_file.append(opt.subject3Path)
        training_database_file.append(opt.subject4Path)
        training_database_file.append(opt.subject5Path)
        training_database_file.append(opt.subject6Path)
        training_database_file.append(opt.subject7Path)
        training_database_file.append(opt.subject8Path)

    elif opt.leave_out == 9:
        test_database_file.append(opt.subject9PathTest)
        test_database_file.append(opt.subject9PathBTest)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject17PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 10:
        test_database_file.append(opt.subject10PathTest)
        test_database_file.append(opt.subject10PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject17PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 11:
        test_database_file.append(opt.subject11PathTest)
        test_database_file.append(opt.subject11PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject17PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 12:
        test_database_file.append(opt.subject12PathTest)
        test_database_file.append(opt.subject12PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject17PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 13:
        test_database_file.append(opt.subject13PathTest)
        test_database_file.append(opt.subject13PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject17PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 14:
        test_database_file.append(opt.subject14PathTest)
        test_database_file.append(opt.subject14PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject17PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 15:
        test_database_file.append(opt.subject15PathTest)
        test_database_file.append(opt.subject15PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject17PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 16:
        test_database_file.append(opt.subject16PathTest)
        test_database_file.append(opt.subject16PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject17PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 17:
        test_database_file.append(opt.subject17PathTest)
        test_database_file.append(opt.subject17PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject18Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject18PathB)
    elif opt.leave_out == 18:
        test_database_file.append(opt.subject18PathTest)
        test_database_file.append(opt.subject18PathBTest)
        training_database_file.append(opt.subject9Path)
        training_database_file.append(opt.subject10Path)
        training_database_file.append(opt.subject11Path)
        training_database_file.append(opt.subject12Path)
        training_database_file.append(opt.subject13Path)
        training_database_file.append(opt.subject14Path)
        training_database_file.append(opt.subject15Path)
        training_database_file.append(opt.subject16Path)
        training_database_file.append(opt.subject17Path)
        training_database_file.append(opt.subject9PathB)
        training_database_file.append(opt.subject10PathB)
        training_database_file.append(opt.subject11PathB)
        training_database_file.append(opt.subject12PathB)
        training_database_file.append(opt.subject13PathB)
        training_database_file.append(opt.subject14PathB)
        training_database_file.append(opt.subject15PathB)
        training_database_file.append(opt.subject16PathB)
        training_database_file.append(opt.subject17PathB)


    elif opt.quick_test == True:
        if opt.vgg == True:
            test_database_file.append(filepath_prefix_qt + '/testfeat4xup_s4_150rh1_sit120rh.p')
            training_database_file.append(filepath_prefix_qt + '/trainfeat4xup_s8_150rh1_sit120rh.p')
        else:
            test_database_file.append(filepath_prefix_qt+'/trainval4_150rh1_sit120rh.p')
            training_database_file.append(filepath_prefix_qt+'/trainval8_150rh1_sit120rh.p')


    else:
        print 'please specify which subject to leave out for validation using --leave_out _'


    #test_database_file = []
    #training_database_file = []

    #test_database_file.append(opt.subject18PathB)
    #training_database_file.append(opt.subject9Path)

    print opt.testPath, 'testpath'
    print opt.modelPath, 'modelpath'



    test_bool = opt.only_test#Whether you want only testing done


    print test_bool, 'test_bool'
    print test_database_file, 'test database file'

    p = PhysicalTrainer(training_database_file, test_database_file, opt)

    if test_bool == True:
        trained_model = load_pickle(opt.modelPath+'/'+training_type+'.p')#Where the trained model is
        p.test_learning_algorithm(trained_model)
        sys.exit()
    else:
        if opt.verbose == True: print 'Beginning Learning'



        if opt.mltype == 'convnet':
            p.init_convnet_train()
        elif opt.mltype != 'convnet':
            p.baseline_train(opt)

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
