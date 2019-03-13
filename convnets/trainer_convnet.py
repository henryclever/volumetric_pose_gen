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
from synthetic_lib import SyntheticLib
from visualization_lib import VisualizationLib
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

        self.num_epochs = 200
        self.include_inter = True
        self.shuffle = False#True


        self.count = 0


        print test_file
        print self.num_epochs, 'NUM EPOCHS!'
        #Entire pressure dataset with coordinates in world frame

        self.save_name = '_quicktest_' + opt.losstype+'_' +str(self.shuffle)+'s_' + str(self.batch_size) + 'b_' + str(self.num_epochs) + 'e'



        print 'appending to','train'+self.save_name
        self.train_val_losses = {}
        self.train_val_losses['train'+self.save_name] = []
        self.train_val_losses['val'+self.save_name] = []
        self.train_val_losses['epoch'+self.save_name] = []

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)


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


        self.train_x_flat = []  # Initialize the testing pressure mat list
        for entry in range(len(dat['images'])):
            self.train_x_flat.append(dat['images'][entry] * 3)

        self.train_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(dat['images'])):
            self.train_a_flat.append(dat['bed_angle_deg'][entry])
        train_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.train_x_flat, self.train_a_flat, self.include_inter, self.mat_size, self.verbose)
        train_xa = np.array(train_xa)

        self.train_x_tensor = torch.Tensor(train_xa)

        

        print self.train_x_tensor.shape, 'tensor shape'

        self.train_y_flat = [] #Initialize the training ground truth list
        for entry in range(len(dat['markers_xyz_m'])):
            if self.loss_vector_type == 'anglesR':
                #print dat['markers_xyz_m'][entry][0:2], dat['body_shape'][entry][0:2], dat['joint_angles'][entry][0:2]
                c = np.concatenate((dat['markers_xyz_m'][entry][0:72] * 1000,
                                    dat['body_shape'][entry][0:10],
                                    dat['joint_angles'][entry][0:72],
                                    dat['root_xyz_shift'][entry][0:3]), axis=0) # shapedirs (N, 6890, 3, 10)
                self.train_y_flat.append(c)
            else:
                self.train_y_flat.append(dat['markers_xyz_m'][entry][0:72] * 1000)



        if self.loss_vector_type == 'anglesR':
            from smpl.smpl_webuser.serialization import load_model
            model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
            self.m = load_model(model_path)

            self.v_template = torch.Tensor(np.array(self.m.v_template))
            self.shapedirs = torch.Tensor(np.array(self.m.shapedirs)).permute(0, 2, 1)

            self.J_regressor = np.zeros((self.m.J_regressor.shape)) + self.m.J_regressor
            self.J_regressor = torch.Tensor(np.array(self.J_regressor).astype(float)).permute(1, 0)




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


        self.test_x_flat = []  # Initialize the testing pressure mat list
        for entry in range(len(test_dat['images'])):
            self.test_x_flat.append(test_dat['images'][entry])

        self.test_a_flat = []  # Initialize the testing pressure mat angle list
        for entry in range(len(test_dat['images'])):
            self.test_a_flat.append(test_dat['bed_angle_deg'][entry])
        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat, self.test_a_flat, self.include_inter, (84, 47), self.verbose)
        test_xa = np.array(test_xa)
        self.test_x_tensor = torch.Tensor(test_xa)


        self.test_y_flat = []  # Initialize the ground truth list
        for entry in range(len(test_dat['markers_xyz_m'])):
            self.test_y_flat.append(test_dat['markers_xyz_m'][entry] * 1000)
        self.test_y_tensor = torch.Tensor(self.test_y_flat)

        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



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
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=self.shuffle)

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.batch_size, shuffle=self.shuffle)



        if self.loss_vector_type == 'anglesK':
            fc_output_size = 85#38 #18 angles for body, 17 lengths for body, 3 torso coordinates
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

        if self.loss_vector_type == 'anglesR':
            fc_output_size = 229# 10 + 3 + 24*3*3 --- betas, root shift, rotations
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type, self.m)
            #self.model = torch.load('/home/ubuntu/Autobed_OFFICIAL_Trials' + '/subject_' + str(self.opt.leave_out) + '/convnets/convnet_9to18_'+str(self.loss_vector_type)+'_sTrue_128b_200e_' + str(self.opt.leave_out) + '.pt', map_location=lambda storage, loc: storage)
            print 'LOADED!!!!!!!!!!!!!!!!!1'
            pp = 0
            for p in list(self.model.parameters()):
                nn = 1
                for s in list(p.size()):
                    nn = nn * s
                pp += nn
            print pp, 'num params'
        elif self.loss_vector_type == 'direct':
            fc_output_size = 72
            self.model = convnet.CNN(self.mat_size, fc_output_size, hidden_dim, kernel_size, self.loss_vector_type)

        # Run model on GPU if available
        if False:#torch.cuda.is_available():
            self.model = self.model.cuda()


        self.criterion = F.cross_entropy



        if self.loss_vector_type == None:
            self.optimizer2 = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=0.0005)
        elif self.loss_vector_type == 'anglesR' or self.loss_vector_type == 'direct':
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


        torch.save(self.model, '/home/henry/data/training/convnet'+self.save_name+'.pt')
        pkl.dump(self.train_val_losses,open('/home/henry/data/training/convnet_losses'+self.save_name+'.p', 'wb'))


    def train_convnet(self, epoch):
        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.
        self.model.train()
        scores = 0
        with torch.autograd.set_detect_anomaly(True):

            #This will loop a total = training_images/batch_size times
            for batch_idx, batch in enumerate(self.train_loader):

                if self.loss_vector_type == 'anglesR':

                    #0:72: positions. 72:82: betas. 82:154: angles.
                    batch.append(batch[1][:, 72:82])
                    batch.append(batch[1][:, 82:154])
                    batch.append(batch[1][:, 154:157])

                    #print batch[1][10:22, 70:84]
                    v_shaped = torch.matmul(batch[1][:, 72:82], self.shapedirs).permute(1, 0, 2) + self.v_template

                    Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
                    Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
                    Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)

                    J = torch.stack([Jx, Jy, Jz], dim=2) #these are the joint locations with home pose (pose is 0 degree on all angles)

                    J = J - J[:, 0:1, :] + batch[1][:, 154:157].unsqueeze(1)


                    Rs = self.model.batch_rodrigues(batch[1][:, 82:154].view(-1, 24, 3)).view(-1, 24, 3, 3)

                    #print Rs.shape #use beta on the end of the network to go through the model to get J. regress B. Also use Rs directly on the network.


                    self.J_transformed, A = self.model.batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

                    #batch.append(torch.cat((batch[1][:, 0:3], batch[1][:, 72:82], batch[1][:, 82:154]), dim = 1))

                    #print (self.J_transformed[13, :, :])*1000# - self.J_transformed[13, 0, 0:3] + batch[1][13, 154:157])*1000 #forward through model
                    #print (torch.Tensor(np.array(self.m.J_transformed)) - torch.Tensor(np.array(self.m.J_transformed[0, :])) + batch[1][1, 154:157])*1000


                    #print batch[1][:, 0:72].view(-1, 24, 3)[13, :, :] #ground truth


                    #cut it off so batch[2] is only the xyz marker targets
                    batch[1] = batch[1][:, 0:72]


                    images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy(), multiple = 2)))
                    images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)


                    #print images_up.shape

                    images, targets, betas = Variable(batch[0].type(dtype), requires_grad = False), Variable(batch[1].type(dtype), requires_grad = False), Variable(batch[2].type(dtype), requires_grad = False)


                    angles_gt = Variable(batch[3].type(dtype), requires_grad = True)
                    root_shift = Variable(batch[4].type(dtype), requires_grad = True)


                    self.optimizer.zero_grad()

                    ground_truth = np.zeros((batch[0].numpy().shape[0], 82)) #82 is 10 shape params and 72 joint locations x,y,z
                    ground_truth = Variable(torch.Tensor(ground_truth).type(dtype))
                    ground_truth[:, 0:10] = betas[:, 0:10]/100
                    ground_truth[:, 10:82] = targets[:, 0:72]/1000


                    scores_zeros = np.zeros((batch[0].numpy().shape[0], 34)) #34 is 10 shape params and 24 joint euclidean errors
                    scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))
                    scores_zeros[:, 0:10] = betas[:, 0:10]

                    scores, targets_est, _, betas_est = self.model.forward_kinematic_R(images_up, targets, is_training = True, betas = betas, angles_gt = angles_gt, root_shift = root_shift) # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.
                    #print lengths_est[0,0:10], 'lengths est'
                    #print batch[0][0,2,10,10], 'angle'

                    #print scores_zeros[0, :]
                    #print scores_zeros[13, :]

                    #print "targets output", targets_est[13, :]

                    self.criterion = nn.L1Loss()

                    loss = self.criterion(scores, scores_zeros)


                elif self.loss_vector_type == 'direct':


                    batch[0], batch[1], _ = SyntheticLib().synthetic_master(batch[0], batch[1], flip=True, shift=True,
                                                                            scale=True, bedangle=True,
                                                                            include_inter=self.include_inter,
                                                                            loss_vector_type=self.loss_vector_type)


                    images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, :, :], multiple = 2)))

                    images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)

                    images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad = False), Variable(batch[1].type(dtype), requires_grad = False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1]/3))).type(dtype), requires_grad = False)

                    self.optimizer.zero_grad()

                    scores, targets_est = self.model.forward_direct(images_up, targets, 'train')

                    self.criterion = nn.L1Loss()

                    loss = self.criterion(scores, scores_zeros)

                #print loss.data.numpy() * 1000, 'loss'

                loss.backward()
                self.optimizer.step()
                loss *= 1000


            if batch_idx % opt.log_interval == 0:
                if self.loss_vector_type == 'anglesR':
                    print targets.data.size()
                    print targets_est.shape

                    #VisualizationLib().print_error(targets.data, targets_est, self.output_size, self.loss_vector_type, data = 'train')
                    #print angles_est[0, :], 'angles'
                    print batch[0][0,2,10,10], 'bed angle'

                elif self.loss_vector_type == 'direct':
                    pass

                VisualizationLib().print_error_train(targets.data, targets_est, self.output_size_train, self.loss_vector_type, data='train')

                self.im_sample = images.data
                #self.im_sample = self.im_sample[:,1, :, :]
                self.im_sample = self.im_sample[0, :].squeeze()
                self.tar_sample = targets.data
                self.tar_sample = self.tar_sample[0, :].squeeze()/1000
                self.sc_sample = targets_est.clone()
                self.sc_sample = self.sc_sample[0, :].squeeze() / 1000
                self.sc_sample = self.sc_sample.view(self.output_size_train)










                val_loss = self.validate_convnet(n_batches=4)
                train_loss = loss.data.item()
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


            if self.loss_vector_type == 'anglesR':

                #print batch[1].shape

                #get the direct joint locations
                batch[1] = batch[1][:, 0:30]

                images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy(), multiple=2)))
                images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)
                images, targets = Variable(batch[0].type(dtype), volatile=True, requires_grad=False), Variable(batch[1].type(dtype), volatile=True,requires_grad=False),

                self.optimizer.zero_grad()


                ground_truth = np.zeros((batch[0].numpy().shape[0], 30))
                ground_truth = Variable(torch.Tensor(ground_truth).type(dtype))
                ground_truth[:, 0:30] = targets[:, 0:30]/1000

                scores_zeros = np.zeros((batch[0].numpy().shape[0], 10))
                scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))


                scores, targets_est, targets_est_reduced, betas_est = self.model.forward_kinematic_R(images_up, targets, is_training=False)  # scores is a variable with 27 for 10 euclidean errors and 17 lengths in meters. targets est is a numpy array in mm.

                self.criterion = nn.L1Loss()
                loss = self.criterion(scores, scores_zeros)
                loss = loss.data.item()



            elif self.loss_vector_type == 'direct':


                images_up_non_tensor = np.array(PreprocessingLib().preprocessing_pressure_map_upsample(batch[0].numpy()[:, :, :, :], multiple = 2))
                images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)
                images, targets, scores_zeros = Variable(batch[0].type(dtype), requires_grad=False), Variable(batch[1].type(dtype), requires_grad=False), Variable(torch.Tensor(np.zeros((batch[1].shape[0], batch[1].shape[1] / 3))).type(dtype), requires_grad=False)

                scores, targets_est = self.model.forward_direct(images_up, targets, 'val')

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


        VisualizationLib().print_error_val(targets.data, targets_est_reduced, self.output_size_val, self.loss_vector_type, data='validate')

        if self.loss_vector_type == 'anglesR':
            #print angles_est[0, :], 'validation angles'
            print betas_est[0, :], 'validation betas'


        #print batch[0][0,2,10,10].item(), 'validation bed angle'
        self.im_sampleval = images.data
        # #self.im_sampleval = self.im_sampleval[:,0,:,:]
        self.im_sampleval = self.im_sampleval[0, :].squeeze()
        self.tar_sampleval = targets.data #this is just 10 x 3
        self.tar_sampleval = self.tar_sampleval[0, :].squeeze() / 1000
        self.sc_sampleval = targets_est #score space is larger is 72 x3
        self.sc_sampleval = self.sc_sampleval[0, :].squeeze() / 1000
        self.sc_sampleval = self.sc_sampleval.view(24, 3)

        if self.opt.visualize == True:
            VisualizationLib().visualize_pressure_map(self.im_sample, self.tar_sample, self.sc_sample,self.im_sampleval, self.tar_sampleval, self.sc_sampleval, block=False)


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

    p.add_option('--log_interval', type=int, default=10, metavar='N',
                 help='number of batches between logging train status')

    opt, args = p.parse_args()

    filepath_prefix_qt = '/home/henry/data'

    training_database_file = []
    test_database_file = []

    training_database_file.append(filepath_prefix_qt+'/training/train_m_sit_95_rightside_stiff.p')
    test_database_file.append(filepath_prefix_qt+'/testing/trainval4_150rh1_sit120rh.p')


    p = PhysicalTrainer(training_database_file, test_database_file, opt)

    p.init_convnet_train()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
