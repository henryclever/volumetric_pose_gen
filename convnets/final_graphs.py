#!/usr/bin/env python
import sys
import os
import numpy as np
import cPickle as pkl
import random
import math

# ROS
#import roslib; roslib.load_manifest('hrl_pose_estimation')

# Graphics
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import pylab as pylab


# Machine Learning
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
import scipy.stats as ss
from scipy.stats import ttest_ind
## from skimage import data, color, exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

#ROS libs
import rospkg
import roslib
import rospy
import tf.transformations as tft
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


# HRL libraries
import pickle
#roslib.load_manifest('hrl_lib')
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Pose Estimation Libraries
from visualization_lib import VisualizationLib
from kinematics_lib import KinematicsLib
from preprocessing_lib import PreprocessingLib
from synthetic_lib import SyntheticLib

#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable



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


class DataVisualizer():
    '''Gets the directory of pkl database and iteratively go through each file,
    cutting up the pressure maps and creating synthetic database'''
    def __init__(self, pkl_directory,  opt):
        self.opt = opt
        self.sitting = False
        self.old = False
        self.normalize = True
        self.include_inter = True
        # Set initial parameters
        self.subject = 4
        self.dump_path = pkl_directory.rstrip('/')

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)

        self.output_size = (10, 3)


        from scipy.signal import savgol_filter
        fig = plt.figure()

        plt.subplot(3, 2, 1)
        plt.axis([0,200,2000,10000])
        #plt.yticks([])
        train_val_loss1 = load_pickle(self.dump_path + '/planesreg/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_100e_000005lr.p')
        for key in train_val_loss1:
            print key
        train_val_loss1_cor = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.p')
        for key in train_val_loss1_cor:
            print key

        y1 = savgol_filter(train_val_loss1['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin'], 501, 3)
        corrected_y1 = savgol_filter(train_val_loss1_cor['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss1['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin'],
                 y1, 'k')
        plt.plot(np.array(train_val_loss1_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
                          corrected_y1, 'k')




        plt.subplot(3, 2, 3)
        plt.axis([0,200,1000,5000])
        #plt.yticks([])
        train_val_loss2 = load_pickle(self.dump_path + '/planesreg/convnet_losses_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_100e_000005lr.p')
        for key in train_val_loss2:
            print key
        train_val_loss2_cor = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.p')
        for key in train_val_loss2_cor:
            print key

        y2 = savgol_filter(train_val_loss2['train_anglesEU_synth_s9_3xreal_128b_101e_0.4rtojtdpth_pmatcntin'], 501, 3)
        corrected_y2 = savgol_filter(train_val_loss2_cor['train_anglesEU_synth_s9_3xreal_128b_101e_0.4rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss2['epoch_anglesEU_synth_s9_3xreal_128b_101e_0.4rtojtdpth_pmatcntin'],
                 y2, 'b')
        plt.plot(np.array(train_val_loss2_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_0.4rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
                          corrected_y2, 'b')



        plt.subplot(3, 2, 5)
        plt.axis([0,200,500,2500])
        #plt.yticks([])
        train_val_loss3 = load_pickle(self.dump_path + '/planesreg/convnet_losses_anglesEU_synth_s9_3xreal_128b_0.1rtojtdpth_pmatcntin_100e_000005lr.p')
        for key in train_val_loss3:
            print key
        y2 = savgol_filter(train_val_loss3['train_anglesEU_synth_s9_3xreal_128b_101e_0.04rtojtdpth_pmatcntin'], 501, 3)

        plt.plot(train_val_loss3['epoch_anglesEU_synth_s9_3xreal_128b_101e_0.04rtojtdpth_pmatcntin'],
                 y2, 'r')





        plt.subplot(3, 2, 2)
        plt.axis([0,200,2000,10000])
        #plt.yticks([])
        train_val_loss4 = load_pickle(self.dump_path + '/planesreg/old/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_100e_000002lr.p')
        train_val_loss4_cor1 = load_pickle(self.dump_path + '/planesreg_correction/old/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr_betasreg.p')
        train_val_loss4_cor2 = load_pickle(self.dump_path + '/planesreg_correction/old/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr_betashold.p')

        y4 = savgol_filter(train_val_loss4['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin'], 501, 3)
        corrected_y4_1 = savgol_filter(train_val_loss4_cor1['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)
        corrected_y4_2 = savgol_filter(train_val_loss4_cor2['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss4['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin'],
                 y4, 'k')
        plt.plot(np.array(train_val_loss4_cor1['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
                          corrected_y4_1, 'k')
        plt.plot(np.array(train_val_loss4_cor2['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
                          corrected_y4_2, 'r')



        plt.subplot(3, 2, 4)
        plt.axis([0,200,1000,5000])
        #plt.yticks([])
        #train_val_loss5 = load_pickle(self.dump_path + '/planesreg/old/convnet_losses_anglesEU_synth_s9_3xreal_128b_0.5rtojtdpth_pmatcntin_100e_000002lr.p')
        #for key in train_val_loss5:
        #    print key
        train_val_loss5_cor = load_pickle(self.dump_path + '/planesreg_correction/old/convnet_losses_anglesEU_synth_s9_3xreal_128b_0.5rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr_betasreg.p')
        for key in train_val_loss5_cor:
            print key

        #y5 = savgol_filter(train_val_loss5['train_anglesEU_synth_s9_3xreal_128b_101e_0.5rtojtdpth_pmatcntin'], 51, 3)
        corrected_y5 = savgol_filter(train_val_loss5_cor['train_anglesEU_synth_s9_3xreal_128b_101e_0.5rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        #plt.plot(train_val_loss5['epoch_anglesEU_synth_s9_3xreal_128b_101e_0.5rtojtdpth_pmatcntin'],
        #         y5, 'k')
        plt.plot(np.array(train_val_loss5_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_0.5rtojtdpth_pmatcntin_depthestin_angleadj'])+175,
                          corrected_y5, 'k')



        plt.subplot(3, 2, 6)
        plt.axis([0,200,500,2500])
        #plt.yticks([])
        train_val_loss6 = load_pickle(self.dump_path + '/planesreg/old/convnet_losses_anglesEU_synth_s9_3xreal_128b_0.05rtojtdpth_pmatcntin_100e_000002lr.p')
        for key in train_val_loss6:
            print key
        y6 = savgol_filter(train_val_loss6['train_anglesEU_synth_s9_3xreal_128b_101e_0.05rtojtdpth_pmatcntin'], 501, 3)

        plt.plot(train_val_loss6['epoch_anglesEU_synth_s9_3xreal_128b_101e_0.05rtojtdpth_pmatcntin'],
                 y6, 'k')





        plt.legend()
        #plt.ylabel('Mean squared error loss over 30 joint vectors')
        #plt.title('Subject 1 laying validation Loss, training performed on subjects 2, 3, 4, 5, 6, 7, 8')



        #plt.axis([0,410,0,30000])
        #plt.axis([0, 200, 10, 15])
        #if self.opt.visualize == True:
        plt.show()
        #plt.close()


if __name__ == "__main__":

    import optparse
    p = optparse.OptionParser()

    p.add_option('--viz', action='store_true',
                 dest='visualize', \
                 default=False, \
                 help='Train only on data from the arms, both sitting and laying.')

    opt, args = p.parse_args()

    #Path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials/'
    #Path = '/home/henry/data/synth/'
    Path = '/media/henry/multimodal_data_2/data/convnets/'

    #Initialize trainer with a training database file
    p = DataVisualizer(pkl_directory=Path, opt = opt)

    #p.all_joint_error()
    #p.dropout_std_threshold()
    #p.error_threshold()
    #p.p_information_std()
    #p.final_foot_variance()
    #p.all_joint_error()
    #p.final_error()
    sys.exit()

