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




        train_val_loss = load_pickle(self.dump_path + '/planesreg/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_100e_000005lr.p')
        for key in train_val_loss:
            print key
        #train_val_loss_2reg = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr_betasreg.p')
        #for key in train_val_loss_2reg:
        #    print key

        #train_val_loss_2hold = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr_betashold.p')
        #for key in train_val_loss_2reg:
        #    print key





        plt.plot(train_val_loss['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin'],
                 train_val_loss['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin'], 'k')
        #plt.plot(np.array(train_val_loss_2reg['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
        #         train_val_loss_2reg['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 'g')
        #plt.plot(np.array(train_val_loss_2hold['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
        #         train_val_loss_2hold['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 'r')

        plt.legend()
        plt.ylabel('Mean squared error loss over 30 joint vectors')
        plt.title('Subject 1 laying validation Loss, training performed on subjects 2, 3, 4, 5, 6, 7, 8')



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

