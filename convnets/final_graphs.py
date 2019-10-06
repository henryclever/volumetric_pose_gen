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
        plt.title('ReLU, Training size: 32K')
        plt.axis([0,400,2000,3000])
        #plt.yticks([])
        train_val_loss1 = load_pickle(self.dump_path + '/planesreg/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_l2cnt_125e_00001lr.p')
        for key in train_val_loss1:
            print key
        train_val_loss2 = load_pickle(self.dump_path + '/planesreg/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_l2cnt_200e_00001lr.p')
        for key in train_val_loss2:
            print key
        train_val_loss3 = load_pickle(self.dump_path + '/planesreg/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_l2cnt_300e_00001lr.p')
        for key in train_val_loss3:
            print key
        train_val_loss4 = load_pickle(self.dump_path + '/planesreg_correction/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt_125e_200e_00001lr.p')
        for key in train_val_loss4:
            print key
        train_val_loss5 = load_pickle(self.dump_path + '/planesreg_correction/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt_125e_300e_00001lr.p')
        for key in train_val_loss5:
            print key
        #train_val_loss1_cor = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.p')
        y1 = savgol_filter(train_val_loss1['train_anglesDC_synth_32000_128b_201e_x5pmult_0.5rtojtdpth_l2cnt'], 501, 3)
        y1_val = savgol_filter(train_val_loss1['val_anglesDC_synth_32000_128b_201e_x5pmult_0.5rtojtdpth_l2cnt'], 501, 3)
        y2 = savgol_filter(train_val_loss2['train_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_l2cnt'], 501, 3)
        y2_val = savgol_filter(train_val_loss2['val_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_l2cnt'], 501, 3)
        y3 = savgol_filter(train_val_loss3['train_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt'], 501, 3)
        y3_val = savgol_filter(train_val_loss3['val_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt'], 501, 3)
        y4 = savgol_filter(train_val_loss4['train_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt'], 501, 3)
        y4_val = savgol_filter(train_val_loss4['val_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt'], 501, 3)
        y5 = savgol_filter(train_val_loss5['train_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt'], 501, 3)
        y5_val = savgol_filter(train_val_loss5['val_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt'], 501, 3)
        #corrected_y1 = savgol_filter(train_val_loss1_cor['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss1['epoch_anglesDC_synth_32000_128b_201e_x5pmult_0.5rtojtdpth_l2cnt'], y1, color='#fd8d3c')
        plt.plot(train_val_loss1['epoch_anglesDC_synth_32000_128b_201e_x5pmult_0.5rtojtdpth_l2cnt'], y1_val, color='#9e9ac8')
        plt.plot(np.array(train_val_loss4['epoch_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt'])+125., y4, color='#d94701')
        plt.plot(np.array(train_val_loss4['epoch_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt'])+125., y4_val, color='#6a51a3')
        plt.plot(np.array(train_val_loss2['epoch_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_l2cnt'])+125., y2, color='#fd8d3c')
        plt.plot(np.array(train_val_loss2['epoch_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_l2cnt'])+125., y2_val, color='#9e9ac8')
        plt.plot(np.array(train_val_loss3['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt'])+200., y3, color='#fd8d3c')
        plt.plot(np.array(train_val_loss3['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt'])+200., y3_val, color='#9e9ac8')
        plt.plot(np.array(train_val_loss5['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt'])+200., y5, color='#d94701')
        plt.plot(np.array(train_val_loss5['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_l2cnt'])+200., y5_val, color='#6a51a3')
        #plt.plot(np.array(train_val_loss1_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
        #                  corrected_y1, 'r')
        plt.legend(['relu', 'relu val',' relu res', 'relu val res'])
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel('loss')



        plt.subplot(3, 2, 2)
        plt.title('tanh, Training size: 32K')
        plt.axis([0,400,2000,3000])
        #plt.yticks([])
        train_val_loss1 = load_pickle(self.dump_path + '/planesreg/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_125e_00001lr.p')
        for key in train_val_loss1:
            print key
        train_val_loss2 = load_pickle(self.dump_path + '/planesreg/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_200e_00001lr.p')
        for key in train_val_loss2:
            print key
        train_val_loss3 = load_pickle(self.dump_path + '/planesreg/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_300e_00001lr.p')
        for key in train_val_loss3:
            print key
        train_val_loss4 = load_pickle(self.dump_path + '/planesreg_correction/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt_125e_200e_00001lr.p')
        for key in train_val_loss4:
            print key
        train_val_loss5 = load_pickle(self.dump_path + '/planesreg/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_400e_00001lr.p')
        for key in train_val_loss5:
            print key
        train_val_loss6 = load_pickle(self.dump_path + '/planesreg_correction/32K/convnet_losses_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt_125e_300e_00001lr.p')
        for key in train_val_loss6:
            print key
        #train_val_loss1_cor = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.p')
        y1 = savgol_filter(train_val_loss1['train_anglesDC_synth_32000_128b_201e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y1_val = savgol_filter(train_val_loss1['val_anglesDC_synth_32000_128b_201e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y2 = savgol_filter(train_val_loss2['train_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y2_val = savgol_filter(train_val_loss2['val_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y3 = savgol_filter(train_val_loss3['train_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y3_val = savgol_filter(train_val_loss3['val_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y4 = savgol_filter(train_val_loss4['train_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'], 501, 3)
        y4_val = savgol_filter(train_val_loss4['val_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'], 501, 3)
        y5 = savgol_filter(train_val_loss5['train_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y5_val = savgol_filter(train_val_loss5['val_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y6 = savgol_filter(train_val_loss6['train_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'], 501, 3)
        y6_val = savgol_filter(train_val_loss6['val_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'], 501, 3)
        #corrected_y1 = savgol_filter(train_val_loss1_cor['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss1['epoch_anglesDC_synth_32000_128b_201e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], y1, color='#fd8d3c')
        plt.plot(train_val_loss1['epoch_anglesDC_synth_32000_128b_201e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], y1_val, color='#9e9ac8')
        plt.plot(np.array(train_val_loss4['epoch_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'])+125., y4, color='#d94701')
        plt.plot(np.array(train_val_loss4['epoch_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'])+125., y4_val, color='#6a51a3')
        plt.plot(np.array(train_val_loss2['epoch_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'])+125., y2, color='#fd8d3c')
        plt.plot(np.array(train_val_loss2['epoch_anglesDC_synth_32000_128b_75e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'])+125., y2_val, color='#9e9ac8')
        plt.plot(np.array(train_val_loss3['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'])+200., y3, color='#fd8d3c')
        plt.plot(np.array(train_val_loss3['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'])+200., y3_val, color='#9e9ac8')
        plt.plot(np.array(train_val_loss5['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'])+300., y5, color='#fd8d3c')
        plt.plot(np.array(train_val_loss5['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'])+300., y5_val, color='#9e9ac8')
        plt.plot(np.array(train_val_loss6['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'])+200., y6, color='#d94701')
        plt.plot(np.array(train_val_loss6['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'])+200., y6_val, color='#6a51a3')
        #plt.plot(np.array(train_val_loss1_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
        #                  corrected_y1, 'r')
        #plt.legend(['x5 tanh l2', 'x5 tanh val l2'])
        plt.legend(['tanh', 'tanh val',' tanh res', 'tanh val res'])
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel('loss')




        plt.subplot(3, 2, 3)
        plt.tight_layout()
        plt.title('ReLU, Training size: 112K')
        plt.axis([0,300,2000,3000])
        #plt.yticks([])
        train_val_loss1 = load_pickle(self.dump_path + '/planesreg/112K/convnet_losses_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_l2cnt_100e_00001lr.p')
        for key in train_val_loss1:
            print key
        #train_val_loss1_cor = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.p')
        y1 = savgol_filter(train_val_loss1['train_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt'], 501, 3)
        y1_val = savgol_filter(train_val_loss1['val_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt'], 501, 3)
        #corrected_y1 = savgol_filter(train_val_loss1_cor['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss1['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt'], y1, color='#fd8d3c')
        plt.plot(train_val_loss1['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt'], y1_val, color='#9e9ac8')
        #plt.plot(np.array(train_val_loss1_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
        #                  corrected_y1, 'r')
        plt.legend(['relu', 'relu val'])
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel('loss')




        plt.subplot(3, 2, 4)
        #plt.tight_layout()
        plt.title('tanh, Training size: 112K')
        plt.axis([0,300,2000,3000])
        #plt.yticks([])
        train_val_loss1 = load_pickle(self.dump_path + '/planesreg/112K/convnet_losses_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_100e_00001lr.p')
        for key in train_val_loss1:
            print key
        train_val_loss2 = load_pickle(self.dump_path + '/planesreg_correction/112K/convnet_losses_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt_50e_150e_00001lr.p')
        for key in train_val_loss2:
            print key
        train_val_loss3 = load_pickle(self.dump_path + '/planesreg_correction/112K/convnet_losses_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt_100e_150e_00001lr.p')
        for key in train_val_loss3:
            print key
        train_val_loss4 = load_pickle(self.dump_path + '/planesreg/112K/convnet_losses_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_100e_200e_00001lr.p')
        for key in train_val_loss4:
            print key
        #train_val_loss1_cor = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.p')
        y1 = savgol_filter(train_val_loss1['train_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y1_val = savgol_filter(train_val_loss1['val_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y2 = savgol_filter(train_val_loss2['train_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'], 501, 3)
        y2_val = savgol_filter(train_val_loss2['val_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'], 501, 3)
        y3 = savgol_filter(train_val_loss3['train_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'], 501, 3)
        y3_val = savgol_filter(train_val_loss3['val_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'], 501, 3)
        y4 = savgol_filter(train_val_loss4['train_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        y4_val = savgol_filter(train_val_loss4['val_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], 501, 3)
        #corrected_y1 = savgol_filter(train_val_loss1_cor['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss1['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], y1, color='#fd8d3c')
        plt.plot(train_val_loss1['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'], y1_val, color='#9e9ac8')
        plt.plot(np.array(train_val_loss2['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'])+50., y2, color='#d94701')
        plt.plot(np.array(train_val_loss2['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'])+50., y2_val, color='#6a51a3')
        plt.plot(np.array(train_val_loss3['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'])+100., y3, color='#d94701')
        plt.plot(np.array(train_val_loss3['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt'])+100., y3_val, color='#6a51a3')
        plt.plot(np.array(train_val_loss4['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'])+100., y4, color='#fd8d3c')
        plt.plot(np.array(train_val_loss4['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt'])+100., y4_val, color='#9e9ac8')
        #plt.plot(np.array(train_val_loss1_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
        #                  corrected_y1, 'r')
        plt.legend(['tanh', 'tanh val',' tanh res', 'tanh val res'])
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel('loss')





        plt.subplot(3, 2, 5)
        plt.tight_layout()
        plt.title('ReLU calib. noise, Training size: 112K')
        plt.axis([0,300,2000,3000])
        #plt.yticks([])
        train_val_loss1 = load_pickle(self.dump_path + '/planesreg/112K/convnet_losses_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_l2cnt_calnoise_100e_00001lr.p')
        for key in train_val_loss1:
            print key
        #train_val_loss1_cor = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.p')
        y1 = savgol_filter(train_val_loss1['train_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt_calnoise'], 501, 3)
        y1_val = savgol_filter(train_val_loss1['val_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt_calnoise'], 501, 3)
        #corrected_y1 = savgol_filter(train_val_loss1_cor['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss1['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt_calnoise'], y1, color='#fd8d3c')
        plt.plot(train_val_loss1['epoch_anglesDC_synth_112000_128b_100e_x5pmult_0.5rtojtdpth_l2cnt_calnoise'], y1_val, color='#9e9ac8')
        #plt.plot(np.array(train_val_loss1_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
        #                  corrected_y1, 'r')
        plt.legend(['relu', 'relu val'])
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel('loss')





        plt.subplot(3, 2, 6)
        plt.tight_layout()
        plt.title('tanh calib. noise, Training size: 112K')
        plt.axis([0,300,2000,3000])
        #plt.yticks([])
        train_val_loss1 = load_pickle(self.dump_path + '/planesreg/112K/convnet_losses_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_calnoise_100e_00001lr.p')
        for key in train_val_loss1:
            print key
        #train_val_loss1_cor = load_pickle(self.dump_path + '/planesreg_correction/convnet_losses_anglesEU_synth_s9_3xreal_128b_1.0rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.p')
        y1 = savgol_filter(train_val_loss1['train_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt_calnoise'], 501, 3)
        y1_val = savgol_filter(train_val_loss1['val_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt_calnoise'], 501, 3)
        #corrected_y1 = savgol_filter(train_val_loss1_cor['train_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'], 501, 3)

        plt.plot(train_val_loss1['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt_calnoise'], y1, color='#fd8d3c')
        plt.plot(train_val_loss1['epoch_anglesDC_synth_32000_128b_100e_x5pmult_0.5rtojtdpth_alltanh_l2cnt_calnoise'], y1_val, color='#9e9ac8')
        #plt.plot(np.array(train_val_loss1_cor['epoch_anglesEU_synth_s9_3xreal_128b_101e_1.0rtojtdpth_pmatcntin_depthestin_angleadj'])+100,
        #                  corrected_y1, 'r')
        plt.legend(['relu', 'relu val'])
        plt.grid()
        plt.xlabel('epochs')
        plt.ylabel('loss')







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
    Path = '/home/henry/data/convnets/'
    #Path = '/media/henry/multimodal_data_2/data/convnets/'

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

