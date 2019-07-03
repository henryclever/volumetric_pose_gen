import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

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
from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
from sklearn import metrics
from sklearn.utils import shuffle



# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable











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

    p.add_option('--log_interval', type=int, default=1, metavar='N',
                 help='number of batches between logging train status')

    opt, args = p.parse_args()


    import time
    for i in range(10):
        print i
        time.sleep(0.1)


    for j in range(20):
        print j
        time.sleep(0.1)

