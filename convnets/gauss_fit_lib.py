#!/usr/bin/env python
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

from kinematics_lib import KinematicsLib

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 84#73 #taxels
NUMOFTAXELS_Y = 47#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)

import scipy.optimize as opt
import pylab as pylab

import itertools
from scipy import linalg
import matplotlib as mpl

from sklearn import mixture

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


# import hrl_lib.util as ut
import cPickle as pickle
# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class GaussFitLib():

    def __init__(self):
        self.up_mult = 1

    def twoD_Gaussian(self, (x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                           + c * ((y - yo) ** 2)))
        return g.ravel()


    def single_fit(self, pmat_sample):# Create x and y indices
        y = np.linspace(0, 83, 84)
        x = np.linspace(0, 46, 47)
        x, y = np.meshgrid(x, y)

        print x.shape,y.shape

        # create data
        #data = self.twoD_Gaussian((x, y), 3, 100, 100, 10, 20, 0, 10).reshape(201, 201)
        #data *= 2
        #data[:, 10:] = data[:, 0:-10]

        # plot twoD_Gaussian data generated above
        data = pmat_sample
        pylab.figure()
        pylab.imshow(data)
        pylab.colorbar()

        #data=data.reshape(201*201)

        data = pmat_sample.reshape(84*47)

        # add some noise to the data and try to fit the data generated beforehand
        initial_guess = (3, 21, 41, 20, 40, 0, 10)

        data_noisy = data + 0.2 * np.random.normal(size=data.shape)

        popt, pcov = opt.curve_fit(self.twoD_Gaussian, (x, y), data_noisy, p0=initial_guess)

        data_fitted = self.twoD_Gaussian((x, y), *popt)

        fig, ax = pylab.subplots(1, 1)
        ax.hold(True)
        ax.imshow(data_noisy.reshape(84, 47), cmap=pylab.cm.jet, origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, data_fitted.reshape(84, 47), 8, colors='w')
        pylab.show()




    def plot_results(self, X, Y_, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        #plt.xlim(-9., 5.)
        #plt.ylim(-3., 6.)
        plt.xlim(-1., 64.*self.up_mult)
        plt.ylim(-1., 27.*self.up_mult)
        #plt.xticks(())
        #plt.yticks(())
        plt.title(title)


    def get_pressure_under_legs(self, im, targets):

        im_expanded = np.zeros((im.shape[0]+20, im.shape[1]+20))
        im_expanded[10:-10, 10:-10] = im
        im = im_expanded

        targets = targets.reshape(24, 3)
        targets = np.concatenate((targets[1:2, :],
                                  targets[2:3, :],
                                  targets[4:5, :],
                                  targets[5:6, :],
                                  targets[7:8, :],
                                  targets[8:9, :],
                                  ), axis = 0)

        targets /= 0.0286
        targets += 0.5
        targets = targets[:, 0:2].astype(int)

        new_im = zeros(shape(im))# + 7

        for idx in range(len(targets)):
            targets[idx][1] = 83 - targets[idx][1]

            tar1 = targets[idx][1]
            tar0 = targets[idx][0]
            if tar1 >= 0 and tar1 < 84 and tar0 >= 0 and tar0 < 47:

                #im[tar1, tar0] = 50

                #print "GOT HERE"
                if idx < 2:
                     new_im[tar1 - 4: tar1 + 5, tar0 - 4: tar0 + 5] = np.copy(im[tar1 - 4: tar1 + 5, tar0 - 4: tar0 + 5])
                elif idx >= 2 and idx < 4:
                     new_im[tar1 - 7: tar1 + 8, tar0 - 7: tar0 + 8] = np.copy(im[tar1 - 7: tar1 + 8, tar0 - 7: tar0 + 8])
                elif idx >= 4 and idx < 6:
                     new_im[tar1 - 10: tar1 + 11, tar0 - 10: tar0 + 11] = np.copy(im[tar1 - 10: tar1 + 11, tar0 - 10: tar0 + 11])

        new_im = new_im[10:-10, 10:-10]

        self.GMM_fit_dots(new_im)
        return new_im








    def GMM_fit_dots(self, pmat_sample):

        self.up_mult = 1
        pmat_sample = ndimage.zoom(pmat_sample, self.up_mult, order=1)
        #pmat_sample[pmat_sample > 0] = 0
        # Number of samples per component
        #n_samples = 500
        self.up_mult = 1
        #pmat_sample = pmat_sample[64:, 0:27]

        # Generate random sample, two components
        #np.random.seed(0)
        #C = np.array([[0., -0.1], [1.7, .4]])
        #X = np.r_[np.dot(np.random.randn(n_samples, 2), C), .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

        print pmat_sample.shape
        print np.max(pmat_sample)

        #print X
        X = []
        for i in range(64*self.up_mult):
            for j in range(27*self.up_mult):
                if pmat_sample[i, j] > 0:
                    for val in range(int(pmat_sample[i, j])):
                        X.append([i, j])

        X = np.array(X)
        print X

        # Fit a Gaussian mixture with EM using five components
        gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
        self.plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                         'Gaussian Mixture')

        # Fit a Dirichlet process Gaussian mixture using five components
        dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full').fit(X)
        self.plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                         'Bayesian Gaussian Mixture with a Dirichlet process prior')

        plt.show()


if __name__ == "__main__":
    GFL = GaussFitLib()

    subj_dat = load_pickle('/home/henry/data/real/trainval4_150rh1_sit120rh.p')
    pmat_sample = np.array(subj_dat['images'][0]).reshape(84, 47)[10:-10, 10:-10]
    #GFL.single_fit(pmat_sample)

    GFL.GMM_fit_dots(pmat_sample)

