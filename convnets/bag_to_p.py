#!/usr/bin/env python

#By Henry M. Clever
#The original bag_to_p.py requires replaying the bag files at original speed, which is cumbersome. 
#This version speeds up the latter and makes a pickle file that is better annotated


import rospy, roslib
import sys, os
import random, math
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
from time import sleep
from scipy import ndimage
from hrl_msgs.msg import FloatArrayBare
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
import rosbag
import copy

roslib.load_manifest('hrl_lib')
from hrl_lib.util import save_pickle, load_pickle



MAT_WIDTH = 0.74#0.762 #metres
MAT_HEIGHT = 1.75 #1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2 
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30 
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1) 
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1) 

class BagfileToPickle():
    '''Detects the head of a person sleeping on the autobed'''
    def __init__(self):
        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.elapsed_time = []
        self.database_path = '/media/henryclever/Seagate Backup Plus Drive/Autobed_OFFICIAL_Trials'
        self.head_center_2d = None
        self.zoom_factor = 2
        self.mat_sampled = False
        self.mat_pose = []
        self.head_pose = []
        self.zoom_factor = 2


        self.bedangle = 0.
        self.params_length = np.zeros((8))  # torso height, torso vert, shoulder right, shoulder left, upper arm right, upper arm left, forearm right, forearm left

    print "Ready to start reading bags."

    def read_bag(self, filepath):

        filename = filepath+'/henryc_on_bed_05102019.bag'


        self.mat_sampled = False

        bag = rosbag.Bag(filename, 'r')
        count = 0


        targets = np.zeros((10,3))
        bed_pos = np.zeros((1,3))
        p_mat = []


        self.final_dataset = {}

        self.final_dataset['images'] = []
        self.final_dataset['bed_angle_deg'] = []

        #don't forget to clear out  the caches of all the labels when you log
        for topic, msg, t in bag.read_messages():
            if topic == '/fsascan':
                self.mat_sampled = True
                p_mat = msg.data
                count += 1





            if self.mat_sampled == True:
                #print self.params_length, 'length'
                if len(p_mat) == 1728:
                    p_mat = np.fliplr(np.flipud(np.array(p_mat).reshape(64, 27)))

                    self.final_dataset['images'].append(list(p_mat.flatten()))
                    self.final_dataset['bed_angle_deg'].append(0)


                    self.mat_sampled = False
                    p_mat = []
                    bed_pos = np.zeros((1,3))
                    print 'keep this target'


                sleep(0.001)

                    #print self.mat_tar_pos[len(mat_tar_pos)-1][2], 'accelerometer reading', len(mat_tar_pos)
               
        bag.close()


        print count, len(self.final_dataset['images'])
        return self.final_dataset


    def read_numpy(self, filepath):

        self.final_dataset = {}

        self.final_dataset['images'] = []
        self.final_dataset['bed_angle_deg'] = []


        for i in range(19, 22):

            file_number = '{0:04}'.format(i)
            newpath = filepath +'data_082018_'+ str(file_number)
            print "loading existing npy files in the new path...."
            pressure_all = np.load(newpath + "/pressure.npy")
            for image in range(pressure_all.shape[0]):
                p_mat = pressure_all[image, :]
                p_mat = np.fliplr(np.flipud(np.array(p_mat).reshape(64, 27)))

                self.final_dataset['images'].append(list(p_mat.flatten()))
                self.final_dataset['bed_angle_deg'].append(0)

            print pressure_all.shape

        print len(self.final_dataset['images'])
        return self.final_dataset

if __name__ == '__main__':
    rospy.init_node('bag_to_pickle')
    bagtopkl = BagfileToPickle()


    #print file_details_dict['9']
    database_path = '/home/henry/data/unlabeled_pmat_data'
    database_path2 = '/media/henry/multimodal_data_1/test_data_09102018_to_09112018/'


    #subject_detaildata = bagtopkl.read_bag(database_path)
    subject_detaildata = bagtopkl.read_numpy(database_path2)
    #pkl.dump(database_path, open(database_path+detail[2],'.p', "wb"))

    #do this when you want to overwrite the current files

    #pkl.dump(subject_detaildata,open(os.path.join(database_path+'/henryc_on_bed_05102019.p'), 'wb'))

    pkl.dump(subject_detaildata, open(os.path.join(database_path+'/henrye_on_bed_09102019.p'), 'wb'))


