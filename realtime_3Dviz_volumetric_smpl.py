import numpy as np
import random
import copy
from opendr.renderer import ColoredRenderer
from opendr.renderer import DepthRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model
from cv_bridge import CvBridge, CvBridgeError
from util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose

#volumetric pose gen libraries
import lib_visualization as libVisualization
import lib_kinematics as libKinematics
import lib_render as libRender
from process_yash_data import ProcessYashData
from preprocessing_lib import PreprocessingLib
import dart_skel_sim
from time import sleep
import rospy
import roslib
from sensor_msgs.msg import PointCloud2
from hrl_msgs.msg import FloatArrayBare
from ar_track_alvar_msgs.msg import AlvarMarkers
import sensor_msgs.point_cloud2
import ros_numpy
from scipy.stats import mode

from scipy.ndimage.filters import gaussian_filter
#ROS
#import rospy
#import tf
DATASET_CREATE_TYPE = 1

import cv2

import math
from random import shuffle
import torch
import torch.nn as nn

import tensorflow as tensorflow
import cPickle as pickle


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL



class GeneratePose():
    def __init__(self, gender, filepath_prefix = '/home/henry'):
        ## Load SMPL model
        self.filepath_prefix = filepath_prefix

        self.index_queue = []
        if gender == "m":
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.reset_pose = False
        self.m = load_model(model_path)

        self.marker0, self.marker1, self.marker2, self.marker3 = None, None, None, None
        self.pressure = None
        self.markers = [self.marker0, self.marker1, self.marker2, self.marker3]
        rospy.Subscriber("/multi_pose/ar_pose_marker", AlvarMarkers, self.callback_bed_tags)
        rospy.Subscriber("/multi_pose/kinect2/qhd/points", PointCloud2, self.callback_points)
        rospy.Subscriber("/multi_pose/fsascan", FloatArrayBare, self.callback_pressure)

        rospy.Subscriber("/abdout0", FloatArrayBare, self.bed_config_callback)
        #rospy.Subscriber("/abdout0", FloatArrayBare, self.callback_bed_state)
        print "init subscriber"

        rospy.init_node('vol_3d_listener', anonymous=False)
        self.point_cloud_array = np.array([[0., 0., 0.]])
        self.pc_isnew = False

    def bed_config_callback(self, msg):
        '''This callback accepts incoming pressure map from
        the Vista Medical Pressure Mat and sends it out.
        Remember, this array needs to be binarized to be used'''
        bedangle = np.round(msg.data[0], 0)

        # this little statement tries to filter the angle data. Some of the angle data is messed up, so we make a queue and take the mode.
        if self.index_queue == []:
            self.index_queue = np.zeros(5)
            if bedangle > 350:
                self.index_queue = self.index_queue + math.ceil(bedangle) - 360
            else:
                self.index_queue = self.index_queue + math.ceil(bedangle)
            bedangle = mode(self.index_queue)[0][0]
        else:
            self.index_queue[1:5] = self.index_queue[0:4]
            if bedangle > 350:
                self.index_queue[0] = math.ceil(bedangle) - 360
            else:
                self.index_queue[0] = math.ceil(bedangle)
            bedangle = mode(self.index_queue)[0][0]

        if bedangle > 180: bedangle = bedangle - 360

        self.bedangle = bedangle


    def callback_bed_tags(self, data):
        self.marker0, self.marker1, self.marker2, self.marker3 = None, None, None, None
        for marker in data.markers:

            if marker.id == 0:
                self.marker0 = np.array([marker.pose.pose.position.x,
                                         marker.pose.pose.position.y,
                                         marker.pose.pose.position.z])*213./228.
            if marker.id == 1:
                self.marker1 = np.array([marker.pose.pose.position.x,
                                         marker.pose.pose.position.y,
                                         marker.pose.pose.position.z])*213./228.
            if marker.id == 2:
                self.marker2 = np.array([marker.pose.pose.position.x,
                                         marker.pose.pose.position.y,
                                         marker.pose.pose.position.z])*213./228.
            if marker.id == 3:
                self.marker3 = np.array([marker.pose.pose.position.x,
                                         marker.pose.pose.position.y,
                                         marker.pose.pose.position.z])*213./228.


        self.markers = [self.marker0, self.marker1, self.marker2, self.marker3]



    def callback_points(self, data):

        point_cloud_array = []
        last_time = time.time()
        for point in sensor_msgs.point_cloud2.read_points(data, skip_nans = True):
            point_cloud_array.append(point[0:3])
        self.point_cloud_array = np.array(point_cloud_array)

        try:
            self.point_cloud_array -= (self.marker2 - np.array([0.0, 0.0, 0.0]))
        except:
            self.point_cloud_array = np.array([[0., 0., 0.]])
        self.pc_isnew = True

        print "Time to convert point cloud is: ", time.time() - last_time




    def callback_pressure(self, data):
        if len(data.data) > 1:
            self.pressure = np.array(data.data)


    def generate_prechecked_pose(self, gender, posture, stiffness, filename):


        prechecked_pose_list = np.load(filename, allow_pickle = True).tolist()



        print len(prechecked_pose_list)
        shuffle(prechecked_pose_list)

        pyRender = libRender.pyRenderMesh()

        for shape_pose_vol in prechecked_pose_list[6:]:
            #print shape_pose_vol
            #print shape_pose_vol[0]
            #print shape_pose_vol[1]
            #print shape_pose_vol[2]
            for idx in range(len(shape_pose_vol[0])):
                #print shape_pose_vol[0][idx]
                self.m.betas[idx] = shape_pose_vol[0][idx]

            print 'init'
            print shape_pose_vol[2][0],shape_pose_vol[2][1],shape_pose_vol[2][2]

            self.m.pose[:] = np.array(72 * [0.])

            for idx in range(len(shape_pose_vol[1])):
                #print shape_pose_vol[1][idx]
                #print self.m.pose[shape_pose_vol[1][idx]]
                #print shape_pose_vol[2][idx]
                pose_index = shape_pose_vol[1][idx]*1

                self.m.pose[pose_index] = shape_pose_vol[2][idx]*1.

                #print idx, pose_index, self.m.pose[pose_index], shape_pose_vol[2][idx]

            print self.m.pose[0:3]
            init_root = np.array(self.m.pose[0:3])+0.000001
            init_rootR = libKinematics.matrix_from_dir_cos_angles(init_root)
            root_rot = libKinematics.eulerAnglesToRotationMatrix([np.pi, 0.0, np.pi/2])
            #print root_rot
            trans_root = libKinematics.dir_cos_angles_from_matrix(np.matmul(root_rot, init_rootR))

            self.m.pose[0] = trans_root[0]
            self.m.pose[1] = trans_root[1]
            self.m.pose[2] = trans_root[2]

            print root_rot
            print init_rootR
            print trans_root
            print init_root, trans_root




            #print self.m.J_transformed[1, :], self.m.J_transformed[4, :]
            # self.m.pose[51] = selection_r
            pyRender.mesh_render_pose_bed(self.m, self.point_cloud_array, self.pc_isnew, self.pressure, self.markers)
            self.point_cloud_array = None

            #dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender = gender, posture = posture, stiffness = stiffness, shiftSIDE = shape_pose_vol[4], shiftUD = shape_pose_vol[5], filepath_prefix=self.filepath_prefix, add_floor = False)

            #dss.run_simulation(10000)
            #generator.standard_render()


            #break

    def estimate_real_time(self, gender, filename):






        pyRender = libRender.pyRenderMesh()
        mat_size = (64, 27)

        if torch.cuda.is_available():
            # Use for GPU
            GPU = True
            dtype = torch.cuda.FloatTensor
            print '######################### CUDA is available! #############################'
        else:
            # Use for CPU
            GPU = False
            dtype = torch.FloatTensor
            print '############################## USING CPU #################################'

        import convnet as convnet
        from torch.autograd import Variable

        if GPU == True:
            model = torch.load(filename)
            model = model.cuda()
        else:
            model = torch.load(filename, map_location='cpu')

        while not rospy.is_shutdown():


            pmat = np.fliplr(np.flipud(np.clip(self.pressure.reshape(mat_size)*5.0, a_min=0, a_max=100)))

            pmat = gaussian_filter(pmat, sigma= 0.5)


            pmat_stack = PreprocessingLib().preprocessing_create_pressure_angle_stack_realtime(pmat, self.bedangle, mat_size)
            pmat_stack = torch.Tensor(pmat_stack)

            images_up_non_tensor = PreprocessingLib().preprocessing_add_image_noise(np.array(PreprocessingLib().preprocessing_pressure_map_upsample(pmat_stack.numpy(), multiple=2)))
            images_up = Variable(torch.Tensor(images_up_non_tensor).type(dtype), requires_grad=False)

            betas_est, root_shift_est, angles_est = model.forward_kinematic_angles_realtime(images_up)

            print betas_est, root_shift_est, angles_est
            angles_est = angles_est.reshape(72)

            for idx in range(10):
                #print shape_pose_vol[0][idx]
                self.m.betas[idx] = betas_est[idx]


            for idx in range(72):
                self.m.pose[idx] = angles_est[idx]


            init_root = np.array(self.m.pose[0:3])+0.000001
            init_rootR = libKinematics.matrix_from_dir_cos_angles(init_root)
            root_rot = libKinematics.eulerAnglesToRotationMatrix([np.pi, 0.0, np.pi/2])
            #print root_rot
            trans_root = libKinematics.dir_cos_angles_from_matrix(np.matmul(root_rot, init_rootR))

            self.m.pose[0] = trans_root[0]
            self.m.pose[1] = trans_root[1]
            self.m.pose[2] = trans_root[2]

            #print self.m.J_transformed[1, :], self.m.J_transformed[4, :]
            # self.m.pose[51] = selection_r
            pyRender.mesh_render_pose_bed(self.m, root_shift_est, self.point_cloud_array, self.pc_isnew, pmat, self.markers)
            self.point_cloud_array = None

            #dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender = gender, posture = posture, stiffness = stiffness, shiftSIDE = shape_pose_vol[4], shiftUD = shape_pose_vol[5], filepath_prefix=self.filepath_prefix, add_floor = False)

            #dss.run_simulation(10000)
            #generator.standard_render()


            #break

if __name__ == "__main__":

    gender = "f"
    filepath_prefix = "/home/henry"


    generator = GeneratePose(gender, filepath_prefix)
    generator.estimate_real_time(gender, filepath_prefix+"/data/synth/convnet_anglesEU_synthreal_s4_3xreal_4xsize_128b_200e.pt")
