import numpy as np
import random
import copy
import lib_pyrender_ez as libRender
import lib_pyrender_ez as libPyRender
from opendr.renderer import ColoredRenderer
from opendr.renderer import DepthRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model
from cv_bridge import CvBridge, CvBridgeError
from util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose

import cPickle as pkl

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/henry/git/volumetric_pose_gen/convnets')


#volumetric pose gen libraries
import lib_visualization as libVisualization
import lib_kinematics as libKinematics
from multipose_lib import ArTagLib
from multipose_lib import VizLib
from process_yash_data import ProcessYashData
from preprocessing_lib import PreprocessingLib
from tensorprep_lib import TensorPrepLib
from time import sleep
import rospy
import roslib
from sensor_msgs.msg import PointCloud2
from hrl_msgs.msg import FloatArrayBare
from ar_track_alvar_msgs.msg import AlvarMarkers
import sensor_msgs.point_cloud2
from scipy.stats import mode
import os.path as osp
import imutils


import matplotlib.cm as cm #use cm.jet(list)

from scipy.ndimage.filters import gaussian_filter
#ROS
#import rospy
#import tf
DATASET_CREATE_TYPE = 1

import cv2
from camera import Camera

import math
from random import shuffle
import torch
import torch.nn as nn

import tensorflow as tensorflow
import cPickle as pickle
VERT_CUT, HORIZ_CUT = 0, 50
pre_VERT_CUT = 40


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL

SHOW_SMPL_EST = True
#PARTICIPANT = "S196"#"S151"
POSE_TYPE = "1"
MAT_SIZE = (64, 27)


PC_WRT_ARTAG_ADJ = [0.11, -0.02, 0.07]
ARTAG_WRT_PMAT = [0.08, 0.05, 0.0]

DROPOUT = False
CAM_BED_DIST = 1.66


#import sys

#sys.path.insert(0, '/home/henry/git/volumetric_pose_gen/convnets')


class Viz3DPose():
    def __init__(self, participant_directory):

        ##load participant info
        participant_info = load_pickle(participant_directory+"/participant_info.p")
        print "participant directory: ", participant_directory
        for entry in participant_info:
            print entry, participant_info[entry]

        self.gender = participant_info['gender']
        self.height_in = participant_info['height_in']
        self.weight_lbs = participant_info['weight_lbs']
        self.adj_2 = participant_info['adj_2']
        self.pose_type_2_list = participant_info['pose_type']

        #participant_directory2 = "/media/henry/multimodal_data_2/CVPR2020_study/S187"
        #participant_info2 = load_pickle(participant_directory2+"/participant_info.p")
        self.calibration_optim_values = participant_info['cal_func']
        #self.calibration_optim_values = [-0.171537,   -4.05880298, -1.51663182,  0.08712198,  0.03664871,  0.09108604,  0.67524232]

        self.tf_corners = participant_info['corners']



        self.index_queue = []
        self.reset_pose = False

        self.marker0, self.marker1, self.marker2, self.marker3 = None, None, None, None
        self.pressure = None
        self.markers = [self.marker0, self.marker1, self.marker2, self.marker3]


        self.point_cloud_array = np.array([[0., 0., 0.]])
        self.pc_isnew = False


        #self.Render = libRender.pyRenderMesh(render = False)
        self.pyRender = libPyRender.pyRenderMesh(render = True)

        self.count = 0



        self.bridge = CvBridge()
        self.color, self.depth_r, self.pressure = 0, 0, 0

        self.kinect_im_size = (960, 540)
        self.pressure_im_size = (64, 27)
        self.pressure_im_size_required = (64, 27)

        # initialization of kinect and thermal cam calibrations from YAML files
        dist_model = 'rational_polynomial'
        self.kcam = Camera('kinect', self.kinect_im_size, dist_model)
        self.kcam.init_from_yaml(osp.expanduser('~/catkin_ws/src/multimodal_pose/calibrations/kinect.yaml'))

        # we are at qhd not hd so need to cut the focal lengths and centers in half
        self.kcam.K[0:2, 0:3] = self.kcam.K[0:2, 0:3] / 2

       # print self.kcam.K

        self.new_K_kin, roi = cv2.getOptimalNewCameraMatrix(self.kcam.K, self.kcam.D, self.kinect_im_size, 1,
                                                            self.kinect_im_size)

        #print self.new_K_kin

        self.drawing = False  # true if mouse is pressed
        self.mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
        self.ix, self.iy = -1, -1
        self.label_index = 0
        self.coords_from_top_left = [0, 0]
        self.overall_image_scale_amount = 0.85
        self.depthcam_midpixel = [0, 0]
        self.depthcam_midpixel2 = [0, 0]
        self.select_new_calib_corners = {}
        self.select_new_calib_corners["lay"] = True
        self.select_new_calib_corners["sit"] = True
        self.calib_corners = {}
        self.calib_corners["lay"] = 8 * [[0, 0]]
        self.calib_corners["sit"] = 8 * [[0, 0]]

        self.final_dataset = {}




    def load_next_file(self, newpath):

        print "loading existing npy files in the new path...."
        time_orig = time.time()
        self.color_all = np.load(newpath+"/color.npy")
        self.depth_r_all = np.load(newpath+"/depth_r.npy")
        self.pressure_all = np.load(newpath+"/pressure.npy")
        self.bedstate_all = np.load(newpath+"/bedstate.npy")
        self.markers_all = np.load(newpath+"/markers.npy", allow_pickle=True)
        self.time_stamp_all = np.load(newpath+"/time_stamp.npy")
        self.point_cloud_autofil_all = np.load(newpath+"/point_cloud.npy")
        #self.config_code_all = np.load(newpath+"/config_code.npy")
        print "Finished. Time taken: ", time.time() - time_orig



    def transform_selected_points(self, image_in, camera_alpha_vert, camera_alpha_horiz, angle, right, up, h_scale_cut, v_scale_cut, coords_subset):
        image = np.copy(image_in)

        h_scale = h_scale_cut[0]
        h_cut = h_scale_cut[1]
        v_scale = v_scale_cut[0]
        v_cut = v_scale_cut[1]
        tf_coords_subset = np.copy(coords_subset)
        #print camera_alpha_vert, camera_alpha_horiz, HORIZ_CUT, VERT_CUT, pre_VERT_CUT, right

        h = VizLib().get_new_K_kin_homography(camera_alpha_vert, camera_alpha_horiz, self.new_K_kin, flip_vert=-1)


        for i in range(4):

            new_coords = np.matmul(h, np.array([tf_coords_subset[i, 1]+pre_VERT_CUT, tf_coords_subset[i, 0]+HORIZ_CUT, 1]))
            new_coords = new_coords/new_coords[2]
            tf_coords_subset[i, 0] = new_coords[1] - HORIZ_CUT
            tf_coords_subset[i, 1] = new_coords[0] - pre_VERT_CUT


            tf_coords_subset[i, 1] = (tf_coords_subset[i, 1] - image.shape[0] / 2) * np.cos(np.deg2rad(angle)) - (
                        tf_coords_subset[i, 0] - image.shape[1] / 2) * np.sin(np.deg2rad(angle)) + image.shape[
                                  0] / 2 - up
            tf_coords_subset[i, 0] = (tf_coords_subset[i, 1] - image.shape[0] / 2) * np.sin(np.deg2rad(angle)) + (
                        tf_coords_subset[i, 0] - image.shape[1] / 2) * np.cos(np.deg2rad(angle)) + image.shape[
                                  1] / 2 - right

            tf_coords_subset[i, 0] = h_scale * (tf_coords_subset[i][0] + h_cut) - h_cut
            tf_coords_subset[i, 1] = v_scale * (tf_coords_subset[i][1] + v_cut) - v_cut

            image[int(tf_coords_subset[i][1] + 0.5) - 2:int(tf_coords_subset[i][1] + 0.5) + 2,
            int(tf_coords_subset[i][0] + 0.5) - 2:int(tf_coords_subset[i][0] + 0.5) + 2, :] = 0

            #print int(tf_coords_subset[i][1] + 0.5), int(tf_coords_subset[i][0] + 0.5), 'coords in color'

        #print image.shape
        return tf_coords_subset, image



    def depth_image(self, depth_r_orig):
        # DEPTH'
        depth_r_reshaped = depth_r_orig / 3 - 300


        depth_r_reshaped = np.clip(depth_r_reshaped, 0, 255)
        depth_r_reshaped = depth_r_reshaped.astype(np.uint8)
        depth_r_reshaped = np.stack((depth_r_reshaped,) * 3, -1)

        depth_r_reshaped = np.rot90(depth_r_reshaped)
        depth_r_orig = np.rot90(depth_r_orig)
        return depth_r_reshaped, depth_r_reshaped.shape, depth_r_orig


    def pressure_image(self, pressure_orig, color_size, tf_corners):
        start_low_pt = [int((tf_corners[0, 0] + tf_corners[2, 0] + 1) / 2) + HORIZ_CUT,
                        int((tf_corners[0, 1] + tf_corners[1, 1] + 1) / 2) + pre_VERT_CUT]
        start_high_pt = [int((tf_corners[1, 0] + tf_corners[3, 0] + 1) / 2) + HORIZ_CUT,
                        int((tf_corners[2, 1] + tf_corners[3, 1] + 1) / 2) + pre_VERT_CUT]

        pressure_im_size_required = [ start_high_pt[1] - start_low_pt[1], start_high_pt[0] - start_low_pt[0]]


        # PRESSURE
        pressure_reshaped_temp = np.reshape(pressure_orig, MAT_SIZE)
        pressure_reshaped_temp = np.flipud(np.fliplr(pressure_reshaped_temp))
        pressure_reshaped = cm.jet(1-pressure_reshaped_temp/100)[:, :, 0:3]
        pressure_reshaped = (pressure_reshaped * 255).astype(np.uint8)
        pressure_reshaped = cv2.resize(pressure_reshaped, (pressure_im_size_required[1], pressure_im_size_required[0])).astype(np.uint8)
        pressure_reshaped = np.rot90(pressure_reshaped, 3)
        pressure_reshaped_temp2 = np.zeros((color_size[1], color_size[0], color_size[2])).astype(np.uint8)
        pressure_reshaped_temp2[:, :, 0] = 50

        pmat_reshaped_size = pressure_reshaped.shape

        pressure_reshaped_temp2[start_low_pt[0]:start_low_pt[0]+pmat_reshaped_size[0], \
                                start_low_pt[1]:start_low_pt[1]+pmat_reshaped_size[1], :] = pressure_reshaped

        pressure_reshaped_temp2[start_low_pt[0]-2:start_low_pt[0]+2,start_low_pt[1]-2:start_low_pt[1]+2,: ] = 255

        pressure_reshaped = pressure_reshaped_temp2

        pressure_reshaped = np.rot90(pressure_reshaped)
        return pressure_reshaped, pressure_reshaped.shape



    def trim_pc_sides(self, camera_alpha_vert, camera_alpha_horiz, h, kinect_rot_cw):

        f_x, f_y, c_x, c_y = self.new_K_kin[0, 0], self.new_K_kin[1, 1], self.new_K_kin[0, 2], self.new_K_kin[1, 2]
        #for i in range(3):
        #    print np.min(self.point_cloud_autofil[:, i]), np.max(self.point_cloud_autofil[:, i])


        self.point_cloud_autofil[:, 0] = self.point_cloud_autofil[:, 0]# - 0.17 - 0.036608



        #CALIBRATE THE POINT CLOUD HERE

        pc_autofil_red = np.copy(self.point_cloud_autofil)

        if pc_autofil_red.shape[0] == 0:
            pc_autofil_red = np.array([[0.0, 0.0, 0.0]])

        #warp it by the homography i.e. rotate a bit
        pc_autofil_red -=[0.0, 0.0, CAM_BED_DIST]

        theta_1 = np.arctan((camera_alpha_vert-1)*CAM_BED_DIST/(270*CAM_BED_DIST/f_y))/2 #short side
        short_side_rot = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta_1), -np.sin(theta_1)], [0.0, np.sin(theta_1), np.cos(theta_1)]])
        pc_autofil_red = np.matmul(pc_autofil_red, short_side_rot)#[0:3, :]

        theta_2 = np.arctan((1-camera_alpha_horiz)*CAM_BED_DIST/(270*CAM_BED_DIST/f_x))/2 #long side
        long_side_rot = np.array([[np.cos(theta_2), 0.0, np.sin(theta_2)], [0.0, 1.0, 0.0], [-np.sin(theta_2), 0.0, np.cos(theta_2)]])
        pc_autofil_red = np.matmul(pc_autofil_red, long_side_rot)#[0:3, :]

        pc_autofil_red +=[0.0, 0.0, CAM_BED_DIST]


        #add the warping translation
        X_single1 = h[0, 2] * CAM_BED_DIST / f_y
        Y_single1 = h[1, 2] * CAM_BED_DIST / f_x

        #print X_single1, Y_single1
        pc_autofil_red += [-Y_single1/2, -X_single1/2, 0.0]


        #rotate normal to the bed
        angle = kinect_rot_cw*np.pi/180.
        z_rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0.0, 0.0, 1.0]])
        pc_autofil_red = np.matmul(pc_autofil_red, z_rot_mat)#[0:3, :]


        #translate by the picture shift amount in the x and y directions


        #print np.min(pc_autofil_red[:, 0]), np.max(pc_autofil_red[:, 0]), "Y min max"
        #print self.tf_corners[2], self.depthcam_midpixel2

        #translate from the 0,0 being the camera to 0,0 being the left corner of the bed measured by the clicked point
        zero_location = np.copy(self.tf_corners[2]) #TF corner needs to be manipulated!
        x_coord_from_camcenter_single = zero_location[0] - self.depthcam_midpixel2[0]
        y_coord_from_camcenter_single = zero_location[1] - self.depthcam_midpixel2[1]
        X_single2 = (x_coord_from_camcenter_single) * CAM_BED_DIST / f_y #shift dim
        Y_single2 = (y_coord_from_camcenter_single) * CAM_BED_DIST / f_x #long dim
        pc_autofil_red += [Y_single2, -X_single2, -CAM_BED_DIST]


        #adjust to fit to the lower left corner step 2
        pc_autofil_red += [self.adj_2[0], self.adj_2[1], 0.0]

        #cut off everything that's not overlying the bed.
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 1] > 0.0, :]
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 1] < 0.0286 * 27, :]

        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 0] > 0.0, :] #up and down bed
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 0] < 0.0286 * 64 * 1.04, :] #up and down bed

        return pc_autofil_red



    def draw_rectangles(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y



        elif event == cv2.EVENT_MOUSEMOVE:
            #print "mouse moved"

            #if self.drawing == True:
            #    for i in [-3, -2, -1, 0, 1, 2, 3]:
            #        cv2.rectangle(self.blurred_color, (self.ix, self.iy), (x, y), (0, 255, 0), 2)

                #cv2.imshow('all_images', self.blurred_color_clone)

            #for i in [-3, -2, -1, 0, 1, 2, 3]:
            #    cv2.arrowedLine(self.blurred_color, (x + 20, y + 20), (x, y),  (0, 0, 255), 2)

            #self.blurred_color[0:70, 400:, :] = 255

            x_coord_from_bed_corner = x-self.coords_from_top_left[0]
            self.x_coord_from_corner = int(x)*1


            y_coord_from_bed_corner = -(y-self.coords_from_top_left[1])
            self.y_coord_from_corner = int(y)*1
            x_coord_from_bed_corner/=self.overall_image_scale_amount
            self.x_coord_from_corner/=self.overall_image_scale_amount
            y_coord_from_bed_corner/=self.overall_image_scale_amount
            self.y_coord_from_corner/=self.overall_image_scale_amount


        elif event == cv2.EVENT_LBUTTONUP:
            if y < 70 and x > 400:
                self.has_blurred_face = True
            else:

                self.drawing = False
                #cv2.rectangle(self.blurred_color, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                #self.blurred_color_clone = self.blurred_color.copy()

                self.label_single_image.append([self.ix, self.iy, x, y])

                self.label_index += 1

                print x, y


                blur_size = 6
                x_sq_list = [x-blur_size*2, x-blur_size, x, x+blur_size, x+blur_size*2]
                y_sq_list = [y-blur_size*2, y-blur_size, y, y+blur_size, y+blur_size*2]

                for x_sq_idx in range(len(x_sq_list)):
                    for y_sq_idx in range(len(y_sq_list)):
                        x_sq = x_sq_list[x_sq_idx]
                        y_sq = y_sq_list[y_sq_idx]

                        b_x_read = int(x_sq/blur_size) * blur_size
                        b_y_read = int(y_sq/blur_size) * blur_size

                        b_x_write = int(x_sq/blur_size) * blur_size
                        b_y_write = int(y_sq/blur_size) * blur_size

                        blur_block = self.blurred_color[b_y_read-blur_size/2:b_y_read+blur_size/2, b_x_read-blur_size/2:b_x_read+blur_size/2, :]
                        blur_mean = np.mean(np.mean(blur_block, axis = 0), axis = 0).astype(np.int16)
                        #print blur_block, blur_mean

                        blur_mean += np.random.randint(-50, 50)
                        blur_mean = np.clip(blur_mean, 0, 255).astype(np.uint8)

                        blur_block = blur_block*0 + blur_mean
                        #print blur_block

                        self.blurred_color[b_y_write - blur_size / 2:b_y_write + blur_size / 2, \
                                           b_x_write - blur_size / 2:b_x_write + blur_size / 2, :] = blur_block



    def blur_face(self, color_reshaped):
        self.blurred_color = np.copy(color_reshaped)

        #cv2.imshow('all_images', self.blurred_color)
        #k = cv2.waitKey(1)
        # cv2.waitKey(0)

        cv2.namedWindow('all_images')
        cv2.setMouseCallback('all_images', self.draw_rectangles)

        self.has_blurred_face = False

        while self.has_blurred_face == False:

            cv2.imshow('all_images', self.blurred_color)
            k = cv2.waitKey(1) & 0xFF

        color_reshaped = np.copy(self.blurred_color)
        return color_reshaped



    def evaluate_data(self):

        self.depthcam_midpixel2 = [self.new_K_kin[1, 2] - HORIZ_CUT, (960 - self.new_K_kin[0, 2]) - pre_VERT_CUT]



        #function_input = np.array(function_input)*np.array([10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 1])
        #function_input += np.array([2.2, 32, -1, 1.2, 32, -5, 1.0, 1.0, 0.96, 0.95, 0.8])
        function_input = np.array(self.calibration_optim_values)*np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.1])
        function_input += np.array([1.2, 32, -5, 1.0, 1.0, 0.96, 0.95])


        kinect_rotate_angle = function_input[3-3]
        kinect_shift_up = int(function_input[4-3])
        kinect_shift_right = int(function_input[5-3])
        camera_alpha_vert = function_input[6-3]
        camera_alpha_horiz = function_input[7-3]

        blah = True



        file_dir = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-"+POSE_TYPE
        file_dirs = [file_dir]

        init_time = time.time()

        RESAVE_DICT = {}
        RESAVE_DICT['images'] = []
        RESAVE_DICT['RGB'] = []
        RESAVE_DICT['depth'] = []
        RESAVE_DICT['pc'] = []
        RESAVE_DICT['pmat_corners'] = []
        RESAVE_DICT['pose_type'] = []

        for file_dir in file_dirs:
            V3D.load_next_file(file_dir)

            start_num = 0
            #print self.color_all.shape

            #for im_num in range(29, 100):
            for im_num in range(start_num, self.color_all.shape[0]):

                #For P188: skip 5. 13 good cross legs


                if PARTICIPANT == "S114" and POSE_TYPE == "2"  and im_num in [26, 29]: continue #these don't have point clouds
                if PARTICIPANT == "S165" and POSE_TYPE == "2" and im_num in [1, 3, 15]: continue #these don't have point clouds
                if PARTICIPANT == "S188" and POSE_TYPE == "2"  and im_num in [5, 17, 21]: continue #these don't have point clouds

                print "NEXT IM!", im_num, " ", time.time() - init_time, self.pose_type_2_list[im_num]

                if POSE_TYPE == "2":
                    RESAVE_DICT['pose_type'].append(self.pose_type_2_list[im_num])
                elif POSE_TYPE == "1":
                    if im_num == 0:
                        if PARTICIPANT == "S145":
                            RESAVE_DICT['pose_type'].append('p_sel_sup')
                        elif PARTICIPANT == "S188":
                            RESAVE_DICT['pose_type'].append('p_sel_ll')
                        else:
                            RESAVE_DICT['pose_type'].append('p_sel_any')
                    if im_num == 1:
                        if PARTICIPANT == "S140" or PARTICIPANT == "S145":
                            RESAVE_DICT['pose_type'].append('p_sel_ll')
                        elif PARTICIPANT == "S188":
                            RESAVE_DICT['pose_type'].append('p_sel_rl')
                        else:
                            RESAVE_DICT['pose_type'].append('p_sel_sup')
                    if im_num == 2:
                        if PARTICIPANT == "S140" or PARTICIPANT == "S145":
                            RESAVE_DICT['pose_type'].append('p_sel_rl')
                        elif PARTICIPANT == "S188":
                            RESAVE_DICT['pose_type'].append('p_sel_prn')
                        else:
                            RESAVE_DICT['pose_type'].append('p_sel_ll')
                    if im_num == 3:
                        if PARTICIPANT == "S140" or PARTICIPANT == "S145":
                            RESAVE_DICT['pose_type'].append('p_sel_prn')
                        elif PARTICIPANT == "S188":
                            RESAVE_DICT['pose_type'].append('p_sel_any')
                        else:
                            RESAVE_DICT['pose_type'].append('p_sel_rl')
                    if im_num == 4:
                        if PARTICIPANT == "S140" or PARTICIPANT == "S188":
                            RESAVE_DICT['pose_type'].append('p_sel_sup')
                        else:
                            RESAVE_DICT['pose_type'].append('p_sel_prn')

                print RESAVE_DICT['pose_type'][-1]

                self.overall_image_scale_amount = 0.85

                half_w_half_l = [0.4, 0.4, 1.1, 1.1]

                all_image_list = []
                self.label_single_image = []

                self.label_index = 0

                self.color = self.color_all[im_num]
                self.depth_r = self.depth_r_all[im_num]
                self.pressure = self.pressure_all[im_num]
                self.bed_state = self.bedstate_all[im_num]

                if self.point_cloud_autofil_all[im_num].shape[0] == 0:
                    self.point_cloud_autofil_all[im_num] = np.array([[0.0, 0.0, 0.0]])
                self.point_cloud_autofil = self.point_cloud_autofil_all[im_num] + self.markers_all[im_num][2]#[0.0, 0.0, 0.0]#0.1]
                #print self.markers_all[im_num]
                #print self.point_cloud_autofil.shape, 'PC AUTOFIL ORIG'



                self.bed_state[0] = self.bed_state[0]*0.0#*head_angle_multiplier
                self.bed_state *= 0
                #self.bed_state += 60.
                #print self.bed_state, np.shape(self.pressure)

                if im_num == start_num and blah == True:
                    markers_c = []
                    markers_c.append(self.markers_all[im_num][0])
                    markers_c.append(self.markers_all[im_num][1])
                    markers_c.append(self.markers_all[im_num][2])
                    markers_c.append(self.markers_all[im_num][3])
                    #for idx in range(4):
                        #if markers_c[idx] is not None:
                            #markers_c[idx] = np.array(markers_c[idx])*213./228.
                blah = False

                #print markers_c, 'Markers C'

                # Get the marker points in 2D on the color image
                u_c, v_c = ArTagLib().color_2D_markers(markers_c, self.new_K_kin)

                # Get the marker points dropped to the height of the pressure mat
                u_c_drop, v_c_drop, markers_c_drop = ArTagLib().color_2D_markers_drop(markers_c, self.new_K_kin)

                #print markers_c_drop, self.new_K_kin, self.pressure_im_size_required, self.bed_state, half_w_half_l
                # Get the geometry for sizing the pressure mat
                pmat_ArTagLib = ArTagLib()
                self.pressure_im_size_required, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend, half_w_half_l = \
                    pmat_ArTagLib.p_mat_geom(markers_c_drop, self.new_K_kin, self.pressure_im_size_required, self.bed_state, half_w_half_l)

                tf_corners = np.zeros((8, 2))
                tf_corners[0:8,:] = np.copy(self.tf_corners)



                #COLOR
                color_reshaped, color_size = VizLib().color_image(self.color, self.kcam, self.new_K_kin,
                                                                  u_c, v_c, u_c_drop, v_c_drop, u_c_pmat, v_c_pmat, camera_alpha_vert, camera_alpha_horiz)
                color_reshaped = imutils.rotate(color_reshaped, kinect_rotate_angle)
                color_reshaped = color_reshaped[pre_VERT_CUT+kinect_shift_up:-pre_VERT_CUT+kinect_shift_up,  HORIZ_CUT+kinect_shift_right : 540 - HORIZ_CUT+kinect_shift_right, :]

                #all_image_list.append(color_reshaped)

                tf_corners[0:4, :], _ = self.transform_selected_points(color_reshaped,
                                                                     camera_alpha_vert,
                                                                     camera_alpha_horiz,
                                                                     kinect_rotate_angle,
                                                                     kinect_shift_right,
                                                                     kinect_shift_up, [1.0, 0],
                                                                     [1.0, 0],
                                                                     np.copy(self.tf_corners[0:4][:]))

                #should blur face here
                #color_reshaped = self.blur_face(color_reshaped)
                RESAVE_DICT['RGB'].append(color_reshaped)
                RESAVE_DICT['pmat_corners'].append(tf_corners[0:4, :])

                #SAVE CALIBRATED COLOR HERE, color_reshaped
                #SAVE CALIBRATED TF CORNERS HERE, tf_corners[0:4, :]
                all_image_list.append(color_reshaped)



                #DEPTH
                h = VizLib().get_new_K_kin_homography(camera_alpha_vert, camera_alpha_horiz, self.new_K_kin)
                depth_r_orig = cv2.warpPerspective(self.depth_r, h, (self.depth_r.shape[1], self.depth_r.shape[0]))
                depth_r_orig = imutils.rotate(depth_r_orig, kinect_rotate_angle)
                depth_r_orig = depth_r_orig[HORIZ_CUT + kinect_shift_right: 540 - HORIZ_CUT + kinect_shift_right, pre_VERT_CUT - kinect_shift_up:-pre_VERT_CUT - kinect_shift_up]
                #SAVE CALIBRATED DEPTH HERE, depth_r_orig
                RESAVE_DICT['depth'].append(depth_r_orig)
                depth_r_reshaped, depth_r_size, depth_r_orig = self.depth_image(depth_r_orig)
                all_image_list.append(depth_r_reshaped)



                #PRESSURE
                self.pressure = np.clip(self.pressure*1, 0, 100)
                pressure_reshaped, pressure_size = self.pressure_image(self.pressure, color_size, tf_corners)

                pressure_reshaped = pressure_reshaped[pre_VERT_CUT:-pre_VERT_CUT,  HORIZ_CUT : 540 - HORIZ_CUT, :]
                all_image_list.append(pressure_reshaped)






                self.all_images = np.zeros((960-np.abs(pre_VERT_CUT)*2, 1, 3)).astype(np.uint8)
                for image in all_image_list:
                    #print image.shape
                    self.all_images = np.concatenate((self.all_images, image), axis = 1)

                self.all_images = self.all_images[VERT_CUT : 960 - VERT_CUT, :, :]



                is_not_mult_4 = True
                while is_not_mult_4 == True:
                    is_not_mult_4 = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount).shape[1]%4
                    self.overall_image_scale_amount+= 0.001


                self.all_images = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount)
                self.cursor_shift = self.all_images.shape[1]/4


                self.all_images_clone = self.all_images.copy()


                cv2.imshow('all_images', self.all_images)
                k = cv2.waitKey(1)
                #cv2.waitKey(0)



                #now do 3D rendering
                pmat = np.fliplr(np.flipud(np.clip(self.pressure.reshape(MAT_SIZE)*float(1), a_min=0, a_max=100)))

                #SAVE PRESSURE HERE, self.pressure
                RESAVE_DICT['images'].append(pmat)

                pc_autofil_red = self.trim_pc_sides(camera_alpha_vert, camera_alpha_horiz, h, kinect_rotate_angle) #this is the point cloud

                #SAVE POINT CLOUD HERE, pc_autofil_red
                RESAVE_DICT['pc'].append(pc_autofil_red)

                camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST] #[dist from foot of bed, dist from left side of mat, dist normal]

                #
                self.pyRender.render_3D_data(camera_point, pmat = pmat, pc = pc_autofil_red)

                self.point_cloud_array = None

                if POSE_TYPE == "2":
                    save_name = '/prescribed'
                elif POSE_TYPE == "1":
                    save_name = '/p_select'

            pkl.dump(RESAVE_DICT,open(participant_directory+save_name+'.p', 'wb'))
            print "SAVED."
                #sleep(3)


if __name__ ==  "__main__":


    participant_list = [#"S103",
                        #"S104",
                        #"S107",
                        #"S114",
                        #"S118",
                        #"S121",
                        #"S130",
                        #"S134",
                        #"S140",
                        #"S141",

                        #"S145",
                        #"S151",
                        #"S163",
                        #"S165",
                        #"S170",
                        #"S179",
                        #"S184",
                        #"S187",
                        #"S188",
                        "S196",
                        ]

    for PARTICIPANT in participant_list:


        participant_directory = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT
        #participant_directory = "/home/henry/Desktop/CVPR2020_study/"+PARTICIPANT

        V3D = Viz3DPose(participant_directory)

        F_eval = V3D.evaluate_data()



