import numpy as np
import random
import copy
import lib_render as libRender
import lib_pyrender as libPyRender
from opendr.renderer import ColoredRenderer
from opendr.renderer import DepthRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model
from cv_bridge import CvBridge, CvBridgeError
from util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose


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
PARTICIPANT = "S140"
MAT_SIZE = (64, 27)

PC_WRT_ARTAG_ADJ = [0.11, -0.02, 0.07]
ARTAG_WRT_PMAT = [0.08, 0.05, 0.0]

DROPOUT = False


import sys

sys.path.insert(0, '/home/henry/git/volumetric_pose_gen/convnets')
from unpack_batch_lib import UnpackBatchLib
import convnet as convnet
from torch.autograd import Variable

if False:#torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    print '######################### CUDA is available! #############################'
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    print '############################## USING CPU #################################'


class Viz3DPose():
    def __init__(self, filepath_prefix, participant_directory):

        ##load participant info
        #if False:
        participant_info = load_pickle("/home/henry/Desktop/CVPR2020_study/P136/participant_info.p")
        print "participant directory: ", participant_directory
        for entry in participant_info:
            print entry, participant_info[entry]

        self.gender = participant_info['gender']
        self.height_in = participant_info['height_in']
        self.weight_lbs = participant_info['weight_lbs']
        self.calibration_optim_values = participant_info['cal_func']
        self.tf_corners = participant_info['corners']
        #except:



        ## Load SMPL model
        self.filepath_prefix = filepath_prefix

        self.index_queue = []
        if self.gender == "m":
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.reset_pose = False
        self.m = load_model(model_path)

        self.marker0, self.marker1, self.marker2, self.marker3 = None, None, None, None
        self.pressure = None
        self.markers = [self.marker0, self.marker1, self.marker2, self.marker3]


        self.point_cloud_array = np.array([[0., 0., 0.]])
        self.pc_isnew = False


        self.CTRL_PNL = {}
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['loss_vector_type'] = 'anglesDC'
        self.CTRL_PNL['verbose'] = False
        self.CTRL_PNL['num_epochs'] = 101
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = True
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['num_input_channels'] = 3
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['repeat_real_data_ct'] = 1
        self.CTRL_PNL['regr_angles'] = 1
        self.CTRL_PNL['dropout'] = DROPOUT
        self.CTRL_PNL['depth_map_labels'] = False
        self.CTRL_PNL['depth_map_output'] = True
        self.CTRL_PNL['depth_map_input_est'] = False#rue #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['depth_map_input_est'] #holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True#False
        self.CTRL_PNL['normalize_input'] = True#False
        self.CTRL_PNL['all_tanh_activ'] = True#False
        self.CTRL_PNL['L2_contact'] = True#False
        self.CTRL_PNL['pmat_mult'] = int(5)
        self.CTRL_PNL['cal_noise'] = False
        self.CTRL_PNL['output_only_prev_est'] = False
        self.CTRL_PNL['double_network_size'] = False



        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['incl_pmat_cntct_input'] = False #if there's calibration noise we need to recompute this every batch
            self.CTRL_PNL['clip_sobel'] = False

        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['depth_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL['num_input_channels'] += 3
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.CTRL_PNL['num_input_channels'] += 2
        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['num_input_channels'] += 1

        pmat_std_from_mult = ['N/A', 11.70153502792190, 19.90905848383454, 23.07018866032369, 0.0, 25.50538629767412]
        if self.CTRL_PNL['cal_noise'] == False:
            sobel_std_from_mult = ['N/A', 29.80360490415032, 33.33532963163579, 34.14427844692501, 0.0, 34.86393494050921]
        else:
            sobel_std_from_mult = ['N/A', 45.61635847182483, 77.74920396659292, 88.89398421073700, 0.0, 97.90075708182506]

        self.CTRL_PNL['norm_std_coeffs'] =  [1./41.80684362163343,  #contact
                                             1./16.69545796387731,  #pos est depth
                                             1./45.08513083167194,  #neg est depth
                                             1./43.55800622930469,  #cm est
                                             1./pmat_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat x5
                                             1./sobel_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat sobel
                                             1./1.0,                #bed height mat
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1. / 30.216647403350,  #weight
                                             1. / 14.629298141231]  #height


        self.CTRL_PNL['filepath_prefix'] = '/home/henry/'

        if self.CTRL_PNL['depth_map_output'] == True:  # we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"

        self.TPL = TensorPrepLib()


        self.count = 0


        self.CTRL_PNL['filepath_prefix'] = '/home/henry/'
        self.CTRL_PNL['aws'] = False
        self.CTRL_PNL['lock_root'] = False



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

        print self.kcam.K

        self.new_K_kin, roi = cv2.getOptimalNewCameraMatrix(self.kcam.K, self.kcam.D, self.kinect_im_size, 1,
                                                            self.kinect_im_size)

        print self.new_K_kin

        self.drawing = False  # true if mouse is pressed
        self.mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
        self.ix, self.iy = -1, -1
        self.label_index = 0
        self.coords_from_top_left = [0, 0]
        self.overall_image_scale_amount = 0.85
        self.depthcam_midpixel = [0, 0]
        self.select_new_calib_corners = {}
        self.select_new_calib_corners["lay"] = True
        self.select_new_calib_corners["sit"] = True
        self.calib_corners = {}
        self.calib_corners["lay"] = 8 * [[0, 0]]
        self.calib_corners["sit"] = 8 * [[0, 0]]

        self.final_dataset = {}

        self.filler_taxels = []
        for i in range(28):
            for j in range(65):
                self.filler_taxels.append([i - 1, j - 1, 20000])
        self.filler_taxels = np.array(self.filler_taxels).astype(int)





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
        self.config_code_all = np.load(newpath+"/config_code.npy")
        self.date_stamp_all = np.load(newpath+"/date_stamp.npy")
        print "Finished. Time taken: ", time.time() - time_orig



    def transform_selected_points(self, image, camera_alpha_vert, camera_alpha_horiz, angle, right, up, h_scale_cut, v_scale_cut, coords_subset):
        h_scale = h_scale_cut[0]
        h_cut = h_scale_cut[1]
        v_scale = v_scale_cut[0]
        v_cut = v_scale_cut[1]
        tf_coords_subset = np.copy(coords_subset)
        print camera_alpha_vert, camera_alpha_horiz, HORIZ_CUT, VERT_CUT, pre_VERT_CUT, right

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
            int(tf_coords_subset[i][0] + 0.5) - 2:int(tf_coords_subset[i][0] + 0.5) + 2, :] = 255

        return tf_coords_subset, image

    def rotate_selected_head_points(self, pressure_im_size_required, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend, u_p_bend_calib, v_p_bend_calib):

        low_vert = np.rint(v_c_pmat[2]).astype(np.uint16)
        low_horiz = np.rint(u_c_pmat[1]).astype(np.uint16)
        legs_bend_loc2 = pressure_im_size_required[0]*20/64 + low_horiz

        HEAD_BEND_TAXEL = 41  # measured from the bottom of the pressure mat
        LEGS_BEND2_TAXEL = 20 #measured from the bottom of the pressure mat
        head_bend_loc = pressure_im_size_required[0]*HEAD_BEND_TAXEL/64 + low_horiz

        head_points_L = [np.rint(v_p_bend_calib[0]).astype(np.uint16) - 3 - HORIZ_CUT + 4,
                         380-np.rint(u_p_bend_calib[0] - head_bend_loc - 3).astype(np.uint16) - pre_VERT_CUT + 4]  # np.copy([head_points1[2][0] - decrease_from_orig_len, head_points1[2][1] - increase_across_pmat])
        head_points_R = [np.rint(v_p_bend_calib[1]).astype(np.uint16) + 4 - HORIZ_CUT - 4,
                         380-np.rint(u_p_bend_calib[1] - head_bend_loc - 3).astype(np.uint16) - pre_VERT_CUT + 4]  # np.copy([head_points1[3][0] - decrease_from_orig_len, head_points1[3][1] + increase_across_pmat])
        legs_points_pre = [pressure_im_size_required[0] * 64 / 64 - pressure_im_size_required[0] * (64 - LEGS_BEND2_TAXEL) / 64, low_vert]  # happens at legs bend2


        legs_points_L = [np.rint(v_p_bend[4]).astype(np.uint16) - 3 - HORIZ_CUT + 4,
                         head_bend_loc - pressure_im_size_required[0] * HEAD_BEND_TAXEL / 64 + 560]  # happens at legs bottom
        legs_points_R = [np.rint(v_p_bend[5]).astype(np.uint16) + 4 - HORIZ_CUT - 4,
                         head_bend_loc - pressure_im_size_required[0] * HEAD_BEND_TAXEL / 64 + 560]  # happens at legs bottom


        return [head_points_L, head_points_R, legs_points_L, legs_points_R]


    def get_3D_coord_from_cam(self, x_coord_from_camcenter, y_coord_from_camcenter, depth_value):
        f_x, f_y, c_x, c_y = self.new_K_kin[0, 0], self.new_K_kin[1, 1], self.new_K_kin[0, 2], self.new_K_kin[1, 2]
        X = (x_coord_from_camcenter)*depth_value/f_y
        Y = (y_coord_from_camcenter)*depth_value/f_x

        X += 0.418
        Y = -Y + 1.0
        Z = -depth_value + 1.54

        return X, Y, Z

    def get_pc_from_depthmap(self, bed_angle, zero_location):
        #bed_angle = 0.
        #x and y are pixel selections

        camera_to_bed_dist = 1.6
        zero_location += 0.5
        zero_location = zero_location.astype(int)

        x = np.arange(0, 440).astype(float)
        x = np.tile(x, (880, 1))
        y = np.arange(0, 880).astype(float)
        y = np.tile(y, (440, 1)).T

        x_coord_from_camcenter = x - self.depthcam_midpixel[0]
        y_coord_from_camcenter = y - self.depthcam_midpixel[1]

        depth_value = self.depth_r_orig.astype(float) / 1000

        f_x, f_y, c_x, c_y = self.new_K_kin[0, 0], self.new_K_kin[1, 1], self.new_K_kin[0, 2], self.new_K_kin[1, 2]
        X = (x_coord_from_camcenter) * depth_value / f_y
        Y = (y_coord_from_camcenter) * depth_value / f_x

        x_coord_from_camcenter_single = zero_location[0] - self.depthcam_midpixel[0]
        y_coord_from_camcenter_single = zero_location[1] - self.depthcam_midpixel[1]
        X_single = (x_coord_from_camcenter_single) * camera_to_bed_dist / f_y
        Y_single = (y_coord_from_camcenter_single) * camera_to_bed_dist / f_x

        X -= X_single
        Y -= (Y_single)

        Y = -Y
        Z = -depth_value + camera_to_bed_dist

        point_cloud = np.stack((Y, X, -Z))
        point_cloud = np.swapaxes(point_cloud, 0, 2)
        point_cloud = np.swapaxes(point_cloud, 0, 1)

        point_cloud_red = np.zeros((point_cloud.shape[0]/10, point_cloud.shape[1]/10, 3))
        for j in range(point_cloud_red.shape[0]):
            for i in range(point_cloud_red.shape[1]):
                point_cloud_red[j, i, :] = np.median(np.median(point_cloud[j*10:(j+1)*10, i*10:(i+1)*10, :], axis = 0), axis = 0)
        self.point_cloud_red = point_cloud_red.reshape(-1, 3)
        self.point_cloud = point_cloud.reshape(-1, 3)
        self.point_cloud[:, 0] += PC_WRT_ARTAG_ADJ[0] + ARTAG_WRT_PMAT[0]
        self.point_cloud[:, 1] += PC_WRT_ARTAG_ADJ[1] + ARTAG_WRT_PMAT[1]
        self.point_cloud[:, 2] += PC_WRT_ARTAG_ADJ[2] + ARTAG_WRT_PMAT[2]
        #print point_cloud.shape, 'pc shape'
        #print point_cloud_red.shape

        return X, Y, Z

    def trim_pc_sides(self):
        pc_autofil_red = self.point_cloud_autofil[self.point_cloud_autofil[:, 1] < 0.9, :] #width of bed
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 1] > -0.05, :]
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 0] > 0.15, :] #up and down bed
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 0] < 2.05, :] #up and down bed

        return pc_autofil_red



    def estimate_pose(self, pmat, bedangle, markers_c, model, model2):
        mat_size = (64, 27)


        pmat = np.fliplr(np.flipud(np.clip(pmat.reshape(MAT_SIZE)*float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100)))

        if self.CTRL_PNL['cal_noise'] == False:
            pmat = gaussian_filter(pmat, sigma=1.0)

        pmat_stack = PreprocessingLib().preprocessing_create_pressure_angle_stack_realtime(pmat, 0.0, mat_size)

        if self.CTRL_PNL['cal_noise'] == False:
            pmat_stack = np.clip(pmat_stack, a_min=0, a_max=100)

        pmat_stack = np.array(pmat_stack)
        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            pmat_contact = np.copy(pmat_stack[:, 0:1, :, :])
            pmat_contact[pmat_contact > 0] = 100
            pmat_stack = np.concatenate((pmat_contact, pmat_stack), axis=1)

        weight_input = self.weight_lbs / 2.20462
        height_input = (self.height_in * 0.0254 - 1) * 100

        batch1 = np.zeros((1, 162))
        if self.gender == 'f':
            batch1[:, 157] += 1
        elif self.gender == 'm':
            batch1[:, 158] += 1
        batch1[:, 160] += weight_input
        batch1[:, 161] += height_input

        if self.CTRL_PNL['normalize_input'] == True:
            self.CTRL_PNL['depth_map_input_est'] = False
            pmat_stack = self.TPL.normalize_network_input(pmat_stack, self.CTRL_PNL)
            batch1 = self.TPL.normalize_wt_ht(batch1, self.CTRL_PNL)

        pmat_stack = torch.Tensor(pmat_stack)
        batch1 = torch.Tensor(batch1)


        if DROPOUT == True:
            pmat_stack = pmat_stack.repeat(25, 1, 1, 1)
            batch1 = batch1.repeat(25, 1)


        batch = []
        batch.append(pmat_stack)
        batch.append(batch1)

        NUMOFOUTPUTDIMS = 3
        NUMOFOUTPUTNODES_TRAIN = 24
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)


        self.CTRL_PNL['adjust_ang_from_est'] = False
        scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpackage_batch_kin_pass(batch, False, model, self.CTRL_PNL)

        mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1) / 16.69545796387731
        mdm_est_neg = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1) / 45.08513083167194
        mdm_est_pos[mdm_est_pos < 0] = 0
        mdm_est_neg[mdm_est_neg > 0] = 0
        mdm_est_neg *= -1
        cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100 / 43.55800622930469

        # 1. / 16.69545796387731,  # pos est depth
        # 1. / 45.08513083167194,  # neg est depth
        # 1. / 43.55800622930469,  # cm est

        sc_sample1 = OUTPUT_DICT['batch_targets_est'].clone()
        sc_sample1 = sc_sample1[0, :].squeeze() / 1000
        sc_sample1 = sc_sample1.view(self.output_size_train)
        #print sc_sample1

        if model2 is not None:
            print "Using model 2"
            batch_cor = []
            batch_cor.append(torch.cat((pmat_stack[:, 0:1, :, :],
                                        mdm_est_pos.type(torch.FloatTensor),
                                        mdm_est_neg.type(torch.FloatTensor),
                                        cm_est.type(torch.FloatTensor),
                                        pmat_stack[:, 1:, :, :]), dim=1))

            if self.CTRL_PNL['full_body_rot'] == False:
                batch_cor.append(torch.cat((batch1,
                                            OUTPUT_DICT['batch_betas_est'].cpu(),
                                            OUTPUT_DICT['batch_angles_est'].cpu(),
                                            OUTPUT_DICT['batch_root_xyz_est'].cpu()), dim=1))
            elif self.CTRL_PNL['full_body_rot'] == True:
                batch_cor.append(torch.cat((batch1,
                                            OUTPUT_DICT['batch_betas_est'].cpu(),
                                            OUTPUT_DICT['batch_angles_est'].cpu(),
                                            OUTPUT_DICT['batch_root_xyz_est'].cpu(),
                                            OUTPUT_DICT['batch_root_atan2_est'].cpu()), dim=1))

            self.CTRL_PNL['adjust_ang_from_est'] = True
            scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpackage_batch_kin_pass(batch_cor, False, model2,
                                                                                        self.CTRL_PNL)



        # print betas_est, root_shift_est, angles_est
        if self.CTRL_PNL['dropout'] == True:
            print OUTPUT_DICT['verts'].shape
            smpl_verts = np.mean(OUTPUT_DICT['verts'], axis = 0)
            dropout_variance = np.std(OUTPUT_DICT['verts'], axis=0)
            dropout_variance = np.linalg.norm(dropout_variance, axis = 1)
        else:
            smpl_verts = OUTPUT_DICT['verts'][0, :, :]
            dropout_variance = None


        smpl_verts = np.concatenate((smpl_verts[:, 1:2] - 0.286 + 0.0143, smpl_verts[:, 0:1] - 0.286 + 0.0143, 2*0.075 -smpl_verts[:, 2:3]), axis = 1)

        smpl_faces = np.array(self.m.f)

        pc_autofil_red = self.trim_pc_sides() #this is the point cloud

        q = OUTPUT_DICT['batch_mdm_est'].data.numpy().reshape(OUTPUT_DICT['batch_mdm_est'].size()[0], 64, 27) * -1
        q = np.mean(q, axis = 0)

        camera_point = [1.09898028, 0.46441343, -1.53]

        if SHOW_SMPL_EST == False:
            smpl_verts *= 0.001

        #print smpl_verts

        viz_type = "3D"

        if viz_type == "2D":
            from visualization_lib import VisualizationLib
            if model2 is not None:
                self.im_sample = INPUT_DICT['batch_images'][0, 4:,:].squeeze() * 20.  # normalizing_std_constants[4]*5.  #pmat
            else:
                self.im_sample = INPUT_DICT['batch_images'][0, 1:,:].squeeze() * 20.  # normalizing_std_constants[4]*5.  #pmat
            self.im_sample_ext = INPUT_DICT['batch_images'][0, 0:, :].squeeze() * 20.  # normalizing_std_constants[0]  #pmat contact
            # self.im_sample_ext2 = INPUT_DICT['batch_images'][im_display_idx, 2:, :].squeeze()*20.#normalizing_std_constants[4]  #sobel
            self.im_sample_ext3 = OUTPUT_DICT['batch_mdm_est'][0, :, :].squeeze().unsqueeze(0) * -1  # est depth output

            # print scores[0, 10:16], 'scores of body rot'

            # print self.im_sample.size(), self.im_sample_ext.size(), self.im_sample_ext2.size(), self.im_sample_ext3.size()

            # self.publish_depth_marker_array(self.im_sample_ext3)



            self.tar_sample = INPUT_DICT['batch_targets']
            self.tar_sample = self.tar_sample[0, :].squeeze() / 1000
            sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
            sc_sample = sc_sample[0, :].squeeze() / 1000


            sc_sample = sc_sample.view(self.output_size_train)

            VisualizationLib().visualize_pressure_map(self.im_sample, sc_sample1, sc_sample,
                                                         # self.im_sample_ext, None, None,
                                                          self.im_sample_ext3, None, None, #, self.tar_sample_val, self.sc_sample_val,
                                                          block=False)

            time.sleep(4)

        elif viz_type == "3D":


            #render everything
            #self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts, smpl_faces, camera_point, bedangle,
            #                                                      pc = pc_autofil_red, pmat = pmat, smpl_render_points = False,
            #                                                      markers = None, dropout_variance = dropout_variance)


            #render in 3D pyrender with pressure mat
            #self.pyRender.render_mesh_pc_bed_pyrender(smpl_verts, smpl_faces, camera_point, bedangle,
            #                                          pc = None, pmat = pmat, smpl_render_points = False,
            #                                          facing_cam_only=False, viz_type = None,
            #                                          markers = None, segment_limbs=False)

            #render in 3D pyrender with segmented limbs
            #self.pyRender.render_mesh_pc_bed_pyrender(smpl_verts, smpl_faces, camera_point, bedangle,
            #                                          pc = None, pmat = None, smpl_render_points = False,
            #                                          facing_cam_only=False, viz_type = None,
            #                                          markers = None, segment_limbs=True)

            #render the error of point cloud points relative to verts
            #self.Render.eval_dist_render_open3d(smpl_verts, smpl_faces, pc_autofil_red, viz_type = 'pc_error',
            #                                      camera_point = camera_point, segment_limbs=False)
            self.Render.render_mesh_pc_bed_pyrender(smpl_verts, smpl_faces, camera_point, bedangle,
                                                      pc = pc_autofil_red, pmat = None, smpl_render_points = False,
                                                      facing_cam_only=True, viz_type = 'pc_error',
                                                      markers = None, segment_limbs=False)

            #render the error of verts relative to point cloud points
            #self.Render.eval_dist_render_open3d(smpl_verts, smpl_faces, pc_autofil_red, viz_type = 'mesh_error',
            #                                      camera_point = camera_point, segment_limbs=False)
            #self.pyRender.render_mesh_pc_bed_pyrender(smpl_verts, smpl_faces, camera_point, bedangle,
            #                                          pc = pc_autofil_red, pmat = None, smpl_render_points = False,
            #                                          facing_cam_only=True, viz_type = 'mesh_error',
            #                                          markers = None, segment_limbs=False)

            time.sleep(1)
            self.point_cloud_array = None



            #dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender = gender, posture = posture, stiffness = stiffness, shiftSIDE = shape_pose_vol[4], shiftUD = shape_pose_vol[5], filepath_prefix=self.filepath_prefix, add_floor = False)

            #dss.run_simulation(10000)
            #generator.standard_render()




    def evaluate_data(self, filename1, filename2=None):


        self.Render = libRender.pyRenderMesh()
        self.pyRender = libPyRender.pyRenderMesh()

        #model = torch.load(filename1, map_location={'cuda:5': 'cuda:0'})
        if GPU == True:
            for i in range(0, 8):
                try:
                    model = torch.load(filename1, map_location={'cuda:'+str(i):'cuda:0'})
                    if self.CTRL_PNL['dropout'] == True:
                        model = model.cuda().train()
                    else:
                        model = model.cuda().eval()
                    break
                except:
                    pass
            if filename2 is not None:
                for i in range(0, 8):
                    try:
                        model2 = torch.load(filename2, map_location={'cuda:'+str(i):'cuda:0'})
                        if self.CTRL_PNL['dropout'] == True:
                            model2 = model2.cuda().train()
                        else:
                            model2 = model2.cuda().eval()
                        break
                    except:
                        pass
            else:
                model2 = None
        else:
            model = torch.load(filename1, map_location='cpu')
            if self.CTRL_PNL['dropout'] == True:
                model = model.train()
            else:
                model = model.eval()
            if filename2 is not None:
                model2 = torch.load(filename2, map_location='cpu')
                if self.CTRL_PNL['dropout'] == True:
                    model2 = model2.train()
                else:
                    model2 = model2.eval()
            else:
                model2 = None

        #function_input = np.array(function_input)*np.array([10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 1])
        #function_input += np.array([2.2, 32, -1, 1.2, 32, -5, 1.0, 1.0, 0.96, 0.95, 0.8])
        function_input = np.array(self.calibration_optim_values)*np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.1])
        function_input += np.array([1.2, 32, -5, 1.0, 1.0, 0.96, 0.95])


        kinect_rotate_angle = function_input[3-3]
        kinect_shift_up = int(function_input[4-3])
        kinect_shift_right = int(function_input[5-3])
        camera_alpha_vert = function_input[6-3]
        camera_alpha_horiz = function_input[7-3]
        pressure_horiz_scale = function_input[8-3]
        pressure_vert_scale = function_input[9-3]
        #head_angle_multiplier = function_input[10-3]


        #file_dir = "/media/henry/multimodal_data_1/all_hevans_data/0905_2_Evening/0255"
        #file_dir_list = ["/media/henry/multimodal_data_2/test_data/data_072019_0001/"]
        blah = True

        #file_dir = "/media/henry/multimodal_data_2/test_data/data_072019_0007"
        #file_dir = "/media/henry/multimodal_data_2/test_data/data_072019_0006"
        #file_dir = "/home/henry/ivy_test_data/data_102019_kneeup_0000"
        #file_dir = "/media/henry/multimodal_data_1/CVPR2020_study/P000/data_102019_kneeup_0000"

        if PARTICIPANT == "P106":
            file_dir = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"_00"
            #file_dir = "/home/henry/Desktop/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"_000"
        else:
            file_dir = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"-2_00"
            #file_dir = "/home/henry/Desktop/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"-2_00"
        file_dir_nums = ["00","01","02","03","04","05","06","07","08","09"]#,"10"]#,"11","12"]
        overall_counter = 1
        overall_counter_disp = 1

        bedstatenpy = []
        colornpy = []
        config_codenpy = []
        date_stampnpy = []
        depth_rnpy = []
        markersnpy = []
        point_cloudnpy = []
        pressurenpy = []
        time_stampnpy = []

        SAVE = True

        for file_dir_num in file_dir_nums:
            file_dir_curr = file_dir + file_dir_num

            print "LOADING", file_dir_curr
            V3D.load_next_file(file_dir_curr)

            start_num = 0
            print self.color_all.shape

            #for im_num in range(29, 100):
            for im_num in range(start_num, self.color_all.shape[0]):

                if PARTICIPANT == "S103" and overall_counter in [26, 27, 28, 45, 53, 54, 55]:#, 52, 53]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S104" and overall_counter in [49, 50]: #S104 is everything but the last two
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S107" and overall_counter in [25, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S114" and overall_counter in [42, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S118" and overall_counter in [11, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S121" and overall_counter in [7, 47]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S130" and overall_counter in [30, 31, 34, 52, 53, 54, 55]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S134" and overall_counter in [49, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S140" and overall_counter in [49, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S141" and overall_counter in [49, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S145" and overall_counter in [23, 49, 50, 51]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S151" and overall_counter in [9, 48]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S163" and overall_counter in [46, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S165" and overall_counter in [19, 45]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S170" and overall_counter in [49, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S179" and overall_counter in [42, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S184" and overall_counter in [49, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S187" and overall_counter in [39, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S188" and overall_counter in [47, 50]:
                    overall_counter += 1
                    pass
                elif PARTICIPANT == "S196" and overall_counter in [20, 36]:
                    overall_counter += 1
                    pass
                #elif overall_counter < 41:# and im_num > 0:
                #    overall_counter += 1
                #    overall_counter_disp += 1
                #    pass

                else:
                    print file_dir_curr, "    subset count: ", im_num, "    overall ct: ", overall_counter_disp, overall_counter
                    overall_counter += 1
                    overall_counter_disp += 1
                    self.overall_image_scale_amount = 0.85

                    half_w_half_l = [0.4, 0.4, 1.1, 1.1]

                    all_image_list = []
                    self.label_single_image = []

                    self.label_index = 0

                    self.color = self.color_all[im_num]
                    self.depth_r = self.depth_r_all[im_num]
                    self.pressure = self.pressure_all[im_num]
                    self.bed_state = self.bedstate_all[im_num]
                    self.point_cloud_autofil = self.point_cloud_autofil_all[im_num] + [0.0, 0.0, 0.1]
                    print self.point_cloud_autofil.shape

                    self.bed_state[0] = self.bed_state[0]#*head_angle_multiplier
                    self.bed_state *= 0
                    #self.bed_state += 60.
                    print self.bed_state, np.shape(self.pressure)

                    bedstatenpy.append(self.bedstate_all[im_num])
                    colornpy.append(self.color_all[im_num])
                    config_codenpy.append(self.config_code_all[im_num])
                    date_stampnpy.append(self.date_stamp_all[im_num])
                    depth_rnpy.append(self.depth_r_all[im_num])
                    markersnpy.append(list(self.markers_all[im_num]))
                    point_cloudnpy.append(self.point_cloud_autofil_all[im_num])
                    pressurenpy.append(self.pressure_all[im_num])
                    time_stampnpy.append(self.time_stamp_all[im_num])

                    if im_num == start_num and blah == True:
                        markers_c = []
                        markers_c.append(self.markers_all[im_num][0])
                        markers_c.append(self.markers_all[im_num][1])
                        markers_c.append(self.markers_all[im_num][2])
                        markers_c.append(self.markers_all[im_num][3])
                        for idx in range(4):
                            if markers_c[idx] is not None:
                                markers_c[idx] = np.array(markers_c[idx])*213./228.
                    blah = False



                    # Get the marker points in 2D on the color image
                    u_c, v_c = ArTagLib().color_2D_markers(markers_c, self.new_K_kin)

                    # Get the marker points dropped to the height of the pressure mat
                    u_c_drop, v_c_drop, markers_c_drop = ArTagLib().color_2D_markers_drop(markers_c, self.new_K_kin)


                    # Get the geometry for sizing the pressure mat
                    pmat_ArTagLib = ArTagLib()
                    self.pressure_im_size_required, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend, half_w_half_l = \
                        pmat_ArTagLib.p_mat_geom(markers_c_drop, self.new_K_kin, self.pressure_im_size_required, self.bed_state, half_w_half_l)

                    tf_corners = np.zeros((8, 2))
                    tf_corners[0:8,:] = np.copy(self.tf_corners)


                    #COLOR
                    #if self.color is not 0:
                    color_reshaped, color_size = VizLib().color_image(self.color, self.kcam, self.new_K_kin,
                                                                      u_c, v_c, u_c_drop, v_c_drop, u_c_pmat, v_c_pmat, camera_alpha_vert, camera_alpha_horiz)
                    color_reshaped = imutils.rotate(color_reshaped, kinect_rotate_angle)
                    color_reshaped = color_reshaped[pre_VERT_CUT+kinect_shift_up:-pre_VERT_CUT+kinect_shift_up,  HORIZ_CUT+kinect_shift_right : 540 - HORIZ_CUT+kinect_shift_right, :]
                    tf_corners[0:4, :], color_reshaped = self.transform_selected_points(color_reshaped,
                                                                                                 camera_alpha_vert,
                                                                                                 camera_alpha_horiz,
                                                                                                 kinect_rotate_angle,
                                                                                                 kinect_shift_right,
                                                                                                 kinect_shift_up, [1.0, 0],
                                                                                                 [1.0, 0],
                                                                                                 np.copy(self.tf_corners[0:4][:]))

                    all_image_list.append(color_reshaped)


                    #DEPTH
                    h = VizLib().get_new_K_kin_homography(camera_alpha_vert, camera_alpha_horiz, self.new_K_kin)
                    depth_r_orig = cv2.warpPerspective(self.depth_r, h, (self.depth_r.shape[1], self.depth_r.shape[0]))
                    depth_r_orig = imutils.rotate(depth_r_orig, kinect_rotate_angle)
                    depth_r_orig = depth_r_orig[HORIZ_CUT + kinect_shift_right: 540 - HORIZ_CUT + kinect_shift_right, pre_VERT_CUT - kinect_shift_up:-pre_VERT_CUT - kinect_shift_up]
                    depth_r_reshaped, depth_r_size, depth_r_orig = VizLib().depth_image(depth_r_orig, u_c, v_c)
                    self.depth_r_orig = depth_r_orig
                    self.depthcam_midpixel = [self.new_K_kin[1, 2] - HORIZ_CUT - kinect_shift_right, (960-self.new_K_kin[0, 2]) - pre_VERT_CUT - kinect_shift_up]

                    all_image_list.append(depth_r_reshaped)


                    self.get_pc_from_depthmap(self.bed_state[0], tf_corners[2, :])

                    #PRESSURE
                    self.pressure = np.clip(self.pressure*4, 0, 100)
                    pressure_reshaped, pressure_size, coords_from_top_left = VizLib().pressure_image(self.pressure, self.pressure_im_size,
                                                                               self.pressure_im_size_required, color_size,
                                                                               u_c_drop, v_c_drop, u_c_pmat, v_c_pmat,
                                                                               u_p_bend, v_p_bend)
                    pressure_shape = pressure_reshaped.shape
                    pressure_reshaped = cv2.resize(pressure_reshaped, None, fx=pressure_horiz_scale,
                                                  fy=pressure_vert_scale)[0:pressure_shape[0],
                                                  0:pressure_shape[1], :]

                    if pressure_horiz_scale < 1.0 or pressure_vert_scale < 1.0:
                        pressure_reshaped_padded = np.zeros(pressure_shape).astype(np.uint8)
                        pressure_reshaped_padded[0:pressure_reshaped.shape[0], 0:pressure_reshaped.shape[1], :] += pressure_reshaped
                        pressure_reshaped = np.copy(pressure_reshaped_padded)

                    coords_from_top_left[0] -= coords_from_top_left[0]*(1-pressure_horiz_scale)
                    coords_from_top_left[1] += (960 - coords_from_top_left[1])*(1-pressure_vert_scale)

                    pressure_reshaped = pressure_reshaped[pre_VERT_CUT:-pre_VERT_CUT,  HORIZ_CUT : 540 - HORIZ_CUT, :]


                    all_image_list.append(pressure_reshaped)



                    self.all_images = np.zeros((960-np.abs(pre_VERT_CUT)*2, 1, 3)).astype(np.uint8)
                    for image in all_image_list:
                        print image.shape
                        self.all_images = np.concatenate((self.all_images, image), axis = 1)

                    self.all_images = self.all_images[VERT_CUT : 960 - VERT_CUT, :, :]



                    is_not_mult_4 = True
                    while is_not_mult_4 == True:
                        is_not_mult_4 = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount).shape[1]%4
                        self.overall_image_scale_amount+= 0.001

                    coords_from_top_left[0] -= (HORIZ_CUT)
                    coords_from_top_left[1] = 960 - pre_VERT_CUT - coords_from_top_left[1]
                    self.coords_from_top_left = (np.array(coords_from_top_left) * self.overall_image_scale_amount)
                    #print self.coords_from_top_left

                    self.all_images = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount)
                    self.cursor_shift = self.all_images.shape[1]/4


                    self.all_images_clone = self.all_images.copy()


                    cv2.imshow('all_images', self.all_images)
                    k = cv2.waitKey(1)
                    if SAVE == False:
                        time.sleep(5)
                    #cv2.waitKey(0)


                    #self.estimate_pose(self.pressure, self.bed_state[0], markers_c, model, model2)
        if SAVE == True:
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/color.npy", colornpy)
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/depth_r.npy", depth_rnpy)
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/pressure.npy", pressurenpy)
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/bedstate.npy", bedstatenpy)
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/markers.npy", np.array(markersnpy), allow_pickle=True)
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/time_stamp.npy", time_stampnpy)
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/point_cloud.npy", point_cloudnpy)
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/config_code.npy", config_codenpy)
            np.save("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2/date_stamp.npy", date_stampnpy)


if __name__ ==  "__main__":

    filepath_prefix = "/home/henry"
    #model_prefix = "/media/henry/multimodal_data_1"
    model_prefix = "/home/henry"
    participant_directory = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT
    #participant_directory = "/home/henry/Desktop/CVPR2020_study/"+PARTICIPANT

    V3D = Viz3DPose(filepath_prefix, participant_directory)
    #V3D.estimate_real_time(filepath_prefix+"/data/synth/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_100e_000005lr.pt",
    #                             filepath_prefix+"/data/synth/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.pt")



    #F_eval = V3D.evaluate_data(model_prefix+"/data/convnets/planesreg/112K/convnet_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_100e_00001lr.pt", \
    #                           model_prefix+"/data/convnets/planesreg_correction/112K/convnet_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt_100e_200e_00001lr.pt")
    F_eval = V3D.evaluate_data("/home/henry/data/convnets/planesreg/184K/convnet_anglesDC_synth_184K_128b_x5pmult_1.0rtojtdpth_tnh_htwt_calnoise_100e_00002lr.pt", \
                                "/home/henry/data/convnets/planesreg_correction/184K/convnet_anglesDC_synth_184000_128b_100e_x5pmult_0.5rtojtdpth_depthestin_angleadj_tnh_htwt_calnoise.pt")


    #F_eval = V3D.evaluate_data(model_prefix+"/data/convnets/planesreg/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_100e_000005lr.pt", \
    #                           model_prefix+"/data/convnets/planesreg_correction/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.pt")