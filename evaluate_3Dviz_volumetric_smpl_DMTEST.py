import numpy as np
import random
import copy
import lib_render as libRender
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
from multipose_lib import ArTagLib
from multipose_lib import VizLib
from process_yash_data import ProcessYashData
from preprocessing_lib import PreprocessingLib
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

WEIGHT_LBS = 190.
HEIGHT_IN = 78.
GENDER = 'm'

MAT_SIZE = (64, 27)

PC_WRT_ARTAG_ADJ = [0.11, -0.02, 0.07]
ARTAG_WRT_PMAT = [0.08, 0.05, 0.0]


import sys

sys.path.insert(0, '/home/henry/git/volumetric_pose_gen/convnets')
from unpack_batch_lib import UnpackBatchLib
import convnet as convnet
from torch.autograd import Variable

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


class Viz3DPose():
    def __init__(self, filepath_prefix = '/home/henry'):
        ## Load SMPL model
        self.filepath_prefix = filepath_prefix

        self.index_queue = []
        if GENDER == "m":
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
        self.CTRL_PNL['loss_vector_type'] = 'anglesEU'
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
        self.CTRL_PNL['depth_map_labels'] = False
        self.CTRL_PNL['depth_map_output'] = True
        self.CTRL_PNL['depth_map_input_est'] = False#rue #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['depth_map_input_est'] #holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True


        self.count = 0

        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['depth_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL['num_input_channels'] += 3
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.CTRL_PNL['num_input_channels'] += 2

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
        #self.config_code_all = np.load(newpath+"/config_code.npy")
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

        p_mat_offset_leg_rest = p_mat_offset_leg_rest_list[calib_index]

        legs_points_L = [np.rint(v_p_bend[4]).astype(np.uint16) - 3 - HORIZ_CUT + 4,
                         head_bend_loc - pressure_im_size_required[0] * HEAD_BEND_TAXEL / 64 + 560 + p_mat_offset_leg_rest]  # happens at legs bottom
        legs_points_R = [np.rint(v_p_bend[5]).astype(np.uint16) + 4 - HORIZ_CUT - 4,
                         head_bend_loc - pressure_im_size_required[0] * HEAD_BEND_TAXEL / 64 + 560 + p_mat_offset_leg_rest]  # happens at legs bottom


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


        pmat = np.fliplr(np.flipud(np.clip(pmat.reshape(MAT_SIZE)*5.0, a_min=0, a_max=100)))

        pmat = gaussian_filter(pmat, sigma= 0.5)


        pmat_stack = PreprocessingLib().preprocessing_create_pressure_angle_stack_realtime(pmat, bedangle, MAT_SIZE)
        pmat_stack = np.clip(pmat_stack, a_min=0, a_max=100)

        pmat_stack = np.array(pmat_stack)
        pmat_contact = np.copy(pmat_stack[:, 0:1, :, :])
        pmat_contact[pmat_contact > 0] = 100
        pmat_stack = np.concatenate((pmat_contact, pmat_stack), axis = 1)

        weight_input = WEIGHT_LBS/2.20462
        height_input = (HEIGHT_IN*0.0254 - 1)*100

        batch1 = np.zeros((1, 162))
        if GENDER == 'f':
            batch1[:, 157] += 1
        elif GENDER == 'm':
            batch1[:, 158] += 1
        batch1[:, 160] += weight_input
        batch1[:, 161] += height_input

        pmat_stack = torch.Tensor(pmat_stack)
        batch1 = torch.Tensor(batch1)

        batch = []
        batch.append(pmat_stack)
        batch.append(batch1)

        self.CTRL_PNL['adjust_ang_from_est'] = False
        scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpackage_batch_kin_pass(batch, False, model, self.CTRL_PNL)


        mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)
        mdm_est_neg = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)
        mdm_est_pos[mdm_est_pos < 0] = 0
        mdm_est_neg[mdm_est_neg > 0] = 0
        mdm_est_neg *= -1
        cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100

        batch_cor = []
        batch_cor.append(torch.cat((pmat_stack[:, 0:1, :, :],
                                  mdm_est_pos.type(torch.FloatTensor),
                                  mdm_est_neg.type(torch.FloatTensor),
                                  cm_est.type(torch.FloatTensor),
                                  pmat_stack[:, 1:, :, :]), dim=1))

        batch_cor.append(torch.cat((batch1,
                          OUTPUT_DICT['batch_betas_est'].cpu(),
                          OUTPUT_DICT['batch_angles_est'].cpu(),
                          OUTPUT_DICT['batch_root_xyz_est'].cpu()), dim = 1))


        self.CTRL_PNL['adjust_ang_from_est'] = True
        scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpackage_batch_kin_pass(batch_cor, False, model2, self.CTRL_PNL)

        betas_est = np.squeeze(OUTPUT_DICT['batch_betas_est_post_clip'].cpu().numpy())
        angles_est = np.squeeze(OUTPUT_DICT['batch_angles_est_post_clip'])
        root_shift_est = np.squeeze(OUTPUT_DICT['batch_root_xyz_est_post_clip'].cpu().numpy())


        #print betas_est.shape, root_shift_est.shape, angles_est.shape

        #print betas_est, root_shift_est, angles_est
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

        #get SMPL mesh
        smpl_verts = (self.m.r - self.m.J_transformed[0, :])+[root_shift_est[1]-0.286+0.15, root_shift_est[0]-0.286, 0.12-root_shift_est[2]]#*228./214.
        smpl_faces = np.array(self.m.f)

        pc_autofil_red = self.trim_pc_sides()



        self.pyRender.mesh_render_pose_bed(smpl_verts, smpl_faces, pc_autofil_red, self.pc_isnew, pmat, markers_c, bedangle, segment_limbs=False)
        self.point_cloud_array = None



        #dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender = gender, posture = posture, stiffness = stiffness, shiftSIDE = shape_pose_vol[4], shiftUD = shape_pose_vol[5], filepath_prefix=self.filepath_prefix, add_floor = False)

        #dss.run_simulation(10000)
        #generator.standard_render()




    def evaluate_data(self, function_input, filename1, filename2):


        self.pyRender = libRender.pyRenderMesh()

        if GPU == True:
            model = torch.load(filename1)
            model = model.cuda().eval()
            if filename2 is not None:
                model2 = torch.load(filename2)
                model2 = model2.cuda().eval()

        else:
            model = torch.load(filename1, map_location='cpu').eval()
            if filename2 is not None:
                model2 = torch.load(filename2, map_location='cpu').eval()


        function_input = np.array(function_input)*np.array([10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 1])
        function_input += np.array([2.2, 32, -1, 1.2, 32, -5, 1.0, 1.0, 0.96, 0.95, 0.8])


        kinect_rotate_angle = function_input[3]
        kinect_shift_up = int(function_input[4])
        kinect_shift_right = int(function_input[5])
        camera_alpha_vert = function_input[6]
        camera_alpha_horiz = function_input[7]
        pressure_horiz_scale = function_input[8]
        pressure_vert_scale = function_input[9]
        head_angle_multiplier = function_input[10]

        self.posture = posture

        tf_corners = {}
        #file_dir = "/media/henry/multimodal_data_1/all_hevans_data/0905_2_Evening/0255"
        #file_dir_list = ["/media/henry/multimodal_data_2/test_data/data_072019_0001/"]
        blah = True

        #file_dir = "/media/henry/multimodal_data_2/test_data/data_072019_0007"
        file_dir = "/media/henry/multimodal_data_2/test_data/data_072019_0006"

        V3D.load_next_file(file_dir)

        start_num = 51
        #for im_num in range(29, 100):
        for im_num in range(start_num, 100):

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

            self.bed_state[0] = self.bed_state[0]*head_angle_multiplier
            self.bed_state *= 0
            #self.bed_state += 60.
            print self.bed_state, np.shape(self.pressure)

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

            tf_corners[posture] = np.zeros((12, 2))
            tf_corners[posture][0:8,:] = np.copy(calib_corners_selection[calib_index][posture])


            #COLOR
            #if self.color is not 0:
            color_reshaped, color_size = VizLib().color_image(self.color, self.kcam, self.new_K_kin,
                                                              u_c, v_c, u_c_drop, v_c_drop, u_c_pmat, v_c_pmat, camera_alpha_vert, camera_alpha_horiz)
            color_reshaped = imutils.rotate(color_reshaped, kinect_rotate_angle)
            color_reshaped = color_reshaped[pre_VERT_CUT+kinect_shift_up:-pre_VERT_CUT+kinect_shift_up,  HORIZ_CUT+kinect_shift_right : 540 - HORIZ_CUT+kinect_shift_right, :]
            tf_corners[posture][0:4, :], color_reshaped = self.transform_selected_points(color_reshaped,
                                                                                         camera_alpha_vert,
                                                                                         camera_alpha_horiz,
                                                                                         kinect_rotate_angle,
                                                                                         kinect_shift_right,
                                                                                         kinect_shift_up, [1.0, 0],
                                                                                         [1.0, 0],
                                                                                         np.copy(calib_corners_selection[calib_index][posture][0:4][:]))

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


            self.get_pc_from_depthmap(self.bed_state[0], tf_corners[posture][2, :])

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


            #cv2.imshow('all_images', self.all_images)
            #k = cv2.waitKey(1)
            #cv2.waitKey(0)


            self.estimate_pose(self.pressure, self.bed_state[0], markers_c, model, model2)


if __name__ ==  "__main__":

    filepath_prefix = "/home/henry"
    model_prefix = "/media/henry/multimodal_data_2"


    V3D = Viz3DPose(filepath_prefix)
    #V3D.estimate_real_time(filepath_prefix+"/data/synth/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_100e_000005lr.pt",
    #                             filepath_prefix+"/data/synth/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.pt")

    file_optimization_picks = [[-0.10380950868835398, -0.24954999664283403, -0.19082899491276584, #05 to 06
                                -0.08444541039339573, 0.016119578206939777, -0.3216531438714622,
                                0.1394849993002632, -0.2842603728947779, 0.13685526544739327,
                                0.03574465742073685, 0.04634027846367891],
                               [0.05001938745147258, 0.15123731597285675, 0.0866959183560595,
                                0.05910474787693074, 0.3861988024865718, -0.17826400482747298,
                                0.09487984735370396, 0.17963550843844123, -0.07599611870098816,
                                -0.034540931786485825, -0.04390890429482567],
                               [-0.13162189596458163, -0.13417259496587142, 0.26466890673798366,
                                -0.1270856703698008, -0.11680215525856938, 0.07887195356115867,
                                0.30269562740592326, 0.59059074808585, -0.24382594982402644,
                                0.13349612802154237, 0.0329615160283741],
                               [-0.017941161914079823, -0.5319409757913965, -0.1475376924455061,
                                -0.029145428963456153, -0.3602627899643398, -0.3772389925744474,
                                0.4807884578370175, 0.2707353595745977, -0.29252052658027033,
                                -0.09886303402191773, -0.004106658497690778],
                               [-0.07327427816490228, -1.2399041363573815, 0.308719334121065,
                                -0.0585650256965522, -1.0359134550133717, 0.012588536106676429,
                                0.31464529310245837, 0.2729018569120285, -0.3683596171492809,
                                0.0016488972973007776, 0.018401395618589333],
                               [-0.10041624345192472, -0.35433669709825616, -0.04952849138799903, #10 to 11
                                -0.10214471176768623, -0.31154356246657955, -0.27799933679718264,
                                0.1081511690729292, 0.3181850307846851, 0.022649443479678604,
                                0.017340699560127338, 0.04380688163789306],
                               [-0.10478270054644222, -0.28345417095232933, -0.07807829083158871,
                                -0.11153260474732456, 0.05599584968235911, -0.05463914177827217,
                                0.19729283572173112, 0.540660476138503, -0.08226193308574829, #11 to 12
                                -0.005466117946628913, -0.008427016596327028]]


    calib_corners_selection = [{'lay': [[44.60093896713615, 186.61971830985917], [360.3286384976526, 196.0093896713615],
                                        [82.15962441314554, 744.131455399061], [332.15962441314554, 741.7840375586854],
                                        [56.33802816901409, 180.7511737089202], [373.23943661971833, 193.66197183098592],
                                        [85.68075117370893, 740.6103286384977], [332.15962441314554, 741.7840375586854]],
                                'sit': [[76.29107981220658, 140.8450704225352], [339.2018779342723, 149.06103286384976],
                                        [84.50704225352113, 741.7840375586854], [334.50704225352115, 737.0892018779343],
                                        [84.50704225352113, 133.80281690140845], [348.59154929577466, 146.71361502347418],
                                        [86.85446009389672, 738.2629107981221], [334.50704225352115, 737.0892018779343]]}, #this is for 05-06
                               {'lay': [[84.50704225352113, 134.97652582159625], [338.02816901408454, 134.97652582159625],
                                        [64.55399061032864, 724.1784037558685], [320.4225352112676, 724.1784037558685],
                                        [95.07042253521126, 130.2816901408451], [348.59154929577466, 133.80281690140845],
                                        [69.24882629107981, 719.4835680751174], [318.07511737089203, 725.3521126760563]],
                                'sit': [[52.816901408450704, 144.3661971830986], [369.71830985915494, 156.10328638497654],
                                        [66.90140845070422, 725.3521126760563], [320.4225352112676, 726.5258215962441],
                                        [64.55399061032864, 138.49765258215962], [380.28169014084506, 157.27699530516432],
                                        [69.24882629107981, 721.830985915493], [321.5962441314554, 728.8732394366198]]}, #this is for 07-08
                               {'lay': [[72.7699530516432, 144.3661971830986], [347.4178403755869, 136.15023474178403],
                                        [73.94366197183099, 717.1361502347418], [327.46478873239437, 707.7464788732394],
                                        [82.15962441314554, 139.67136150234742], [357.981220657277, 134.97652582159625],
                                        [76.29107981220658, 717.1361502347418], [327.46478873239437, 706.5727699530516]],
                                'sit': [[45.774647887323944, 150.23474178403757], [377.9342723004695, 157.27699530516432],
                                        [73.94366197183099, 721.830985915493], [326.2910798122066, 710.0938967136151],
                                        [58.68544600938967, 147.88732394366198], [390.84507042253523, 157.27699530516432],
                                        [77.46478873239437, 717.1361502347418], [327.46478873239437, 708.9201877934272]]}, #this is for 09-10
                               {'lay': [[86.85446009389672, 169.01408450704227], [332.15962441314554, 173.70892018779344],
                                        [79.81220657276995, 714.7887323943662], [329.81220657277, 705.3990610328639],
                                        [93.89671361502347, 164.31924882629107], [339.2018779342723, 173.70892018779344],
                                        [80.98591549295774, 708.9201877934272], [327.46478873239437, 707.7464788732394]],
                                'sit': [[49.29577464788733, 138.49765258215962], [363.849765258216, 144.3661971830986],
                                        [75.11737089201878, 713.6150234741784], [329.81220657277, 711.2676056338029],
                                        [59.859154929577464, 131.45539906103286], [375.5868544600939, 140.8450704225352],
                                        [77.46478873239437, 708.9201877934272], [326.2910798122066, 708.9201877934272]]}] #this is for 11-12

    pmat_head_corner_shift_list = [[0.0, 0.0],
                                  [0.0, 0.0],
                                  [0.0, 3.5],
                                  [0.0, 2.0],
                                  [0.0, 3.0],
                                  [0.0, 5.0]]

    p_mat_offset_leg_rest_list = [0, 4, -2, 4, -10, -5]

    calib_index = 0


    #file_dir = "/home/henry/test/calib_data_082018_0096"

    # file_dir = "/media/henry/multimodal_data_1/all_hevans_data/Calibration_0907_Evening_0908_Morning/data_082018_0096"
    # 96 - 85 is a good pick for seated
    # 98 - 40 is a good pick for flat


    posture = "lay"
    if posture == "lay":
        folder_idx = 1
        image_idx = 2
    elif posture == "sit":
        folder_idx = 3
        image_idx = 4

    F_eval = V3D.evaluate_data(file_optimization_picks[calib_index], \
                               model_prefix+"/data/convnets/planesreg/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_100e_000005lr.pt", \
                               model_prefix+"/data/convnets/planesreg_correction/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.pt")