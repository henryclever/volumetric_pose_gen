#!/usr/bin/env python
import rospy
import roslib
roslib.load_manifest('multimodal_pose')
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from hrl_msgs.msg import FloatArrayBare
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ar_track_alvar_msgs.msg import AlvarMarkers
import os.path as osp
from camera import Camera
import pickle
import time
import imutils
import math
import cPickle as pkl
import os
SHORT = False

import rospy


from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker

from multipose_lib import ArTagLib
from multipose_lib import VizLib



COLOR, DEPTH, PRESSURE = True, True, True
#VERT_CUT, VERT_CUT, HORIZ_CUT, HORIZ_CUT = 215, 220, 150, 115
VERT_CUT, HORIZ_CUT = 0, 50
pre_VERT_CUT = 40


 
class RealTimeViz():
    def __init__(self):
        self.bridge = CvBridge()

        self.color, self.depth_r, self.pressure = 0, 0, 0

        self.computer = 'henry'

        self.kinect_im_size = (960, 540)
        self.pressure_im_size = (64, 27)
        self.pressure_im_size_required = (64, 27)

        # initialization of kinect and thermal cam calibrations from YAML files
        dist_model = 'rational_polynomial'
        self.kcam = Camera('kinect', self.kinect_im_size, dist_model)
        self.kcam.init_from_yaml(osp.expanduser('~/catkin_ws/src/multimodal_pose/calibrations/kinect.yaml'))

        #we are at qhd not hd so need to cut the focal lengths and centers in half
        self.kcam.K[0:2,0:3] = self.kcam.K[0:2,0:3]/2

        print self.kcam.K

        self.new_K_kin, roi = cv2.getOptimalNewCameraMatrix(self.kcam.K, self.kcam.D, self.kinect_im_size, 1, self.kinect_im_size)



        self.drawing = False  # true if mouse is pressed
        self.mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
        self.ix, self.iy = -1, -1
        self.label_index = 0
        self.coords_from_top_left = [0, 0]
        self.overall_image_scale_amount = 0.85
        self.depthcam_midpixel = [0, 0]
        self.select_new_calib_corners = True
        self.calib_corners = 8*[[0, 0]]


        self.filler_taxels = []
        for i in range(28):
            for j in range(65):
                self.filler_taxels.append([i - 1, j - 1, 20000])
        self.filler_taxels = np.array(self.filler_taxels).astype(int)


    def truncate(self, number, digits):
        stepper = pow(10.0, digits)
        return math.trunc(stepper * number) / stepper

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

        point_cloud = np.stack((X,Y,Z))
        point_cloud = np.swapaxes(point_cloud, 0, 2)
        point_cloud = np.swapaxes(point_cloud, 0, 1)

        point_cloud_red = np.zeros((point_cloud.shape[0]/10, point_cloud.shape[1]/10, 3))
        for j in range(point_cloud_red.shape[0]):
            for i in range(point_cloud_red.shape[1]):
                point_cloud_red[j, i, :] = np.median(np.median(point_cloud[j*10:(j+1)*10, i*10:(i+1)*10, :], axis = 0), axis = 0)
        point_cloud_red = point_cloud_red.reshape(-1, 3)

        PointCloudArray = MarkerArray()
        pointcloudPublisher = rospy.Publisher("/point_cloud", MarkerArray)
        self.publish_markerarray(PointCloudArray, pointcloudPublisher, point_cloud_red)

        #get distance to the mat
        bed_angle = bed_angle*np.pi/180.
        bed_angle_sin = np.sin(bed_angle)
        bed_angle_cos = np.cos(bed_angle)

        point_cloud = point_cloud.reshape(-1, 3)

        point_cloud[:, 1] += 0.286
        point_cloud_taxel = np.copy(point_cloud)
        point_cloud_rot_taxel = np.copy(point_cloud)

        bend_taxel_loc = 52
        bend_loc = bend_taxel_loc*0.0286

        bending_3d_point = np.array([[-0.2, bend_loc-0.286, 0.0]])
        BendLocArray = MarkerArray()
        bendlocPublisher = rospy.Publisher("/bend_loc", MarkerArray)
        self.publish_markerarray(BendLocArray, bendlocPublisher, bending_3d_point,
                                 markertype=Marker().ARROW, color = [1.0, 0.0, 0.0], x_len = 1.5)


        point_cloud_rot_taxel[:, 1] = bed_angle_sin*point_cloud_taxel[:, 2] - bed_angle_cos*(bend_loc - point_cloud_taxel[:, 1]) + bend_loc
        point_cloud_rot_taxel[:, 2] = bed_angle_cos*point_cloud_taxel[:, 2] + bed_angle_sin*(bend_loc - point_cloud_taxel[:, 1])

        #import matplotlib.pyplot as plt
        #plt.plot(-point_cloud_taxel[:, 1], point_cloud[:, 2], 'r.')

        body_point_cloud = point_cloud_taxel[point_cloud_taxel[:, 1] < bend_loc]
        head_point_cloud = point_cloud_rot_taxel[point_cloud_rot_taxel[:, 1] >= bend_loc]

        point_cloud_taxel = np.concatenate((point_cloud_taxel, point_cloud_taxel[0:200000, :]*0+3.0), axis = 0)

        point_cloud_taxel[0:body_point_cloud.shape[0], :] = body_point_cloud
        point_cloud_taxel[body_point_cloud.shape[0]:(body_point_cloud.shape[0] + head_point_cloud.shape[0]), :] = head_point_cloud
        point_cloud_taxel[body_point_cloud.shape[0] + head_point_cloud.shape[0]:, :] *= 0

        #plt.plot(-point_cloud_taxel[:, 1], point_cloud_taxel[:, 2], 'k.')
        #plt.axis([-2.5, 1.0, -0.3, 1.0])
        #plt.show()

        point_cloud_taxel[:, 1] -= 0.286
        point_cloud_taxel /= 0.0286
        point_cloud_taxel[:, 2] *= 1000
        point_cloud_taxel[:, 0] *= 1.04
        point_cloud_taxel_int = point_cloud_taxel.astype(int)

        point_cloud_taxel_int = np.concatenate((self.filler_taxels, point_cloud_taxel_int), axis=0)

        point_cloud_sorting_method = (point_cloud_taxel_int[:, 0:1] + 1) * 10000000 + \
                                     (point_cloud_taxel_int[:, 1:2] + 1) * 100000 + \
                                      point_cloud_taxel_int[:, 2:3]

        point_cloud_taxel_int = np.concatenate((point_cloud_sorting_method, point_cloud_taxel_int), axis=1)

        point_cloud_taxel_int = np.unique(point_cloud_taxel_int, return_inverse=False, axis=0)

        point_cloud_taxel_int[1:, 0] = np.abs((point_cloud_taxel_int[:-1, 1] - point_cloud_taxel_int[1:, 1]) +
                                              (point_cloud_taxel_int[:-1, 2] - point_cloud_taxel_int[1:, 2]))
        point_cloud_taxel_int[1:, 1:4] = point_cloud_taxel_int[1:, 1:4] * point_cloud_taxel_int[1:, 0:1]
        point_cloud_taxel_int = point_cloud_taxel_int[point_cloud_taxel_int[:, 0] != 0, :]
        point_cloud_taxel_int = point_cloud_taxel_int[:, 1:]
        point_cloud_taxel_int = point_cloud_taxel_int[point_cloud_taxel_int[:, 1] < 64, :]
        point_cloud_taxel_int = point_cloud_taxel_int[point_cloud_taxel_int[:, 1] >= 0, :]
        point_cloud_taxel_int = point_cloud_taxel_int[point_cloud_taxel_int[:, 0] < 27, :]
        point_cloud_taxel_int = point_cloud_taxel_int[point_cloud_taxel_int[:, 0] >= 0, :]

        point_cloud_taxel_int[point_cloud_taxel_int == 20000] = 0
        point_cloud_taxel_int = point_cloud_taxel_int.astype(float)
        point_cloud_taxel_int[:, 2] /= 1000

        point_cloud_taxel_int *= 0.0286

        DepthMapArray = MarkerArray()
        depthmapPublisher = rospy.Publisher("/depth_map", MarkerArray)
        self.publish_markerarray(DepthMapArray, depthmapPublisher, point_cloud_taxel_int)

        return X, Y, Z



    def publish_markerarray(self, TargetArray, publisher, data, markertype = Marker().SPHERE, color = [0.0, 0.69, 0.0], x_len = 0.07):
        for joint in range(0, data.shape[0]):
            Tmarker = Marker()
            Tmarker.type = markertype
            Tmarker.header.frame_id = "map"
            Tmarker.action = Tmarker.ADD
            Tmarker.scale.x = x_len
            Tmarker.scale.y = 0.07
            Tmarker.scale.z = 0.07
            Tmarker.color.a = 1.0
            Tmarker.color.r = color[0]
            Tmarker.color.g = color[1]
            Tmarker.color.b = color[2]
            Tmarker.pose.orientation.w = 1.0
            Tmarker.pose.position.x = data[joint, 0]  # - INTER_SENSOR_DISTANCE * 10
            Tmarker.pose.position.y = data[joint, 1]  # - INTER_SENSOR_DISTANCE * 10
            Tmarker.pose.position.z = data[joint, 2]
            TargetArray.markers.append(Tmarker)
            tid = 0
            for m in TargetArray.markers:
                m.id = tid
                tid += 1
        # print TargetArray
        publisher.publish(TargetArray)


    def draw_rectangles(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:


            self.all_images = self.all_images_clone.copy()

            if self.drawing == True:
                for i in [-3, -2, -1, 0, 1, 2, 3]:
                    cv2.rectangle(self.all_images, (self.ix+self.cursor_shift*i, self.iy), (x+self.cursor_shift*i, y), (0, 255, 0), 2)

                #cv2.imshow('all_images', self.all_images_clone)

            for i in [-3, -2, -1, 0, 1, 2, 3]:
                cv2.arrowedLine(self.all_images, (x + 20 + self.cursor_shift*i, y + 20), (x + self.cursor_shift*i, y),  (0, 0, 255), 2)

            self.all_images[0:70, 400:, :] = 255

            x_coord_from_bed_corner = x-self.coords_from_top_left[0]
            self.x_coord_from_corner = int(x)*1
            x_coord_from_camcenter = float(x)/self.overall_image_scale_amount-self.depthcam_midpixel[0]
            x_depth_image_idx = float(x)/self.overall_image_scale_amount

            if x > self.cursor_shift and x <= 2*self.cursor_shift:
                x_coord_from_bed_corner -= self.cursor_shift
                self.x_coord_from_corner -= self.cursor_shift
                x_depth_image_idx -= self.cursor_shift/self.overall_image_scale_amount
                x_coord_from_camcenter -= self.cursor_shift/self.overall_image_scale_amount
            elif x > 2*self.cursor_shift and x <= 3*self.cursor_shift:
                x_coord_from_bed_corner -= 2*self.cursor_shift
                self.x_coord_from_corner -= 2*self.cursor_shift
                x_depth_image_idx -= 2*self.cursor_shift/self.overall_image_scale_amount
                x_coord_from_camcenter -= 2*self.cursor_shift/self.overall_image_scale_amount
            elif x > 3*self.cursor_shift and x <= 4*self.cursor_shift:
                x_coord_from_bed_corner -= 3*self.cursor_shift
                self.x_coord_from_corner -= 3*self.cursor_shift
                x_depth_image_idx -= 3*self.cursor_shift/self.overall_image_scale_amount
                x_coord_from_camcenter -= 3*self.cursor_shift/self.overall_image_scale_amount

            y_coord_from_bed_corner = -(y-self.coords_from_top_left[1])
            self.y_coord_from_corner = int(y)*1
            x_coord_from_bed_corner/=self.overall_image_scale_amount
            self.x_coord_from_corner/=self.overall_image_scale_amount
            y_coord_from_bed_corner/=self.overall_image_scale_amount
            self.y_coord_from_corner/=self.overall_image_scale_amount

            y_depth_image_idx = float(y)/self.overall_image_scale_amount
            y_coord_from_camcenter = float(y)/self.overall_image_scale_amount-self.depthcam_midpixel[1]

            depth_value = float(self.depth_r_orig[int(y_depth_image_idx), int(x_depth_image_idx-1)])/1000
            #print depth_value, int(x_depth_image_idx), int(y_depth_image_idx)

            X, Y, Z = self.get_3D_coord_from_cam(x_coord_from_camcenter, y_coord_from_camcenter, depth_value)

            cv2.putText(self.all_images, "pixels from bed corner: "+str(int(x_coord_from_bed_corner))+", "+str(int(y_coord_from_bed_corner)), (800, 26), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(self.all_images, "pixels from corner: "+str(int(self.x_coord_from_corner))+", "+str(int(self.y_coord_from_corner)), (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            #cv2.putText(self.all_images, "dist from corner: "+str(self.truncate(X, 3))+", "+str(self.truncate(Y, 3))+", "+str(self.truncate(Z, 3))+", ", (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(self.all_images, self.labels[self.label_index], (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)



        elif event == cv2.EVENT_LBUTTONUP:

            self.drawing = False
            cv2.rectangle(self.all_images, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.all_images_clone = self.all_images.copy()

            self.label_single_image.append([self.ix, self.iy, x, y])


            if OPTIM == True:
                if self.select_new_calib_corners == True:
                    self.calib_corners[self.label_index] = [self.x_coord_from_corner, self.y_coord_from_corner]
                    #print self.calib_corners

            self.label_index += 1


    def load_next_file(self, newpath):

        print "loading existing npy files in the new path...."
        time_orig = time.time()
        self.color_all = np.load(newpath+"/color.npy")
        self.depth_r_all = np.load(newpath+"/depth_r.npy")
        self.pressure_all = np.load(newpath+"/pressure.npy")
        self.bedstate_all = np.load(newpath+"/bedstate.npy")
        self.markers_all = np.load(newpath+"/markers.npy", allow_pickle=True)
        self.time_stamp_all = np.load(newpath+"/time_stamp.npy")
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


    def optim_calib_criteria(self, function_input):
        print function_input

        # file_dir = "/media/henry/multimodal_data_1/all_hevans_data/Calibration_0907_Evening_0908_Morning/data_082018_0096"
        # 96 - 85 is a good pick for seated
        # 98 - 40 is a good pick for flat


        self.labels = ["Up L Color", "Up R Color", "Low L Color", "Low R Color",
                       "Up L Pressure", "Up R Pressure", "Low L Pressure", "Low R Pressure"]

        function_input = np.array(function_input)*np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.1])
        function_input += np.array([1.2, 32, -5, 1.0, 1.0, 0.96, 0.95])

        if self.select_new_calib_corners == True:
            function_input *= 0
            function_input[3:7] += 1

        #function_input[5] = float(function_input[6])*0 + 0.5

        kinect_rotate_angle = function_input[0]
        kinect_shift_up = int(function_input[1])
        kinect_shift_right = int(function_input[2])
        camera_alpha_vert = function_input[3]
        camera_alpha_horiz = function_input[4]
        pressure_horiz_scale = function_input[5]
        pressure_vert_scale = function_input[6]


        #print camera_alpha_vert, camera_alpha_horiz, pressure_horiz_scale, pressure_vert_scale, pressure_vert_scale

        #print kinect_shift_up

        #if kinect_shift_up >= pre_VERT_CUT-1: return 3000.
        #elif thermal_shift_up >= pre_VERT_CUT-1: return 4000.

        F_eval = 0

        pmat_head_corner_idx = 1
        folder_idx = 1
        image_idx = 2


        pmat_head_corner_shift = pmat_head_corner_shift_list[calib_index][pmat_head_corner_idx]

        file_dir = file_dir_list[calib_index][0] + file_dir_list[calib_index][folder_idx]

        file_parent_dir = file_dir_list[calib_index][0]

        rtv.load_next_file(file_dir)

        im_num = file_dir_list[calib_index][image_idx]


        self.overall_image_scale_amount = 0.85

        cv2.namedWindow('all_images')
        cv2.setMouseCallback('all_images', self.draw_rectangles)

        half_w_half_l = [0.4, 0.4, 1.1, 1.1]
        label_all_images = []

        all_image_list = []
        self.label_single_image = []

        self.label_index = 0



        self.color = self.color_all[im_num]
        self.depth_r = self.depth_r_all[im_num]
        self.pressure = self.pressure_all[im_num]
        self.bed_state = self.bedstate_all[im_num]


        self.bed_state[0] = self.bed_state[0]*0.0
        self.bed_state[1] = self.bed_state[1]*0.0
        self.bed_state[2] = self.bed_state[2]*0.0

        #print self.bed_state, im_num


        markers_c = []
        markers_c.append(self.markers_all[im_num][0])
        markers_c.append(self.markers_all[im_num][1])
        markers_c.append(self.markers_all[im_num][2])
        markers_c.append(self.markers_all[im_num][3])

        # Get the marker points in 2D on the color image
        u_c, v_c = ArTagLib().color_2D_markers(markers_c, self.new_K_kin)

        # Get the marker points dropped to the height of the pressure mat
        u_c_drop, v_c_drop, markers_c_drop = ArTagLib().color_2D_markers_drop(markers_c, self.new_K_kin)

        # Get the geometry for sizing the pressure mat
        pmat_ArTagLib = ArTagLib(pmat_head_corner_shift)
        self.pressure_im_size_required, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend, half_w_half_l = \
            pmat_ArTagLib.p_mat_geom(markers_c_drop, self.new_K_kin, self.pressure_im_size_required, self.bed_state, half_w_half_l)

        u_p_bend_calib = pmat_ArTagLib.u_p_bend_calib
        v_p_bend_calib = pmat_ArTagLib.v_p_bend_calib



        #self.calib_corners[posture] = [[49, 143], [369, 154], [63, 725], [319, 729], [63, 136], [379, 155], [68, 721], [317, 727], [46, 129],
        # [385, 131], [87, 743], [350, 739]]
        tf_corners = np.zeros((8, 2))
        tf_corners[0:8,:] = np.copy(self.calib_corners)
        #print tf_corners

        #print self.time_stamp_all[im_num], self.config_code_all[im_num]


        #COLOR
        #if self.color is not 0:
        color_reshaped, color_size = VizLib().color_image(self.color, self.kcam, self.new_K_kin,
                                                          u_c, v_c, u_c_drop, v_c_drop, u_c_pmat, v_c_pmat, camera_alpha_vert, camera_alpha_horiz)

        color_reshaped = imutils.rotate(color_reshaped, kinect_rotate_angle)
        color_reshaped = color_reshaped[pre_VERT_CUT+kinect_shift_up:-pre_VERT_CUT+kinect_shift_up,  HORIZ_CUT+kinect_shift_right : 540 - HORIZ_CUT+kinect_shift_right, :]
        tf_corners[0:4, :], color_reshaped = self.transform_selected_points(color_reshaped, camera_alpha_vert, camera_alpha_horiz, kinect_rotate_angle,
                                                                           kinect_shift_right, kinect_shift_up, [1.0, 0], [1.0, 0],
                                                                           np.copy(self.calib_corners[0:4][:]))

        all_image_list.append(color_reshaped)

        #DEPTH
        if self.depth_r is not 0 and DEPTH is True:
            h = VizLib().get_new_K_kin_homography(camera_alpha_vert, camera_alpha_horiz, self.new_K_kin)
            depth_r_orig = cv2.warpPerspective(self.depth_r, h, (self.depth_r.shape[1], self.depth_r.shape[0]))
            depth_r_orig = imutils.rotate(depth_r_orig, kinect_rotate_angle)
            depth_r_orig = depth_r_orig[HORIZ_CUT + kinect_shift_right: 540 - HORIZ_CUT + kinect_shift_right, pre_VERT_CUT - kinect_shift_up:-pre_VERT_CUT - kinect_shift_up]
            depth_r_reshaped, depth_r_size, depth_r_orig = VizLib().depth_image(depth_r_orig, u_c, v_c)
            self.depth_r_orig = depth_r_orig
            self.depthcam_midpixel = [self.new_K_kin[1, 2] - HORIZ_CUT - kinect_shift_right, (960-self.new_K_kin[0, 2]) - pre_VERT_CUT - kinect_shift_up]

            #depth_r_reshaped, depth_r_size, depth_r_orig = VizLib().depth_image(self.depth_r, u_c, v_c)
            #depth_r_reshaped = imutils.rotate(depth_r_reshaped, kinect_rotate_angle)
            #depth_r_shape = depth_r_reshaped.shape
            #depth_r_reshaped = cv2.resize(depth_r_reshaped, None, fx=camera_scale, fy=camera_scale*camera_vertical_scale)[0:depth_r_shape[0], 0:depth_r_shape[1], :]
            #depth_r_reshaped = depth_r_reshaped[pre_VERT_CUT+kinect_shift_up:-pre_VERT_CUT+kinect_shift_up,  HORIZ_CUT+kinect_shift_right : 540 - HORIZ_CUT+kinect_shift_right, :]
            all_image_list.append(depth_r_reshaped)

        #PRESSURE
        if self.pressure is not 0:
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

            #print tf_corners, 'tf corners1'
            #tf_corners[4:8, :] = self.rotate_selected_head_points(self.pressure_im_size_required,
            #                                                                u_c_pmat, v_c_pmat,
            #                                                                u_p_bend, v_p_bend,
            #                                                                u_p_bend_calib, v_p_bend_calib)

            #print tf_corners, 'tf corners1'
            tf_corners[4:8, :], pressure_reshaped = self.transform_selected_points(pressure_reshaped, 1.0, 1.0, 0.0, 0, 0,
                                                                               [pressure_horiz_scale, HORIZ_CUT],
                                                                               [pressure_vert_scale, pre_VERT_CUT],
                                                                               tf_corners[4:8, :])

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

            coords_from_top_left[0] -= (HORIZ_CUT)
            coords_from_top_left[1] = 960 - pre_VERT_CUT - coords_from_top_left[1]
            self.coords_from_top_left = (np.array(coords_from_top_left) * self.overall_image_scale_amount)
            #print self.coords_from_top_left

            self.all_images = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount)
            self.cursor_shift = self.all_images.shape[1]/3


            self.all_images_clone = self.all_images.copy()


            #self.select_new_calib_corners = True
            if self.select_new_calib_corners == True:
                while self.label_index < 0:
                    print "got here", self.all_images.shape
                    print self.calib_corners
                    cv2.namedWindow('all_images')
                    cv2.setMouseCallback('all_images', self.draw_rectangles)

                    print "got here2"
                    cv2.imshow('all_images', self.all_images)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('m'):
                        self.mode = not self.mode
                    elif k == 27:
                        break
                    print "got here3"
                    time.sleep(0.01)

                print self.calib_corners

                self.calib_corners = calib_corners_selection[calib_index]


                self.select_new_calib_corners = False
                F_eval += 1000

            else:
                print np.linalg.norm(tf_corners[0, :]-tf_corners[4, :])
                print np.linalg.norm(tf_corners[1, :]-tf_corners[5, :])
                print np.linalg.norm(tf_corners[2, :]-tf_corners[6, :])
                print np.linalg.norm(tf_corners[3, :]-tf_corners[7, :])
                print self.calib_corners
                #print tf_corners, 'tf2'
                f_eval1 = np.linalg.norm(tf_corners[0, :]-tf_corners[4, :]) + \
                          np.linalg.norm(tf_corners[1, :]-tf_corners[5, :]) + \
                          np.linalg.norm(tf_corners[2, :]-tf_corners[6, :]) + \
                          np.linalg.norm(tf_corners[3, :]-tf_corners[7, :])


                F_eval += f_eval1



                cv2.imshow('all_images', self.all_images)
                k = cv2.waitKey(1)
                print "F_eval is ", F_eval

            print "Now its", self.select_new_calib_corners
            label_all_images.append(self.label_single_image)


            #cv2.waitKey(0)
        return F_eval

    def evaluate_data(self, function_input):


        function_input = np.array(function_input)*np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.1])
        function_input += np.array([1.2, 32, -5, 1.0, 1.0, 0.96, 0.95])

        kinect_rotate_angle = function_input[0]
        kinect_shift_up = int(function_input[1])
        kinect_shift_right = int(function_input[2])
        camera_alpha_vert = function_input[3]
        camera_alpha_horiz = function_input[4]
        pressure_horiz_scale = function_input[5]
        pressure_vert_scale = function_input[6]

        #file_dir = "/media/henry/multimodal_data_1/all_hevans_data/0905_2_Evening/0255"
        #file_dir = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"-C_0000"
        file_dir = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-2"

        rtv.load_next_file(file_dir)

        for im_num in range(20, 21):

            self.overall_image_scale_amount = 0.85

            half_w_half_l = [0.4, 0.4, 1.1, 1.1]

            all_image_list = []
            self.label_single_image = []

            self.label_index = 0

            self.color = self.color_all[im_num]
            self.depth_r = self.depth_r_all[im_num]
            self.pressure = self.pressure_all[im_num]
            self.bed_state = self.bedstate_all[im_num]


            self.bed_state[0] = self.bed_state[0]*0.0
            self.bed_state[1] = self.bed_state[1]*0.0
            self.bed_state[2] = self.bed_state[2]*0.0

            markers_c = []
            markers_c.append(self.markers_all[im_num][0])
            markers_c.append(self.markers_all[im_num][1])
            markers_c.append(self.markers_all[im_num][2])
            markers_c.append(self.markers_all[im_num][3])



            # Get the marker points in 2D on the color image
            u_c, v_c = ArTagLib().color_2D_markers(markers_c, self.new_K_kin)

            # Get the marker points dropped to the height of the pressure mat
            u_c_drop, v_c_drop, markers_c_drop = ArTagLib().color_2D_markers_drop(markers_c, self.new_K_kin)

            # Get the geometry for sizing the pressure mat
            print markers_c_drop, self.new_K_kin, self.pressure_im_size_required, self.bed_state, half_w_half_l
            pmat_ArTagLib = ArTagLib()
            self.pressure_im_size_required, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend, half_w_half_l = \
                pmat_ArTagLib.p_mat_geom(markers_c_drop, self.new_K_kin, self.pressure_im_size_required, self.bed_state, half_w_half_l)

            tf_corners = np.zeros((8, 2))
            tf_corners[0:8,:] = np.copy(calib_corners_selection[calib_index])


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
                                                                             np.copy(calib_corners_selection[calib_index][0:4][:]))

            all_image_list.append(color_reshaped)


            #DEPTH
            if self.depth_r is not 0 and DEPTH is True:
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
            if self.pressure is not 0:
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
            self.cursor_shift = self.all_images.shape[1]/3


            self.all_images_clone = self.all_images.copy()


            cv2.imshow('all_images', self.all_images)
            k = cv2.waitKey(1)
            cv2.waitKey(0)

if __name__ == '__main__':

    rospy.init_node('depth_cam_node')

    rtv = RealTimeViz()

    PARTICIPANT = "S151"

    file_dir_list = [["/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT,
                      "/data_"+PARTICIPANT+"-C_0000", 0]]
    #file_dir_list = [["/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT,
    #                  "/data_checked_"+PARTICIPANT+"-2", 0]]

    #file_optimization_picks = [[-0.08444541039339573, 0.016119578206939777, -0.3216531438714622,
    #                            0.1394849993002632, -0.2842603728947779, 0.13685526544739327,
    #                            0.03574465742073685]]
    #file_optimization_picks = [[-0.0734413,  -6.34291415, -0.13737343,  0.08815166, -0.23490408,  0.36510975, 0.90384743]]
    #file_optimization_picks = [[-0.20560566, -4.29998152, -1.33375077,  0.19550086, -0.01974559, 0.47828653,  0.74054469]]
    #file_optimization_picks = [[ 0.04137659, -5.39991712, -1.82485526,  0.13043731,  0.12628347,  0.82542503, 0.91061574]]
    file_optimization_picks = [[ 0, 0, 0, 0, 0, 0, 0]]


    calib_corners_selection = [[[75.20564042303172, 131.60987074030552], [324.3243243243243, 131.60987074030552],
                                [85.78143360752057, 740.3055229142186], [330.19976498237367, 730.9048178613397],
                                [92.8319623971798, 137.48531139835487], [338.4253819036428, 137.48531139835487],
                                [94.00705052878966, 726.2044653349002], [338.4253819036428, 728.5546415981199]]]


    pmat_head_corner_shift_list = [[0.0, 0.0],
                                  [0.0, 0.0],
                                  [0.0, 0.0]]

    p_mat_offset_leg_rest_list = [0, 0, 0]

    calib_index = 0

    OPTIM = True
    if OPTIM == False:
        #file_dir = "/home/henry/test/calib_data_082018_0096"

        # file_dir = "/media/henry/multimodal_data_1/all_hevans_data/Calibration_0907_Evening_0908_Morning/data_082018_0096"
        # 96 - 85 is a good pick for seated
        # 98 - 40 is a good pick for flat


        posture = "lay"
        if posture == "lay":
            folder_idx = 1
            image_idx = 2

        F_eval = rtv.evaluate_data(file_optimization_picks[calib_index])

    else:
        #function_input = 11*[0.0]
        #F_eval = rtv.optim_calib_criteria(function_input)
        ##print F_eval, "F eval"
        #F_eval = rtv.optim_calib_criteria(function_input)
        #print F_eval, "F eval"
        #F_eval = rtv.optim_calib_criteria(function_input)
        #print F_eval, "F eval"
        #F_eval = rtv.optim_calib_criteria(function_input)
        #print F_eval, "F eval"

        import cma

        opts = cma.CMAOptions()
        opts['seed'] = 235  # 456
        opts['ftarget'] = -1
        opts['popsize'] = 50
        opts['maxiter'] = 3000
        opts['maxfevals'] = 1e8
        opts['CMA_cmean'] = 0.25
        opts['tolfun'] = 1e-3
        opts['tolfunhist'] = 1e-12
        opts['tolx'] = 5e-4
        opts['maxstd'] = 4.0
        opts['tolstagnation'] = 100
        #es = cma.CMAEvolutionStrategy(7 * [0.0], 0.0000001)
        es = cma.CMAEvolutionStrategy(file_optimization_picks, 0.2)
        # optim_param_list = 19 * [0]
        es.optimize(rtv.optim_calib_criteria)

        #[0.50115223 - 5.6193305   1.44500491  0.5604416 - 4.48488604 - 2.48505691
        # 0.97121924  0.89736777  2.99696483]