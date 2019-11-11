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



# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    RESULT_TYPE = "real"

    if RESULT_TYPE == "real":

        participant_list = ["S103",
                            "S104",
                            "S107",
                            "S114",
                            "S118",
                            "S121",
                            "S130",
                            "S134",
                            "S140",
                            "S141",

                            "S145",
                            "S151",
                            "S163",
                            "S165", #at least 3 pc corrupted
                            "S170",
                            "S179",
                            "S184",
                            "S187",
                            "S188", #1 bad prone posture classified as supine, 2 pc corrupted
                            "S196",]

        NETWORK_2 = "0.5rtojtdpth_depthestin_angleadj_tnh_htwt_calnoise"

        DATA_TYPE = "2"


        recall_avg_list = []
        precision_avg_list = []
        overlap_d_err_avg_list = []
        v_to_gt_err_avg_list = []
        gt_to_v_err_avg_list = []

        for participant in participant_list:
            current_results_dict = load_pickle("/media/henry/multimodal_data_2/data/final_results/"+NETWORK_2+"/results_real_"
                                               +participant+"_"+DATA_TYPE+"_"+NETWORK_2+".p")
            #for entry in current_results_dict:
            #    print entry

            #precision =

            #to test posture
            body_roll_rad = current_results_dict['body_roll_rad']


            recall = current_results_dict['recall']
            curr_recall = np.mean(recall)
            recall_avg_list.append(curr_recall)

            precision = current_results_dict['precision']
            curr_precision = np.mean(precision)
            precision_avg_list.append(curr_precision)


            overlap_d_err = current_results_dict['overlap_d_err']
            curr_overlap_d_err = np.mean(overlap_d_err)
            #print curr_overlap_d_err, 'overlap d err'
            overlap_d_err_avg_list.append(curr_overlap_d_err)


            v_limb_to_gt_err = current_results_dict['v_limb_to_gt_err']

            v_to_gt_err = current_results_dict['v_to_gt_err']
            curr_v_to_gt_err = np.mean(v_to_gt_err)
            #print curr_v_to_gt_err, 'nearest to gt'
            v_to_gt_err_avg_list.append(curr_v_to_gt_err)

            gt_to_v_err = current_results_dict['gt_to_v_err']
            curr_gt_to_v_err = np.mean(gt_to_v_err)
            gt_to_v_err_avg_list.append(curr_gt_to_v_err)
            #print  curr_gt_to_v_err

           # break

        print "average recall: ", np.mean(recall_avg_list)
        print "average precision: ", np.mean(precision_avg_list)
        print "average overlap depth err: ", np.mean(overlap_d_err_avg_list)
        print "average v to gt err: ", np.mean(v_to_gt_err_avg_list)
        print "average gt to v err: ", np.mean(gt_to_v_err_avg_list)
