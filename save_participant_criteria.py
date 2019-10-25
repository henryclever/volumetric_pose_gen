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


if __name__ == '__main__':

    participant_list = ["P106",
                        "P136",
                        "S145",
                        "S151",
                        "S141",
                        "S184",
                        "S104",]

    all_participant_info = {}
    all_participant_info["P106_cal_func"] = [-0.2651102, -4.88367133, -1.06805906, 0.01747648, -0.33419488, 0.57763764, 0.49226572]
    all_participant_info["P136_cal_func"] = [-0.06452759, -3.77696914, -0.32980742, -0.06308034, -0.16521433, 0.26381002, 0.50454449]
    all_participant_info["S145_cal_func"] = [-0.14032882, -5.13431041, -1.43073795,  0.15330748, -0.26953927, 0.53783705,  0.6659938]
    all_participant_info["S151_cal_func"] = [-0.20560566, -4.29998152, -1.33375077, 0.19550086, -0.01974559, 0.47828653, 0.74054469]
    all_participant_info["S141_cal_func"] = [-0.23182547, -6.09160234, -0.59679228, -0.16791456, -0.44618475, 0.30850779, 0.73350682]
    all_participant_info["S184_cal_func"] = [-0.13151854, -5.04392044, -0.25583815,  0.07010212, -0.15954844, 0.34441231, 0.69325676]
    all_participant_info["S104_cal_func"] = [-0.12315983, -4.59910646, -0.67867036,  0.13046543, -0.0162565, 0.42552982,  0.70797418]

    all_participant_info["P106_corners"] = [[70.50528789659225, 137.48531139835487], [325.4994124559342, 131.60987074030552],
                                            [89.30669800235017, 726.2044653349002], [333.72502937720327, 716.8037602820211],
                                            [92.8319623971798, 139.83548766157463], [338.4253819036428, 141.0105757931845],
                                            [91.65687426556993, 728.5546415981199], [338.4253819036428, 728.5546415981199]]
    all_participant_info["P136_corners"] = [[83.43125734430082, 135.13513513513513], [331.37485311398353, 135.13513513513513],
                                            [79.90599294947121, 728.5546415981199], [319.62397179788485, 728.5546415981199],
                                            [92.8319623971798, 135.13513513513513], [341.9506462984724, 135.13513513513513],
                                            [92.8319623971798, 729.7297297297298], [339.60047003525267, 730.9048178613397]]
    all_participant_info["S145_corners"] = [[75.20564042303172, 126.90951821386604], [329.0246768507638, 130.43478260869566],
                                            [82.25616921269095, 736.780258519389], [326.67450058754406, 729.7297297297298],
                                            [92.8319623971798,  137.48531139835487], [338.4253819036428, 137.48531139835487],
                                            [92.8319623971798, 732.0799059929495], [339.60047003525267, 732.0799059929495]]
    all_participant_info["S151_corners"] = [[75.20564042303172, 131.60987074030552], [324.3243243243243, 131.60987074030552],
                                            [85.78143360752057, 740.3055229142186], [330.19976498237367, 730.9048178613397],
                                            [92.8319623971798, 137.48531139835487], [338.4253819036428, 137.48531139835487],
                                            [94.00705052878966, 726.2044653349002], [338.4253819036428, 728.5546415981199]]
    all_participant_info["S141_corners"] = [[68.1551116333725, 130.43478260869566], [319.62397179788485, 121.0340775558167],
                                            [86.95652173913044, 737.9553466509989], [321.9741480611046, 729.7297297297298],
                                            [92.8319623971798, 136.310223266745], [339.60047003525267, 137.48531139835487],
                                            [94.00705052878966, 730.9048178613397], [338.4253819036428, 732.0799059929495]]
    all_participant_info["S184_corners"] = [[84.6063454759107, 129.2596944770858], [332.5499412455934, 130.43478260869566],
                                            [88.1316098707403, 733.2549941245594], [331.37485311398353, 730.9048178613397],
                                            [92.8319623971798, 139.83548766157463], [339.60047003525267, 139.83548766157463],
                                            [92.8319623971798, 729.7297297297298], [339.60047003525267, 730.9048178613397]]
    all_participant_info["S104_corners"] = [[84.6063454759107, 129.2596944770858], [332.5499412455934, 130.43478260869566],
                                            [88.1316098707403, 733.2549941245594], [331.37485311398353, 730.9048178613397],
                                            [92.8319623971798, 139.83548766157463], [339.60047003525267, 139.83548766157463],
                                            [92.8319623971798, 729.7297297297298], [339.60047003525267, 730.9048178613397]]

    all_participant_info["P106_weight_lbs"] = 165.
    all_participant_info["P136_weight_lbs"] = 143.
    all_participant_info["S145_weight_lbs"] = 160.
    all_participant_info["S151_weight_lbs"] = 140.
    all_participant_info["S141_weight_lbs"] = 130.
    all_participant_info["S184_weight_lbs"] = 147.
    all_participant_info["S104_weight_lbs"] = 93.
    #all_participant_info["S1_weight_lbs"] =

    all_participant_info["P106_height_in"] = 68.
    all_participant_info["P136_height_in"] = 60.
    all_participant_info["S145_height_in"] = 64.
    all_participant_info["S151_height_in"] = 67.
    all_participant_info["S141_height_in"] = 67.
    all_participant_info["S184_height_in"] = 72.
    all_participant_info["S104_height_in"] = 60.
    #all_participant_info["S1_height_in"] =

    all_participant_info["P106_gender"] = 'f'
    all_participant_info["P136_gender"] = 'f'
    all_participant_info["S145_gender"] = 'm'
    all_participant_info["S151_gender"] = 'f'
    all_participant_info["S141_gender"] = 'm'
    all_participant_info["S184_gender"] = 'm'
    all_participant_info["S104_gender"] = 'f'
    #all_participant_info["S1_gender"] =



    for idx in range(len(participant_list)):
        participant = participant_list[idx]

        file_dir = "/media/henry/multimodal_data_1/CVPR2020_study/"+participant

        participant_info = {}
        participant_info["cal_func"] = all_participant_info[participant + "_cal_func"]
        participant_info["corners"] = all_participant_info[participant + "_corners"]
        participant_info["weight_lbs"] = all_participant_info[participant + "_weight_lbs"]
        participant_info["height_in"] = all_participant_info[participant + "_height_in"]
        participant_info["gender"] = all_participant_info[participant + "_gender"]

        pkl.dump(participant_info, open(file_dir+'/participant_info.p', 'wb'))

