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

def sortSecond(val):
    return val[0:2]

if __name__ == '__main__':

    participant_list = [#"P106",
                        #"P136",

                        "S103",
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

    all_participant_info = {}
    #all_participant_info["P106_"]

    import os
    import re

    for participant_number in participant_list[0:]:
        prescribed_pose_full_list = os.listdir("/media/henry/multimodal_data_2/CVPR2020_study/"+participant_number+"/nominal_poses")
        prescribed_pose_full_list = sorted(prescribed_pose_full_list, key=lambda prescribed_pose_full_list: int(prescribed_pose_full_list[0:2]))
        type_only = []
        for prescribed_pose_full in prescribed_pose_full_list:
            prescribed_pose_full = prescribed_pose_full[5:]
            m = re.search(r"\d", prescribed_pose_full)
            end_str = m.start()


            type_only.append(prescribed_pose_full[0:end_str-1])

        #print type_only
        list_ct = [0, 0, 0, 0, 0, 0, 0, 0]
        for item in type_only:
            if item == 'rollpi': list_ct[0] += 1
            elif item == 'rollpi_plo': list_ct[1] += 1
            elif item == 'supine': list_ct[2] += 1
            elif item == 'supine_plo': list_ct[3] += 1
            elif item == 'hbh': list_ct[4] += 1
            elif item == 'phu': list_ct[5] += 1
            elif item == 'sl': list_ct[6] += 1
            elif item == 'xl': list_ct[7] += 1
        #print list_ct, participant_number

        #if participant_number == "S163": print type_only

        entry = participant_number+"_pose_type"
        print entry
        all_participant_info[entry] = type_only

        #break

    all_participant_info["P106_cal_func"] = [-0.2651102, -4.88367133, -1.06805906, 0.01747648, -0.33419488, 0.57763764, 0.49226572]
    all_participant_info["P136_cal_func"] = [-0.06452759, -3.77696914, -0.32980742, -0.06308034, -0.16521433, 0.26381002, 0.50454449]

    all_participant_info["S103_cal_func"] = [-0.10907472, -6.11589825, -0.8472371,  0.04121596, -0.423374,   0.52237392,   0.83548438]
    all_participant_info["S104_cal_func"] = [-0.11990132, -4.35882353, -1.46408096,  0.30141501, -0.07335344,  0.52156819, 0.66163672]
    all_participant_info["S107_cal_func"] = [-0.17825959, -4.3641031,  -1.55253708,  0.24462174,  0.11118524,  0.49288832, 0.73780083]
    all_participant_info["S114_cal_func"] = [-0.19358307, -4.78395817, -1.04920762,  0.34339822, -0.26543413,  0.52102034, 0.68994741]
    all_participant_info["S118_cal_func"] = [ 0.02556939, -5.58683842, -0.67511668,  0.22057323, -0.20244089,  0.23653652, 0.78438752]
    all_participant_info["S121_cal_func"] = [-0.07715107, -5.76113304,  0.04654912,  0.0791208 , -0.16071217,  0.39971254, 0.81864972]
    all_participant_info["S130_cal_func"] = [-0.09772414, -3.17238773,  0.6734164,  -0.04743994,  0.51325556,  0.21297593, 0.8021125 ]
    all_participant_info["S134_cal_func"] = [-0.14373483, -5.58845298, -0.34678163, -0.05787642, -0.29117663,  0.46273827, 0.82665275]
    all_participant_info["S140_cal_func"] = [-0.11456735, -5.9978343,  -0.11636638, -0.02846276, -0.18410768,  0.37660601, 0.91588008]
    all_participant_info["S141_cal_func"] = [-0.19529118, -5.90449592, -0.51361719, -0.10806887, -0.46784223,  0.26836483, 0.70037546]

    all_participant_info["S145_cal_func"] = [-0.14032882, -5.13431041, -1.43073795,  0.15330748, -0.26953927, 0.53783705,   0.6659938]
    all_participant_info["S151_cal_func"] = [-0.20560566, -4.29998152, -1.33375077,  0.19550086, -0.01974559, 0.47828653,  0.74054469]
    all_participant_info["S163_cal_func"] = [-0.12348539, -4.67494759, -0.80566931,  0.13654094,  0.06810821,  0.48612277, 0.86403426]
    all_participant_info["S165_cal_func"] = [-0.1515425,  -5.49957539, -0.39966683,  0.08126986, -0.33688394,  0.20450442, 0.79982267]
    all_participant_info["S170_cal_func"] = [-0.09337398, -5.42101035, -1.20956034,  0.23614273, -0.38397128,  0.62846808, 0.72308932]
    all_participant_info["S179_cal_func"] = [-0.09248161, -5.48091354, -0.97657492,  0.04471973, -0.25751861,  0.56196299, 0.83406462]
    all_participant_info["S184_cal_func"] = [-0.13151854, -5.04392044, -0.25583815,  0.07010212, -0.15954844,  0.34441231, 0.69325676]
    all_participant_info["S187_cal_func"] = [ 0.04137659, -5.39991712, -1.82485526,  0.13043731,  0.12628347,  0.82542503, 0.91061574]
    all_participant_info["S188_cal_func"] = [-0.0734413,  -6.34291415, -0.13737343,  0.08815166, -0.23490408,  0.36510975, 0.90384743]
    all_participant_info["S196_cal_func"] = [ 0.009228,   -6.05030867,  0.50362194, -0.0522779, -0.56428847,  0.24056171,  0.69630618]

    all_participant_info["P106_corners"] = [[70.50528789659225, 137.48531139835487], [325.4994124559342, 131.60987074030552],
                                            [89.30669800235017, 726.2044653349002], [333.72502937720327, 716.8037602820211],
                                            [92.8319623971798, 139.83548766157463], [338.4253819036428, 141.0105757931845],
                                            [91.65687426556993, 728.5546415981199], [338.4253819036428, 728.5546415981199]]
    all_participant_info["P136_corners"] = [[83.43125734430082, 135.13513513513513], [331.37485311398353, 135.13513513513513],
                                            [79.90599294947121, 728.5546415981199], [319.62397179788485, 728.5546415981199],
                                            [92.8319623971798, 135.13513513513513], [341.9506462984724, 135.13513513513513],
                                            [92.8319623971798, 729.7297297297298], [339.60047003525267, 730.9048178613397]]


    all_participant_info["S103_corners"] = [[79.90599294947121, 125.73443008225617], [336.07520564042306, 130.43478260869566],
                                            [88.1316098707403, 735.6051703877791], [329.0246768507638, 736.780258519389],
                                            [94.00705052878966, 138.66039952996476], [339.60047003525267, 141.0105757931845],
                                            [95.18213866039953, 726.2044653349002], [339.60047003525267, 727.3795534665101]]
    all_participant_info["S104_corners"] = [[82.25616921269095, 129.2596944770858], [332.5499412455934, 136.310223266745],
                                            [84.6063454759107, 736.780258519389], [330.19976498237367, 727.3795534665101],
                                            [94.00705052878966, 139.83548766157463], [338.4253819036428, 141.0105757931845],
                                            [95.18213866039953, 728.5546415981199], [338.4253819036428, 727.3795534665101]]
    all_participant_info["S107_corners"] = [[78.73090481786134, 128.08460634547592], [325.4994124559342, 130.43478260869566],
                                            [82.25616921269095, 737.9553466509989], [331.37485311398353, 728.5546415981199],
                                            [94.00705052878966, 139.83548766157463], [339.60047003525267, 139.83548766157463],
                                            [94.00705052878966, 728.5546415981199], [338.4253819036428, 728.5546415981199]]
    all_participant_info["S114_corners"] = [[81.08108108108108, 129.2596944770858], [334.9001175088132, 135.13513513513513],
                                            [92.8319623971798, 741.4806110458285], [338.4253819036428, 732.0799059929495],
                                            [94.00705052878966, 137.48531139835487], [339.60047003525267, 136.310223266745],
                                            [92.8319623971798, 727.3795534665101], [339.60047003525267, 730.9048178613397]]
    all_participant_info["S118_corners"] = [[90.48178613396004, 118.68390129259694], [337.25029377203293, 130.96004700352526],
                                            [78.73090481786134, 733.2549941245594], [318.448883666275, 736.2549941245594],
                                            [94.00705052878966, 138.66039952996476], [340.77555816686254, 137.48531139835487],
                                            [94.00705052878966, 728.5546415981199], [339.60047003525267, 730.9048178613397]]
    all_participant_info["S121_corners"] = [[92.8319623971798, 122.20916568742656], [341.9506462984724, 125.73443008225617],
                                            [90.48178613396004, 734.4300822561693], [333.72502937720327, 733.2549941245594],
                                            [94.00705052878966, 138.66039952996476], [340.77555816686254, 138.66039952996476],
                                            [94.00705052878966, 729.7297297297298], [339.60047003525267, 729.7297297297298]]
    all_participant_info["S130_corners"] = [[85.78143360752057, 129.2596944770858], [334.9001175088132, 128.08460634547592],
                                            [88.1316098707403, 734.4300822561693], [332.5499412455934, 735.6051703877791],
                                            [94.00705052878966, 141.0105757931845], [338.4253819036428, 141.0105757931845],
                                            [96.3572267920094, 727.3795534665101], [337.25029377203293, 726.2044653349002]]
    all_participant_info["S134_corners"] = [[79.90599294947121, 130.43478260869566], [331.37485311398353, 126.90951821386604],
                                            [86.95652173913044, 739.1304347826087], [329.0246768507638, 740.3055229142186],
                                            [91.65687426556993, 137.48531139835487], [337.25029377203293, 137.48531139835487],
                                            [91.65687426556993, 728.5546415981199], [337.25029377203293, 730.9048178613397]]
    all_participant_info["S140_corners"] = [[84.6063454759107, 125.73443008225617], [332.5499412455934, 126.2596944770858],
                                            [88.1316098707403, 737.9553466509989], [327.8495887191539, 735.6051703877791],
                                            [92.8319623971798, 139.83548766157463], [338.4253819036428, 141.0105757931845],
                                            [94.18213866039953, 727.3795534665101], [338.4253819036428, 729.7297297297298]]
    all_participant_info["S141_corners"] = [[70.1551116333725, 128.08460634547592], [321.448883666275, 121.0340775558167],
                                            [85.78143360752057, 741.4806110458285], [320.7990599294947, 729.7297297297298],
                                            [92.65687426556993, 135.13513513513513], [339.60047003525267, 135.13513513513513],
                                            [94.00705052878966, 729.7297297297298], [339.60047003525267, 730.9048178613397]]

    all_participant_info["S145_corners"] = [[75.20564042303172, 126.90951821386604], [329.0246768507638, 130.43478260869566],
                                            [82.25616921269095, 736.780258519389], [326.67450058754406, 729.7297297297298],
                                            [92.8319623971798,  137.48531139835487], [338.4253819036428, 137.48531139835487],
                                            [92.8319623971798, 732.0799059929495], [339.60047003525267, 732.0799059929495]]
    all_participant_info["S151_corners"] = [[75.20564042303172, 131.60987074030552], [324.3243243243243, 131.60987074030552],
                                            [85.78143360752057, 740.3055229142186], [330.19976498237367, 730.9048178613397],
                                            [92.8319623971798, 137.48531139835487], [338.4253819036428, 137.48531139835487],
                                            [94.00705052878966, 726.2044653349002], [338.4253819036428, 728.5546415981199]]
    all_participant_info["S163_corners"] = [[85.78143360752057, 126.90951821386604], [332.5499412455934, 128.08460634547592],
                                            [85.78143360752057, 740.3055229142186], [331.37485311398353, 736.780258519389],
                                            [94.00705052878966, 137.48531139835487], [339.60047003525267, 136.310223266745],
                                            [95.18213866039953, 726.2044653349002], [338.4253819036428, 728.5546415981199]]
    all_participant_info["S165_corners"] = [[78.73090481786134, 128.08460634547592], [326.67450058754406, 131.60987074030552],
                                            [89.30669800235017, 740.3055229142186], [323.14923619271445, 733.2549941245594],
                                            [92.8319623971798, 136.310223266745], [339.60047003525267, 138.66039952996476],
                                            [94.00705052878966, 727.3795534665101], [339.60047003525267, 727.3795534665101]]
    all_participant_info["S170_corners"] = [[82.25616921269095, 125.73443008225617], [338.4253819036428, 135.13513513513513],
                                            [88.1316098707403, 737.9553466509989], [332.5499412455934, 734.9048178613397],
                                            [94.00705052878966, 138.66039952996476], [337.25029377203293, 138.66039952996476],
                                            [95.18213866039953, 728.5546415981199], [337.25029377203293, 728.5546415981199]]
    all_participant_info["S179_corners"] = [[82.25616921269095, 125.73443008225617], [334.9001175088132, 129.2596944770858],
                                            [83.43125734430082, 740.3055229142186], [326.67450058754406, 738.6051703877791],
                                            [94.00705052878966, 136.310223266745], [337.9506462984724, 137.48531139835487],
                                            [94.00705052878966, 727.3795534665101], [337.25029377203293, 728.5546415981199]]
    all_participant_info["S184_corners"] = [[84.6063454759107, 129.2596944770858], [332.5499412455934, 130.43478260869566],
                                            [88.1316098707403, 733.2549941245594], [331.37485311398353, 730.9048178613397],
                                            [92.8319623971798, 139.83548766157463], [339.60047003525267, 139.83548766157463],
                                            [92.8319623971798, 729.7297297297298], [339.60047003525267, 730.9048178613397]]
    all_participant_info["S187_corners"] = [[89.30669800235017, 118.68390129259694], [343.1257344300823, 128.08460634547592],
                                            [69.33019976498238, 730.9048178613397], [325.4994124559342, 733.2549941245594],
                                            [94.00705052878966, 141.0105757931845], [338.4253819036428, 142.18566392479437],
                                            [94.00705052878966, 726.2044653349002], [338.4253819036428, 727.3795534665101]]
    all_participant_info["S188_corners"] = [[88.1316098707403, 122.20916568742656], [336.07520564042306, 125.73443008225617],
                                            [89.30669800235017, 734.4300822561693], [327.8495887191539, 735.6051703877791],
                                            [92.8319623971798, 142.18566392479437], [337.25029377203293, 141.0105757931845],
                                            [95.18213866039953, 726.2044653349002], [337.25029377203293, 728.5546415981199]]
    all_participant_info["S196_corners"] = [[90.48178613396004, 126.90951821386604], [340.77555816686254, 135.13513513513513],
                                            [77.55581668625148, 732.0799059929495], [321.9741480611046, 734.9048178613397],
                                            [95.18213866039953, 139.83548766157463], [336.77555816686254, 142.18566392479437],
                                            [95.18213866039953, 726.2044653349002], [339.60047003525267, 727.3795534665101]]

    all_participant_info["P106_adj_2"] = []
    all_participant_info["P136_adj_2"] = [-0.0286, -0.0286]

    all_participant_info["S103_adj_2"] = [0.0, 0.0]
    all_participant_info["S104_adj_2"] = [0.0, 0.0143]
    all_participant_info["S107_adj_2"] = [0.0, 0.0]
    all_participant_info["S114_adj_2"] = [-0.0572, 0.0429]
    all_participant_info["S118_adj_2"] = [-0.0286, 0.0]
    all_participant_info["S121_adj_2"] = [0.0, -0.0143]
    all_participant_info["S130_adj_2"] = [-0.0143, 0.0]
    all_participant_info["S134_adj_2"] = [-0.0429, 0.0]
    all_participant_info["S140_adj_2"] = [-0.0286, -0.0143]
    all_participant_info["S141_adj_2"] = [-0.0429, 0.0]

    all_participant_info["S145_adj_2"] = [-0.0429, 0.0]
    all_participant_info["S151_adj_2"] = [-0.0429, 0.0]
    all_participant_info["S163_adj_2"] = [-0.0286, -0.0]
    all_participant_info["S165_adj_2"] = [-0.0429, 0.0]
    all_participant_info["S170_adj_2"] = [-0.0429, 0.0]
    all_participant_info["S179_adj_2"] = [-0.0286, 0.0]
    all_participant_info["S184_adj_2"] = [-0.0286, 0.0]
    all_participant_info["S187_adj_2"] = [0.0, -0.0286]
    all_participant_info["S188_adj_2"] = [0.0, 0.0]
    all_participant_info["S196_adj_2"] = [-0.0, -0.0429]

    all_participant_info["P106_weight_lbs"] = 165.
    all_participant_info["P136_weight_lbs"] = 143.

    all_participant_info["S103_weight_lbs"] = 186.
    all_participant_info["S104_weight_lbs"] = 93.
    all_participant_info["S107_weight_lbs"] = 164.
    all_participant_info["S114_weight_lbs"] = 140.
    all_participant_info["S118_weight_lbs"] = 135. #61 kgs
    all_participant_info["S121_weight_lbs"] = 185.
    all_participant_info["S130_weight_lbs"] = 120.
    all_participant_info["S134_weight_lbs"] = 121.
    all_participant_info["S140_weight_lbs"] = 160.
    all_participant_info["S141_weight_lbs"] = 130.

    all_participant_info["S145_weight_lbs"] = 160.
    all_participant_info["S151_weight_lbs"] = 140.
    all_participant_info["S163_weight_lbs"] = 120.
    all_participant_info["S165_weight_lbs"] = 205.
    all_participant_info["S170_weight_lbs"] = 159. #72 kgs
    all_participant_info["S179_weight_lbs"] = 143.
    all_participant_info["S184_weight_lbs"] = 147.
    all_participant_info["S187_weight_lbs"] = 120.
    all_participant_info["S188_weight_lbs"] = 225.
    all_participant_info["S196_weight_lbs"] = 120.


    all_participant_info["P106_height_in"] = 68.
    all_participant_info["P136_height_in"] = 60.

    all_participant_info["S103_height_in"] = 68.
    all_participant_info["S104_height_in"] = 60.
    all_participant_info["S107_height_in"] = 66.
    all_participant_info["S114_height_in"] = 68.5
    all_participant_info["S118_height_in"] = 64. #
    all_participant_info["S121_height_in"] = 73.
    all_participant_info["S130_height_in"] = 65.
    all_participant_info["S134_height_in"] = 64.5
    all_participant_info["S140_height_in"] = 62.
    all_participant_info["S141_height_in"] = 67.

    all_participant_info["S145_height_in"] = 64.
    all_participant_info["S151_height_in"] = 67.
    all_participant_info["S163_height_in"] = 70.
    all_participant_info["S165_height_in"] = 78.
    all_participant_info["S170_height_in"] = 70. #
    all_participant_info["S179_height_in"] = 68.
    all_participant_info["S184_height_in"] = 72.
    all_participant_info["S187_height_in"] = 63.
    all_participant_info["S188_height_in"] = 78.
    all_participant_info["S196_height_in"] = 68.

    all_participant_info["P106_gender"] = 'f'
    all_participant_info["P136_gender"] = 'f'

    all_participant_info["S103_gender"] = 'f'
    all_participant_info["S104_gender"] = 'f'
    all_participant_info["S107_gender"] = 'f'
    all_participant_info["S114_gender"] = 'm'
    all_participant_info["S118_gender"] = 'f'
    all_participant_info["S121_gender"] = 'm'
    all_participant_info["S130_gender"] = 'f'
    all_participant_info["S134_gender"] = 'f'
    all_participant_info["S140_gender"] = 'f'
    all_participant_info["S141_gender"] = 'm'

    all_participant_info["S145_gender"] = 'm'
    all_participant_info["S151_gender"] = 'f'
    all_participant_info["S163_gender"] = 'm'
    all_participant_info["S165_gender"] = 'm'
    all_participant_info["S170_gender"] = 'm'
    all_participant_info["S179_gender"] = 'm'
    all_participant_info["S184_gender"] = 'm'
    all_participant_info["S187_gender"] = 'f'
    all_participant_info["S188_gender"] = 'm'
    all_participant_info["S196_gender"] = 'f'



    for idx in range(len(participant_list)):
        participant = participant_list[idx]

        file_dir = "/media/henry/multimodal_data_2/CVPR2020_study/"+participant

        participant_info = {}
        #participant_info["cal_func"] = all_participant_info[participant + "_cal_func"]
        #participant_info["adj_2"] = all_participant_info[participant + "_adj_2"]
        #participant_info["corners"] = all_participant_info[participant + "_corners"]
        participant_info["weight_lbs"] = all_participant_info[participant + "_weight_lbs"]
        participant_info["height_in"] = all_participant_info[participant + "_height_in"]
        participant_info["gender"] = all_participant_info[participant + "_gender"]
        #participant_info["pose_type"] = all_participant_info[participant + "_pose_type"]

        pkl.dump(participant_info, open(file_dir+'/participant_info_red.p', 'wb'))

