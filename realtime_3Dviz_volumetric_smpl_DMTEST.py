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
from tensorprep_lib import TensorPrepLib
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

WEIGHT_LBS = 70.
HEIGHT_IN = 45.
GENDER = 'm'
import rospy
from hrl_msgs.msg import MeshAttr



class GeneratePose():
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
        rospy.Subscriber("/multi_pose/ar_pose_marker", AlvarMarkers, self.callback_bed_tags)
        rospy.Subscriber("/multi_pose/kinect2/qhd/points", PointCloud2, self.callback_points)
        rospy.Subscriber("/multi_pose/fsascan", FloatArrayBare, self.callback_pressure)

        rospy.Subscriber("/abdout0", FloatArrayBare, self.bed_config_callback)
        #rospy.Subscriber("/abdout0", FloatArrayBare, self.callback_bed_state)
        print "init subscriber"

        rospy.init_node('vol_3d_listener', anonymous=False)
        self.point_cloud_array = np.array([[0., 0., 0.]])
        self.pc_isnew = False


        self.CTRL_PNL = {}
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['loss_vector_type'] = 'anglesDC' #'anglesEU'
        self.CTRL_PNL['loss_vector_type'] = 'anglesDC' #'anglesEU'
        self.CTRL_PNL['verbose'] = False
        self.CTRL_PNL['num_epochs'] = 101
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = True
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['num_input_channels'] = 3
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['GPU'] = True
        self.CTRL_PNL['dtype'] = torch.cuda.FloatTensor
        self.CTRL_PNL['repeat_real_data_ct'] = 1
        self.CTRL_PNL['regr_angles'] = 1
        self.CTRL_PNL['dropout'] = False
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

        self.bedangle = bedangle + 5.


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


    def generate_prechecked_pose(self, posture, stiffness, filename):


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

    def estimate_real_time(self, filename1, filename2 = None):




        pyRender = libRender.pyRenderMesh()
        mat_size = (64, 27)
        from unpack_batch_lib import UnpackBatchLib

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
            for i in range(0, 8):
                try:
                    model = torch.load(filename1, map_location={'cuda:'+str(i):'cuda:0'})
                    model = model.cuda().eval()
                    break
                except:
                    pass
            if filename2 is not None:
                for i in range(0, 8):
                    try:
                        model2 = torch.load(filename2, map_location={'cuda:'+str(i):'cuda:0'})
                        model2 = model2.cuda().eval()
                        break
                    except:
                        pass
            else:
                model2 = None

        else:
            model = torch.load(filename1, map_location='cpu').eval()
            if filename2 is not None:
                model2 = torch.load(filename2, map_location='cpu').eval()
            else:
                model2 = None


        pub = rospy.Publisher('meshTopic', MeshAttr)
        #rospy.init_node('talker', anonymous=False)
        while not rospy.is_shutdown():


            pmat = np.fliplr(np.flipud(np.clip(self.pressure.reshape(mat_size)*float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100)))

            if self.CTRL_PNL['cal_noise'] == False:
                pmat = gaussian_filter(pmat, sigma= 0.5)


            pmat_stack = PreprocessingLib().preprocessing_create_pressure_angle_stack_realtime(pmat, self.bedangle, mat_size)

            if self.CTRL_PNL['cal_noise'] == False:
                pmat_stack = np.clip(pmat_stack, a_min=0, a_max=100)

            pmat_stack = np.array(pmat_stack)
            if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
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

            if self.CTRL_PNL['normalize_input'] == True:
                self.CTRL_PNL['depth_map_input_est'] = False
                pmat_stack = self.TPL.normalize_network_input(pmat_stack, self.CTRL_PNL)
                batch1 = self.TPL.normalize_wt_ht(batch1, self.CTRL_PNL)


            pmat_stack = torch.Tensor(pmat_stack)
            batch1 = torch.Tensor(batch1)


            batch = []
            batch.append(pmat_stack)
            batch.append(batch1)

            self.CTRL_PNL['adjust_ang_from_est'] = False
            scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpackage_batch_kin_pass(batch, False, model, self.CTRL_PNL)


            mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1) / 16.69545796387731
            mdm_est_neg = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1) / 45.08513083167194
            mdm_est_pos[mdm_est_pos < 0] = 0
            mdm_est_neg[mdm_est_neg > 0] = 0
            mdm_est_neg *= -1
            cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100 / 43.55800622930469

            #1. / 16.69545796387731,  # pos est depth
            #1. / 45.08513083167194,  # neg est depth
            #1. / 43.55800622930469,  # cm est

            if model2 is not None:
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
                                      OUTPUT_DICT['batch_root_xyz_est'].cpu()), dim = 1))
                elif self.CTRL_PNL['full_body_rot'] == True:
                    batch_cor.append(torch.cat((batch1,
                                      OUTPUT_DICT['batch_betas_est'].cpu(),
                                      OUTPUT_DICT['batch_angles_est'].cpu(),
                                      OUTPUT_DICT['batch_root_xyz_est'].cpu(),
                                      OUTPUT_DICT['batch_root_atan2_est'].cpu()), dim = 1))


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

            print self.m.r
            #print OUTPUT_DICT['verts']

            pyRender.mesh_render_pose_bed_orig(self.m, root_shift_est, self.point_cloud_array, self.pc_isnew, pmat*5./float(self.CTRL_PNL['pmat_mult']), self.markers, self.bedangle)
            self.point_cloud_array = None

            #dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender = gender, posture = posture, stiffness = stiffness, shiftSIDE = shape_pose_vol[4], shiftUD = shape_pose_vol[5], filepath_prefix=self.filepath_prefix, add_floor = False)

            #dss.run_simulation(10000)
            #generator.standard_render()
            #rate = rospy.Rate(0.5)  # 10hz
            #pub.publish(vertices=np.array(self.m.r), faces=np.array(self.m.f))

            #break

if __name__ ==  "__main__":

    filepath_prefix = "/home/henry"


    generator = GeneratePose(filepath_prefix)
    #generator.estimate_real_time(filepath_prefix+"/data/synth/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_100e_000005lr.pt",
    #                             filepath_prefix+"/data/synth/convnet_anglesEU_synth_s9_3xreal_128b_0.7rtojtdpth_pmatcntin_depthestin_angleadj_100e_000005lr.pt") #init sampling from yash dataset
    #generator.estimate_real_time(filepath_prefix+"/data/convnets/planesreg/DC_L2depth/convnet_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_125e_00001lr.pt",
     #                            filepath_prefix+"/data/convnets/planesreg_correction/DC_L2depth/convnet_anglesDC_synth_32000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt_125e_200e_00001lr.pt")


    #generator.estimate_real_time(filepath_prefix+"/data/convnets/planesreg/DC_L2depth/convnet_anglesDC_synth_32000_128b_x1pmult_0.5rtojtdpth_alltanh_l2cnt_calnoise_300e_00001lr.pt")
    #generator.estimate_real_time(filepath_prefix+"/data/convnets/planesreg/112K/convnet_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_calnoise_50e_00001lr.pt")


    generator.estimate_real_time(filepath_prefix+"/data/convnets/planesreg/112K/convnet_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_alltanh_l2cnt_100e_00001lr.pt",
                                 filepath_prefix+"/data/convnets/planesreg_correction/112K/convnet_anglesDC_synth_112000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_alltanh_l2cnt_100e_200e_00001lr.pt")
    #generator.estimate_real_time(filepath_prefix+"/data/convnets/planesreg/184K/convnet_anglesDC_synth_184K_128b_x5pmult_0.5rtojtdpth_tnh_calnoise_50e_00002lr.pt")

    #generator.estimate_real_time(filepath_prefix+"/data/convnets/planesreg/convnet_anglesDC_synth_32000_128b_x1pmult_0.5rtojtdpth_l2cnt_calnoise_150e_00001lr.pt")
    #generator.estimate_real_time(filepath_prefix+"/data/convnets/planesreg/convnet_anglesDC_synth_32000_128b_201e_0.5rtojtdpth_pmatcntin_100e_00001lr.pt",
    #                             filepath_prefix+"/data/convnets/planesreg_correction/convnet_anglesDC_synth_32000_128b_201e_0.5rtojtdpth_pmatcntin_depthestin_angleadj_50e_00001lr.pt")
