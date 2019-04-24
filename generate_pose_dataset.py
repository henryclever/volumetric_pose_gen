import numpy as np
import random
import copy
from opendr.renderer import ColoredRenderer
from opendr.renderer import DepthRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model
from util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose

#volumetric pose gen libraries
import lib_visualization as libVisualization
import lib_kinematics as libKinematics
import lib_render as libRender
from process_yash_data import ProcessYashData
#import dart_skel_sim
from time import sleep

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
    def __init__(self, gender, posture = "lay", filepath_prefix = '/home/henry'):
        ## Load SMPL model (here we load the female model)
        self.filepath_prefix = filepath_prefix

        if gender == "m":
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.reset_pose = False
        self.m = load_model(model_path)

        if posture == "sit":
            filename = filepath_prefix+'/git/volumetric_pose_gen/init_pose_angles/all_sit_angles.p'
        else:
            filename = filepath_prefix+'/git/volumetric_pose_gen/init_pose_angles/all_angles.p'
        with open(filename, 'rb') as fp:
            self.angles_data = pickle.load(fp)
        shuffle(self.angles_data)



    def sample_body_shape(self, sampling, sigma, one_side_range):
        mu = 0
        for i in range(10):
            if sampling == "NORMAL":
                self.m.betas[i] = random.normalvariate(mu, sigma)
            elif sampling == "UNIFORM":
                self.m.betas[i]  = np.random.uniform(-one_side_range, one_side_range)

        #print self.m.pose.shape
        #print self.m.pose, 'pose'
        #print self.m.betas, 'betas'


    def get_noisy_angle(self, angle, angle_min, angle_max):
        not_within_bounds = True
        mu = 0
        counter = 0
        sigma = np.pi/16
        while not_within_bounds == True:

            noisy_angle = angle + random.normalvariate(mu, sigma)
            if noisy_angle > angle_min and noisy_angle < angle_max:
                not_within_bounds = False
            else:
                print noisy_angle, angle_min, angle_max
                counter += 1
                if counter > 10:
                    self.reset_pose = True
                    break
                pass

        return noisy_angle





    def map_shuffled_yash_to_smpl_angles(self, posture, shiftUD,  alter_angles = True):
        angle_type = "angle_axis"
        #angle_type = "euler"
        if alter_angles == True:

            try:
                entry = self.angles_data.pop()
            except:
                print "############################# RESAMPLING !! #################################"
                if posture == "sit":
                    filename = self.filepath_prefix+'/git/volumetric_pose_gen/init_pose_angles/all_sit_angles.p'
                else:
                    filename = self.filepath_prefix+'/git/volumetric_pose_gen/init_pose_angles/all_angles.p'
                with open(filename, 'rb') as fp:
                    self.angles_data = pickle.load(fp)
                shuffle(self.angles_data)
                entry = self.angles_data.pop()


            if posture == "sit":
                if shiftUD >= 0.1:
                    self.m.pose[0] = np.pi/3
                    self.m.pose[9] = 0.0
                    self.m.pose[18] = 0.0
                    self.m.pose[27] = 0.0
                elif 0.0 <= shiftUD < 0.1:
                    self.m.pose[0] = np.pi/6
                    self.m.pose[9] = np.pi/6
                    self.m.pose[18] = 0.0
                    self.m.pose[27] = 0.0
                elif -0.1 <= shiftUD < 0.0:
                    self.m.pose[0] = np.pi/9
                    self.m.pose[9] = np.pi/9
                    self.m.pose[18] = np.pi/9
                    self.m.pose[27] = 0.0
                elif shiftUD < -0.1:
                    self.m.pose[0] = np.pi/12
                    self.m.pose[9] = np.pi/12
                    self.m.pose[18] = np.pi/12
                    self.m.pose[27] = np.pi/12
                self.m.pose[36] = np.pi / 12
                self.m.pose[45] = np.pi / 12

            R_root = libKinematics.eulerAnglesToRotationMatrix([-float(self.m.pose[0])/2, 0.0, 0.0])

            R_l_hip_rod = libKinematics.matrix_from_dir_cos_angles(entry['l_hip_'+angle_type])
            R_r_hip_rod = libKinematics.matrix_from_dir_cos_angles(entry['r_hip_'+angle_type])

            R_l = np.matmul(R_root, R_l_hip_rod)
            R_r = np.matmul(R_root, R_r_hip_rod)

            new_left_hip = libKinematics.dir_cos_angles_from_matrix(R_l)
            new_right_hip = libKinematics.dir_cos_angles_from_matrix(R_r)

            flip_leftright = random.choice([True, False])

            while True:
                self.reset_pose = False
                if flip_leftright == False:

                    self.m.pose[3] = generator.get_noisy_angle(new_left_hip[0], -2.047187297216041, 0.0008725992352640336)
                    self.m.pose[4] = generator.get_noisy_angle(new_left_hip[1], -1.0056561780573234, 0.9792596381050885)
                    self.m.pose[5] = generator.get_noisy_angle(new_left_hip[2], -0.83127128871961, 0.9840833280290882)
                    self.m.pose[12] = generator.get_noisy_angle(entry['l_knee_angle_axis'][0], 0.0,  2.320752282574325)

                    self.m.pose[6] = generator.get_noisy_angle(new_right_hip[0], -2.047187297216041, 0.0008725992352640336)
                    self.m.pose[7] = generator.get_noisy_angle(new_right_hip[1], -0.9792596381050885, 1.0056561780573234)
                    self.m.pose[8] = generator.get_noisy_angle(new_right_hip[2], -0.9840833280290882,  0.83127128871961)
                    self.m.pose[15] = generator.get_noisy_angle(entry['r_knee_angle_axis'][0], 0.0, 2.320752282574325)


                    self.m.pose[39] = generator.get_noisy_angle(entry['l_shoulder_'+angle_type][0]*1/3, -1.8819412381973686 * 1 / 3, 1.5740500958475525 * 1 / 3)
                    self.m.pose[40] = generator.get_noisy_angle(entry['l_shoulder_'+angle_type][1]*1/3, -2.452871806175492 * 1 / 3, 1.6424506514942065 * 1 / 3)
                    self.m.pose[41] = generator.get_noisy_angle(entry['l_shoulder_'+angle_type][2]*1/3, -1.8997886932520462 * 1 / 3, 1.9844820788036448 * 1 / 3)
                    self.m.pose[48] = generator.get_noisy_angle(entry['l_shoulder_'+angle_type][0]*2/3, -1.8819412381973686 * 2 / 3, 1.5740500958475525 * 2 / 3)
                    self.m.pose[49] = generator.get_noisy_angle(entry['l_shoulder_'+angle_type][1]*2/3, -2.452871806175492 * 2 / 3, 1.6424506514942065 * 2 / 3)
                    self.m.pose[50] = generator.get_noisy_angle(entry['l_shoulder_'+angle_type][2]*2/3, -1.8997886932520462 * 2 / 3, 1.9844820788036448 * 2 / 3)
                    self.m.pose[55] = generator.get_noisy_angle(entry['l_elbow_angle_axis'][1], -2.146677709782182, 0.0)

                    self.m.pose[42] = generator.get_noisy_angle(entry['r_shoulder_'+angle_type][0]*1/3, -1.8819412381973686 * 1 / 3, 1.5740500958475525 * 1 / 3)
                    self.m.pose[43] = generator.get_noisy_angle(entry['r_shoulder_'+angle_type][1]*1/3, -1.6424506514942065 * 1 / 3, 2.452871806175492 * 1 / 3)
                    self.m.pose[44] = generator.get_noisy_angle(entry['r_shoulder_'+angle_type][2]*1/3, -1.9844820788036448 * 1 / 3, 1.8997886932520462 * 1 / 3)
                    self.m.pose[51] = generator.get_noisy_angle(entry['r_shoulder_'+angle_type][0]*2/3, -1.8819412381973686 * 2 / 3, 1.5740500958475525 * 2 / 3)
                    self.m.pose[52] = generator.get_noisy_angle(entry['r_shoulder_'+angle_type][1]*2/3, -1.6424506514942065 * 2 / 3, 2.452871806175492 * 2 / 3)
                    self.m.pose[53] = generator.get_noisy_angle(entry['r_shoulder_'+angle_type][2]*2/3, -1.9844820788036448 * 2 / 3, 1.8997886932520462 * 2 / 3)
                    self.m.pose[58] = generator.get_noisy_angle(entry['r_elbow_angle_axis'][1], 0.0, 2.146677709782182)

                elif flip_leftright == True:

                    self.m.pose[3] = generator.get_noisy_angle(new_right_hip[0], -2.047187297216041, 0.0008725992352640336)
                    self.m.pose[4] = generator.get_noisy_angle(-new_right_hip[1], -1.0056561780573234, 0.9792596381050885)
                    self.m.pose[5] = generator.get_noisy_angle(-new_right_hip[2], -0.83127128871961, 0.9840833280290882)
                    self.m.pose[12] = generator.get_noisy_angle(entry['r_knee_angle_axis'][0], 0.0,  2.320752282574325)

                    self.m.pose[6] = generator.get_noisy_angle(new_left_hip[0], -2.047187297216041, 0.0008725992352640336)
                    self.m.pose[7] = generator.get_noisy_angle(-new_left_hip[1], -0.9792596381050885, 1.0056561780573234)
                    self.m.pose[8] = generator.get_noisy_angle(-new_left_hip[2], -0.9840833280290882,  0.83127128871961)
                    self.m.pose[15] = generator.get_noisy_angle(entry['l_knee_angle_axis'][0], 0.0, 2.320752282574325)


                    self.m.pose[39] = generator.get_noisy_angle(entry['r_shoulder_'+angle_type][0]*1/3, -1.8819412381973686 * 1 / 3, 1.5740500958475525 * 1 / 3)
                    self.m.pose[40] = generator.get_noisy_angle(-entry['r_shoulder_'+angle_type][1]*1/3, -2.452871806175492 * 1 / 3, 1.6424506514942065 * 1 / 3)
                    self.m.pose[41] = generator.get_noisy_angle(-entry['r_shoulder_'+angle_type][2]*1/3, -1.8997886932520462 * 1 / 3, 1.9844820788036448 * 1 / 3)
                    self.m.pose[48] = generator.get_noisy_angle(entry['r_shoulder_'+angle_type][0]*2/3, -1.8819412381973686 * 2 / 3, 1.5740500958475525 * 2 / 3)
                    self.m.pose[49] = generator.get_noisy_angle(-entry['r_shoulder_'+angle_type][1]*2/3, -2.452871806175492 * 2 / 3, 1.6424506514942065 * 2 / 3)
                    self.m.pose[50] = generator.get_noisy_angle(-entry['r_shoulder_'+angle_type][2]*2/3, -1.8997886932520462 * 2 / 3, 1.9844820788036448 * 2 / 3)
                    self.m.pose[55] = generator.get_noisy_angle(-entry['r_elbow_angle_axis'][1], -2.146677709782182, 0.0)

                    self.m.pose[42] = generator.get_noisy_angle(entry['l_shoulder_'+angle_type][0]*1/3, -1.8819412381973686 * 1 / 3, 1.5740500958475525 * 1 / 3)
                    self.m.pose[43] = generator.get_noisy_angle(-entry['l_shoulder_'+angle_type][1]*1/3, -1.6424506514942065 * 1 / 3, 2.452871806175492 * 1 / 3)
                    self.m.pose[44] = generator.get_noisy_angle(-entry['l_shoulder_'+angle_type][2]*1/3, -1.9844820788036448 * 1 / 3, 1.8997886932520462 * 1 / 3)
                    self.m.pose[51] = generator.get_noisy_angle(entry['l_shoulder_'+angle_type][0]*2/3, -1.8819412381973686 * 2 / 3, 1.5740500958475525 * 2 / 3)
                    self.m.pose[52] = generator.get_noisy_angle(-entry['l_shoulder_'+angle_type][1]*2/3, -1.6424506514942065 * 2 / 3, 2.452871806175492 * 2 / 3)
                    self.m.pose[53] = generator.get_noisy_angle(-entry['l_shoulder_'+angle_type][2]*2/3, -1.9844820788036448 * 2 / 3, 1.8997886932520462 * 2 / 3)
                    self.m.pose[58] = generator.get_noisy_angle(-entry['l_elbow_angle_axis'][1], 0.0, 2.146677709782182)



                if self.reset_pose == True:
                    pass
                else:
                    break

        from capsule_body import get_capsules, joint2name, rots0
        capsules = get_capsules(self.m)
        joint2name = joint2name
        rots0 = rots0
        print len(capsules)

        return self.m, capsules, joint2name, rots0



    def map_random_selection_to_smpl_angles(self, alter_angles):
        if alter_angles == True:
            selection_r_leg = ProcessYashData().sample_angles('r_leg')
            self.m.pose[6] = selection_r_leg['rG_ext']
            self.m.pose[7] = selection_r_leg['rG_yaw']#/2
            self.m.pose[8] = selection_r_leg['rG_abd']

            self.m.pose[15] = selection_r_leg['rK']
            #self.m.pose[16] = selection_r_leg['rG_yaw']/2

            selection_l_leg = ProcessYashData().sample_angles('l_leg')
            self.m.pose[3] = selection_l_leg['lG_ext']
            self.m.pose[4] = selection_l_leg['lG_yaw']#/2
            self.m.pose[5] = selection_l_leg['lG_abd']

            self.m.pose[12] = selection_l_leg['lK']
            #self.m.pose[13] = selection_l_leg['lG_yaw']/2

            selection_r_arm = ProcessYashData().sample_angles('r_arm')
            self.m.pose[51] = selection_r_arm['rS_roll']*2/3
            self.m.pose[52] = selection_r_arm['rS_yaw']*2/3
            self.m.pose[53] = selection_r_arm['rS_pitch']*2/3
            self.m.pose[42] = selection_r_arm['rS_roll']*1/3
            self.m.pose[43] = selection_r_arm['rS_yaw']*1/3
            self.m.pose[44] = selection_r_arm['rS_pitch']*1/3

            self.m.pose[58] = selection_r_arm['rE']

            selection_l_arm = ProcessYashData().sample_angles('l_arm')
            self.m.pose[48] = selection_l_arm['lS_roll']*2/3
            self.m.pose[49] = selection_l_arm['lS_yaw']*2/3
            self.m.pose[50] = selection_l_arm['lS_pitch']*2/3
            self.m.pose[39] = selection_l_arm['lS_roll']*1/3
            self.m.pose[40] = selection_l_arm['lS_yaw']*1/3
            self.m.pose[41] = selection_l_arm['lS_pitch']*1/3

            self.m.pose[55] = selection_l_arm['lE']

        #self.m.pose[51] = selection_r
        from capsule_body import get_capsules, joint2name, rots0
        capsules = get_capsules(self.m)
        joint2name = joint2name
        rots0 = rots0


        return self.m, capsules, joint2name, rots0


    def generate_dataset(self, gender, posture, num_data, stiffness):
        #NEED FOR DATASET: pose Nx72, shape Nx10
        shape_pose_vol_list = []
        #contact_check_bns = [1, 2, 4, 5, 7, 8, 14, 15, 16, 17, 18, 19]
        contact_check_bns = [4, 5, 7, 8, 16, 17, 18, 19]
        contact_exceptions = [[9, 14],[9, 15]]


        for i in range(num_data):
            shape_pose_vol = [[],[],[],[],[],[],[]]

            root_rot = np.random.uniform(-np.pi / 16, np.pi / 16)
            shift_side = np.random.uniform(-0.2, 0.2)  # in meters
            shift_ud = np.random.uniform(-0.2, 0.2)  # in meters
            shape_pose_vol[3] = root_rot
            shape_pose_vol[4] = shift_side
            shape_pose_vol[5] = shift_ud

            #generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)
            in_collision = True

            self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.
            dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture=posture, stiffness=None, check_only_distal = False, filepath_prefix=self.filepath_prefix, add_floor = False)
            print "dataset create type", DATASET_CREATE_TYPE
            #print self.m.pose
            volumes = dss.getCapsuleVolumes(mm_resolution = 1., dataset_num = DATASET_CREATE_TYPE)

            libRender.standard_render(self.m)
            print volumes
            shape_pose_vol[6] = volumes


            while in_collision == True:
                #m, capsules, joint2name, rots0 = generator.map_random_selection_to_smpl_angles(alter_angles = True)
                m, capsules, joint2name, rots0 = generator.map_shuffled_yash_to_smpl_angles(posture, shift_ud, alter_angles = True)

                shape_pose_vol[0] = np.asarray(m.betas).tolist()

                #print "stepping", m.pose
                dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture = posture, stiffness=None, filepath_prefix=self.filepath_prefix, add_floor = False)

                #print "stepping", m.pose
                invalid_pose = False
                #run a step to check for collisions
                dss.run_sim_step()

                #dss.world.check_collision()
                #print "checked collisions"
                #dss.run_simulation(1)
                #print dss.world.CollisionResult()
                #print dss.world.collision_result.contacted_bodies
                print dss.world.collision_result.contact_sets
                if len(dss.world.collision_result.contacted_bodies) != 0:
                    for contact_set in dss.world.collision_result.contact_sets:
                        if contact_set[0] in contact_check_bns or contact_set[1] in contact_check_bns: #consider removing spine 3 and upper legs
                            if contact_set in contact_exceptions:
                                pass

                            else:
                                #print "one of the limbs in contact"
                                print contact_set
                                #dss.run_simulation(1)

                                print "resampling pose from the same shape, invalid pose"
                                libRender.standard_render(self.m)
                                in_collision = True
                                invalid_pose = True
                            break

                    if invalid_pose == False:
                        print "resampling shape and pose, collision not important."

                        libRender.standard_render(self.m)
                        in_collision = False
                else: # no contacts anywhere.

                    print "resampling shape and pose, no collision."
                    in_collision = False



                #dss.world.skeletons[0].remove_all_collision_pairs()

                libRender.standard_render(self.m)
                dss.world.reset()
                dss.world.destroy()

            pose_indices = [0, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 27, 36, 39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 55, 58]
            pose_angles = []
            for index in pose_indices:
                pose_angles.append(float(m.pose[index]))

            shape_pose_vol[1] = pose_indices
            shape_pose_vol[2] = pose_angles


            shape_pose_vol_list.append(shape_pose_vol)

        print "SAVING! "
        #print shape_pose_vol_list
        #pickle.dump(shape_pose_vol_list, open("/home/henry/git/volumetric_pose_gen/valid_shape_pose_vol_list1.pkl", "wb"))
        np.save(self.filepath_prefix+"/git/volumetric_pose_gen/valid_shape_pose_vol_"+gender+"_"+posture+"_"+str(num_data)+"_"+stiffness+"_stiff.npy", np.array(shape_pose_vol_list))

    def fix_dataset(self, gender, posture, num_data, stiffness, filepath_prefix):


        filename = filepath_prefix+"/data/init_poses_contactissue/valid_shape_pose_vol_"+gender+"_"+posture+"_"+str(num_data)+"_"+stiffness+"_stiff.npy"

        old_pose_list = np.load(filename).tolist()


        #NEED FOR DATASET: pose Nx72, shape Nx10
        shape_pose_vol_list = []
        #contact_check_bns = [1, 2, 4, 5, 7, 8, 14, 15, 16, 17, 18, 19]
        contact_check_bns = [4, 5, 7, 8, 16, 17, 18, 19]
        contact_exceptions = [[9, 14],[9, 15]]


        for ct in range(len(old_pose_list)):
            #if ct == 20: break


            shape_pose_vol_old = old_pose_list[ct]
            shape_pose_vol = []


            in_collision = True
            pose_shapes = shape_pose_vol_old[0]

            for i in range(10):
                self.m.betas[i] = pose_shapes[i]

            shift_ud = shape_pose_vol_old[5]

            while in_collision == True:

                #m, capsules, joint2name, rots0 = generator.map_random_selection_to_smpl_angles(alter_angles = True)
                m, capsules, joint2name, rots0 = generator.map_shuffled_yash_to_smpl_angles(posture, shift_ud, alter_angles = True)

                #print "stepping", m.pose
                dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture = posture, stiffness=None, filepath_prefix=self.filepath_prefix, add_floor = False)
                #print "stepping", m.pose

                invalid_pose = False
                #run a step to check for collisions
                dss.run_sim_step()

                #dss.world.check_collision()

                print dss.world.collision_result.contact_sets
                if len(dss.world.collision_result.contacted_bodies) != 0:
                    for contact_set in dss.world.collision_result.contact_sets:
                        if contact_set[0] in contact_check_bns or contact_set[1] in contact_check_bns: #consider removing spine 3 and upper legs
                            if contact_set in contact_exceptions:
                                pass

                            else:
                                #print "one of the limbs in contact"
                                print contact_set
                                #dss.run_simulation(1)

                                print "resampling pose from the same shape, invalid pose"
                                libRender.standard_render(self.m)
                                in_collision = True
                                invalid_pose = True
                            break

                    if invalid_pose == False:
                        print "resampling shape and pose, collision not important."

                        libRender.standard_render(self.m)
                        in_collision = False
                else: # no contacts anywhere.

                    print "resampling shape and pose, no collision."
                    in_collision = False

                dss.world.reset()
                dss.world.destroy()

            pose_indices = [0, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 27, 36, 39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 55, 58]
            pose_angles = []
            for index in pose_indices:
                pose_angles.append(float(m.pose[index]))


            for item in shape_pose_vol_old:
                shape_pose_vol.append(item)


            shape_pose_vol[2] = pose_angles
            #print shape_pose_vol[5:]


            #for item in shape_pose_vol:
            #    print item

            shape_pose_vol_list.append(shape_pose_vol)

        print "SAVING! "
        #print shape_pose_vol_list
        #pickle.dump(shape_pose_vol_list, open("/home/henry/git/volumetric_pose_gen/valid_shape_pose_vol_list1.pkl", "wb"))
        np.save(self.filepath_prefix+"/data/init_poses/valid_shape_pose_vol_"+gender+"_"+posture+"_"+str(len(old_pose_list))+"_"+stiffness+"_stiff.npy", np.array(shape_pose_vol_list))


    def generate_prechecked_pose(self, gender, posture, stiffness, filename):


        prechecked_pose_list = np.load(filename).tolist()

        import trimesh
        import pyrender
        self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 1.0 ,0.5])

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


            for idx in range(len(shape_pose_vol[1])):
                #print shape_pose_vol[1][idx]
                #print self.m.pose[shape_pose_vol[1][idx]]
                #print shape_pose_vol[2][idx]

                self.m.pose[shape_pose_vol[1][idx]] = shape_pose_vol[2][idx]


            print "shift up down", shape_pose_vol[5]


            #self.m.pose[3] = -np.pi/10
            #self.m.pose[5] = np.pi/12
            #self.m.pose[8] = np.pi/6
            #self.m.pose[12] = np.pi/4
            #self.m.pose[44] = np.pi/6
            #self.m.pose[53] = np.pi/4
            #self.m.pose[41] = -np.pi/10
            #self.m.pose[50] = -np.pi/8
            #self.m.pose[48] = -np.pi/6
            #self.m.pose[58] = np.pi/6
            #self.m.pose[55] = -np.pi/6

            ## Write to an .obj file
            #outmesh_path = "./data/person.obj"
            #with open(outmesh_path, 'w') as fp:
            #    for v in self.m.r:
            #        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            #    for f in self.m.f + 1:  # Faces are 1-based, not 0-based in obj files
            #        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

            #rospy.init_node("smpl_viz")

            #while not rospy.is_shutdown():
            #    libVisualization.rviz_publish_output(np.array(self.m.J_transformed))
            #    libVisualization.rviz_publish_output_limbs_direct(np.array(self.m.J_transformed))
            #self.m.pose[0] = np.pi/6

            #print self.m.J_transformed[1, :], self.m.J_transformed[4, :]



            #R_l_hip_rod = libKinematics.matrix_from_dir_cos_angles([float(self.m.pose[3]), float(self.m.pose[4]), float(self.m.pose[5]) ])
            #R_r_hip_rod = libKinematics.matrix_from_dir_cos_angles([float(self.m.pose[6]), float(self.m.pose[7]), float(self.m.pose[8]) ])
            #R_root = libKinematics.eulerAnglesToRotationMatrix([-float(self.m.pose[0]), 0.0, 0.0])

            #R_l = np.matmul(R_root, R_l_hip_rod)
            #R_r = np.matmul(R_root, R_r_hip_rod)

            #new_left_hip = libKinematics.dir_cos_angles_from_matrix(R_l)
            #new_right_hip = libKinematics.dir_cos_angles_from_matrix(R_r)


            #self.m.pose[3] = new_left_hip[0]
            #self.m.pose[4] = new_left_hip[1]
            #self.m.pose[5] = new_left_hip[2]
            #self.m.pose[6] = new_right_hip[0]
            #self.m.pose[7] = new_right_hip[1]
            #self.m.pose[8] = new_right_hip[2]


            #print self.m.J_transformed[1, :], self.m.J_transformed[4, :]
            # self.m.pose[51] = selection_r


            #pyRender.mesh_render(self.m)


            verts = np.array(self.m.r)
            faces = np.array(self.m.f)
            tm = trimesh.base.Trimesh(vertices=verts, faces=faces)


            smpl_mesh = pyrender.Mesh.from_trimesh(tm, material=self.human_mat, wireframe=True , smooth = False)# smoothing doesn't do anything to wireframe

            #dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender = gender, posture = posture, stiffness = stiffness, shiftSIDE = shape_pose_vol[4], shiftUD = shape_pose_vol[5], filepath_prefix=self.filepath_prefix, add_floor = False)

            #dss.run_simulation(10000)
            #generator.standard_render()


            #break

    def doublecheck_prechecked_list(self, gender, posture, stiffness, filename):
        prechecked_pose_list = np.load(filename).tolist()
        contact_check_bns = [4, 5, 7, 8, 16, 17, 18, 19]

        left_abd, right_abd = [], []
        for shape_pose_vol in prechecked_pose_list[0:100]:
            for idx in range(len(shape_pose_vol[0])):
                #print shape_pose_vol[0][idx]
                self.m.betas[idx] = shape_pose_vol[0][idx]


            for idx in range(len(shape_pose_vol[1])):
                #print shape_pose_vol[1][idx]
                #print self.m.pose[shape_pose_vol[1][idx]]
                #print shape_pose_vol[2][idx]

                self.m.pose[shape_pose_vol[1][idx]] = shape_pose_vol[2][idx]

            left_abd.append(np.array(self.m.pose[5]))
            right_abd.append(float(self.m.pose[8]))
            #sleep(1)

            dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture=posture, stiffness=stiffness,
                                            shiftSIDE=shape_pose_vol[4], shiftUD=shape_pose_vol[5],
                                            filepath_prefix=self.filepath_prefix, add_floor = False)
            dss.run_sim_step()
            dss.world.destroy()
        print np.mean(left_abd) #.15 .16
        print np.mean(right_abd) #-.18 -.13

                # print "one of the limbs in contact"
            #print dss.world.collision_result.contact_sets

if __name__ == "__main__":

    gender = "m"
    num_data = 4000
    posture = "lay"
    stiffness = "rightside"
    filepath_prefix = "/home/henry"



    DATASET_CREATE_TYPE = None




    if DATASET_CREATE_TYPE == None:
        generator = GeneratePose(gender, posture, filepath_prefix)
        generator.generate_prechecked_pose(gender, posture, stiffness, filepath_prefix+"/data/init_poses/valid_shape_pose_vol_"+gender+"_"+posture+"_"+str(num_data)+"_"+stiffness+"_stiff.npy")
        #generator.generate_dataset(gender = gender, posture = posture, num_data = num_data, stiffness = stiffness)
        #generator.doublecheck_prechecked_list(gender, posture, stiffness, filepath_prefix+"/data/init_poses/valid_shape_pose_"+gender+"_"+posture+"_"+str(num_data)+"_"+stiffness+"_stiff.npy")

    if DATASET_CREATE_TYPE == 1:
        generator = GeneratePose("m",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "rightside")
        #generator.fix_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "rightside", filepath_prefix = filepath_prefix)
        generator = GeneratePose("f",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "rightside")
        #generator.fix_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "rightside", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 2:
        generator = GeneratePose("m",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "leftside")
        #generator.fix_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "leftside", filepath_prefix = filepath_prefix)
        generator = GeneratePose("f",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "leftside")
        #generator.fix_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "leftside", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 3:
        generator = GeneratePose("m",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "upperbody")
        #generator.fix_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "upperbody", filepath_prefix = filepath_prefix)
        generator = GeneratePose("f",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "upperbody")
        #generator.fix_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "upperbody", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 4:
        generator = GeneratePose("m",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "lowerbody")
        #generator.fix_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "lowerbody", filepath_prefix = filepath_prefix)
        generator = GeneratePose("f",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "lowerbody")
        #generator.fix_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "lowerbody", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 5:
        generator = GeneratePose("m",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "none")
        #generator.fix_dataset(gender = "m", posture = "sit", num_data = 2000, stiffness = "none", filepath_prefix = filepath_prefix)
        generator = GeneratePose("f",  "sit", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "none")
        #generator.fix_dataset(gender = "f", posture = "sit", num_data = 2000, stiffness = "none", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 6:
        generator = GeneratePose("m",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "rightside")
        #generator.fix_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "rightside", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 7:
        generator = GeneratePose("f",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "rightside")
        #generator.fix_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "rightside", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 8:
        generator = GeneratePose("m",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "leftside")
        #generator.fix_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "leftside", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 9:
        generator = GeneratePose("f",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "leftside")
        #generator.fix_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "leftside", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 10:
        generator = GeneratePose("m",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "upperbody")
        #generator.fix_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "upperbody", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 11:
        generator = GeneratePose("f",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "upperbody")
        #generator.fix_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "upperbody", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 12:
        generator = GeneratePose("m",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "lowerbody")
        #generator.fix_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "lowerbody", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 13:
        generator = GeneratePose("f",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "lowerbody")
        #generator.fix_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "lowerbody", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 14:
        generator = GeneratePose("m",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "none")
        #generator.fix_dataset(gender = "m", posture = "lay", num_data = 4000, stiffness = "none", filepath_prefix = filepath_prefix)
    elif DATASET_CREATE_TYPE == 15:
        generator = GeneratePose("f",  "lay", filepath_prefix)
        #generator.generate_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "none")
        #generator.fix_dataset(gender = "f", posture = "lay", num_data = 4000, stiffness = "none", filepath_prefix = filepath_prefix)
