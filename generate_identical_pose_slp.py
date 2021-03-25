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
import lib_kinematics as libKinematics
import lib_render as libRender
import dart_skel_sim_slp as dart_skel_sim
from time import sleep

import cPickle as pkl

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)
#ROS
#import rospy
#import tf
DATASET_CREATE_TYPE =5

import cv2

import math
from random import shuffle
import torch
import torch.nn as nn

import tensorflow as tensorflow
import cPickle as pickle


#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL



class GeneratePose():
    def __init__(self, gender, filepath_prefix = '/home/henry'):
        ## Load SMPL model (here we load the female model)
        self.filepath_prefix = filepath_prefix

        if gender == "m":
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = filepath_prefix+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.reset_pose = False
        self.m = load_model(model_path)

        filename = self.filepath_prefix+'/data/init_ik_solutions/all_lay_angles_side_up.p'
        with open(filename, 'rb') as fp:
            self.angles_data = pickle.load(fp)
        shuffle(self.angles_data)



    def assign_body_shape(self, original_betas):
        for i in range(10):
            self.m.betas[i] = original_betas[i]	






    def map_slp_to_rand_angles(self, original_pose, alter_angles = True):

        R_root = libKinematics.matrix_from_dir_cos_angles(original_pose[0:3])

        flip_root_euler = np.pi
        flip_root_R = libKinematics.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])

        root_rot_rand_R = libKinematics.eulerAnglesToRotationMatrix([0.0, 0.0, 0.0])  # randomize the root rotation
        # root_rot_rand_R = libKinematics.eulerAnglesToRotationMatrix([0.0,  0.0, 0.0]) #randomize the root rotation

        dir_cos_root = libKinematics.dir_cos_angles_from_matrix(np.matmul(root_rot_rand_R, np.matmul(R_root, flip_root_R)))

        # R_root2 = libKinematics.matrix_from_dir_cos_angles([original_pose[0]-4*np.pi, original_pose[1], original_pose[2]])
        # dir_cos_root2 = libKinematics.dir_cos_angles_from_matrix(R_root2)
        # print('eulers2', libKinematics.rotationMatrixToEulerAngles(R_root2))

        self.m.pose[0] = dir_cos_root[0]
        self.m.pose[1] = dir_cos_root[1]
        self.m.pose[2] = dir_cos_root[2]

        for i in range(3, 72):
            self.m.pose[i] = original_pose[i]



        from capsule_body import get_capsules, joint2name, rots0
        capsules = get_capsules(self.m)
        joint2name = joint2name
        rots0 = rots0
        print len(capsules)

        return self.m, capsules, joint2name, rots0





    def generate_dataset(self, gender, some_subject, num_samp_per_slp_pose):
        #NEED FOR DATASET: pose Nx72, shape Nx10
        shape_pose_vol_list = []

        for pose_num in range(1, 2):#46):#45 pose per participant.
            #here load some subjects joint angle data within danaLab and found by SMPLIFY

            try:
                original_pose_data = load_pickle('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
            except:
                continue

            print(original_pose_data)


            original_pose = original_pose_data['pose']
            original_shape = original_pose_data['betas']
            original_cam_t = original_pose_data['cam_t']

            for i in range(num_samp_per_slp_pose):
                shape_pose_vol = [[],[],[],[],[],[],[]]

                #root_rot = np.random.uniform(-np.pi / 16, np.pi / 16)
                shape_pose_vol[3] = None #instead of root_rot
                shape_pose_vol[4] = original_cam_t[0]
                shape_pose_vol[5] = original_cam_t[1]

                generator.assign_body_shape(original_shape)
                in_collision = True

                print('init m betas', self.m.betas)
                self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.
                dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture='lay', stiffness=None, check_only_distal = True, filepath_prefix=self.filepath_prefix, add_floor = False)
                print "dataset create type", DATASET_CREATE_TYPE
                #print self.m.pose
                volumes = dss.getCapsuleVolumes(mm_resolution = 1., dataset_num = DATASET_CREATE_TYPE)
                #volumes = 0
                print volumes
                shape_pose_vol[6] = volumes
                dss.world.reset()
                dss.world.destroy()



                self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.

                m, capsules, joint2name, rots0 = generator.map_slp_to_rand_angles(original_pose)
                libRender.standard_render(m)

                #print "GOT HERE"
                #time.sleep(2)

                shape_pose_vol[0] = np.asarray(m.betas).tolist()


                #pose_indices = [0, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 27, 36, 39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53, 55, 58]
                pose_angles = []
                pose_indices = []
                for index in range(72):
                    pose_indices.append(int(index))
                    pose_angles.append(float(m.pose[index]))
                    

                shape_pose_vol[1] = pose_indices
                shape_pose_vol[2] = pose_angles


                shape_pose_vol_list.append(shape_pose_vol)

        print "SAVING! "
        #print shape_pose_vol_list
        #pickle.dump(shape_pose_vol_list, open("/home/henry/git/volumetric_pose_gen/valid_shape_pose_vol_list1.pkl", "wb"))
        np.save(self.filepath_prefix+"/data/init_poses/slp_identical/identical_shape_pose_vol_"+some_subject+"_"+gender+"_"+str(len(shape_pose_vol_list))+".npy", np.array(shape_pose_vol_list))




    def resave_individual_files(self):

        import os


        #starting_subj = 51
        #for starting_subj in [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]:
        for starting_subj in [51]:#, 11, 21, 31, 41, 51, 61, 71, 81, 91]:

            posture_list = [["lay", 0], ["ls", 15],['rs', 30]]
            for posture in posture_list:


                for gender in ["f", "m"]:

                    new_valid_shape_pose_vol_list = []

                    #for i in range(51,103):
                    for i in range(starting_subj,starting_subj+40):
                        if i == 7: continue
                        some_subject = '%05d' % (i)


                        onlyfiles = next(os.walk('/home/henry/git/smplify_public/output_'+some_subject))[2]
                        print(len(onlyfiles))
                        number_files = str(int(len(onlyfiles)*1/2))

                        try:
                            subject_new_samples = np.load(self.filepath_prefix + "/data/init_poses/slp_identical/identical_bins/identical_shape_pose_vol_" +some_subject+ "_"+gender + "_"+number_files+".npy", allow_pickle=True)
                        except:
                            continue

                        print(len(subject_new_samples))

                        within_list_pose_ct = 0

                        #for pose_num in range(1, 46):#45 pose per participant.
                        for pose_num in range(1, 46):#45 pose per participant.

                            #here load some subjects joint angle data within danaLab and found by SMPLIFY

                            try:
                                original_pose_data = load_pickle('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                                #print(original_pose_data)
                                original_pose = original_pose_data['pose']

                                #if pose_num-1


                                if pose_num - 1 >= posture[1] and pose_num - 1 < posture[1] + 15:


                                    new_valid_shape_pose_vol_list.append(subject_new_samples[within_list_pose_ct])

                                        #print(some_subject, 'appending: ', pose_num, '  within list ct:', within_list_pose_ct+i)

                                within_list_pose_ct += 1

                                #print(i, pose_num)
                                #here make an if statement

                            except:
                                #pass
                                print(i,pose_num,' is missing for subject:', some_subject)

                        print('poses within: ', within_list_pose_ct, '    poses appended: ', len(new_valid_shape_pose_vol_list), posture[0])

                    np.save(self.filepath_prefix + "/data/init_poses/slp_identical/identical_shape_pose_vol_" + posture[0] + "_" + gender + "_" +str(starting_subj) + "to" +str(starting_subj+39) + "_" +  str(len(new_valid_shape_pose_vol_list)) + ".npy", new_valid_shape_pose_vol_list)






    def fix_root(self):

        import os



        for gender in ["f", "m"]:

            #for i in range(51,103):
            for i in range(1, 103):
                if i == 7: continue
                some_subject = '%05d' % (i)


                onlyfiles = next(os.walk('/home/henry/git/smplify_public/output_'+some_subject))[2]
                print(len(onlyfiles))
                number_files = str(int(len(onlyfiles)*1/2))

                new_valid_shape_pose_vol_list = []

                try:
                    subject_new_samples = np.load(self.filepath_prefix + "/data/init_poses/slp_identical/identical_bins/identical_shape_pose_vol_" +some_subject+ "_"+gender + "_"+number_files+".npy", allow_pickle=True)
                except:
                    continue

                print(len(subject_new_samples))

                #for pose_num in range(1, 46):#45 pose per participant.
                for pose_num in range(0, 45):#45 pose per participant.

                    #here load some subjects joint angle data within danaLab and found by SMPLIFY
                    try:
                        root_angles = np.array(subject_new_samples[pose_num][2][0:3]) * 1.

                        R_root = libKinematics.matrix_from_dir_cos_angles(root_angles)

                        flip_root_euler = np.pi
                        flip_root_R = libKinematics.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])

                        root_rot_rand_R = libKinematics.eulerAnglesToRotationMatrix([0.0, 0.0, 0.0])  # randomize the root rotation
                        # root_rot_rand_R = libKinematics.eulerAnglesToRotationMatrix([0.0,  0.0, 0.0]) #randomize the root rotation

                        dir_cos_root = libKinematics.dir_cos_angles_from_matrix(np.matmul(root_rot_rand_R, np.matmul(R_root, flip_root_R)))

                        print(subject_new_samples[pose_num][2][0:3], dir_cos_root)
                        subject_new_samples[pose_num][2][0] = dir_cos_root[0]
                        subject_new_samples[pose_num][2][1] = dir_cos_root[1]
                        subject_new_samples[pose_num][2][2] = dir_cos_root[2]
                        #print(subject_new_samples[pose_num][2][0:3])

                    except:
                        #pass
                        print(i,pose_num,' is missing for subject:', some_subject)


                #np.save(self.filepath_prefix + "/data/init_poses/slp_identical/identical_bins/identical_shape_pose_vol_" +some_subject+ "_"+gender + "_"+number_files+".npy", subject_new_samples)









if __name__ == "__main__":


    for i in range (2, 3):
        some_subject = '%05d' % i
        filepath_prefix = "/home/henry"



        phys_arr = np.load('../../data/SLP/danaLab/physiqueData.npy')
        phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
        gender_bin = phys_arr[int(some_subject) - 1][2]
        if int(gender_bin) == 1:
            gender = "m"
        else:
            gender = "f"


        #DATASET_CREATE_TYPE = 1'''



        generator = GeneratePose(gender, filepath_prefix)


        generator.resave_individual_files()
        #generator.fix_root()
        #generator.generate_dataset(gender, some_subject = some_subject, num_samp_per_slp_pose = 1)
        #generator.fix_dataset(gender = "m", num_data = 3000, filepath_prefix = filepath_prefix)
        #generator.doublecheck_prechecked_list(gender, filepath_prefix+"/data/init_poses/valid_shape_pose_"+gender+"_"+str(num_data)+".npy")

