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



    def sample_body_shape(self, sampling, sigma, one_side_range):
        mu = 0
        for i in range(10):
            if sampling == "NORMAL":
                self.m.betas[i] = random.normalvariate(mu, sigma)
            elif sampling == "UNIFORM":
                self.m.betas[i]  = np.random.uniform(-one_side_range, one_side_range)



    def get_noisy_angle(self, angle, angle_min, angle_max):
        not_within_bounds = True
        mu = 0
        counter = 0
        #sigma = np.pi/16
        sigma = np.pi/12

        #print "angle to make noisy", angle, angle_min, angle_max,
        while not_within_bounds == True:

            noisy_angle = angle + random.normalvariate(mu, sigma)
            if noisy_angle > angle_min and noisy_angle < angle_max:
                #print "angle, min, max", noisy_angle, angle_min, angle_max
                not_within_bounds = False
            else:
                print "angle, min, max", noisy_angle, angle_min, angle_max
                counter += 1
                if counter > 10:
                    self.reset_pose = True
                    break
                pass

        #print "  noisy angle", noisy_angle
        return noisy_angle

    def get_noisy_angle_hard_limit(self, angle, angle_min, angle_max):
        not_within_bounds = True
        mu = 0
        counter = 0
        #sigma = np.pi/16
        sigma = np.pi/12

        noisy_angle = angle + random.normalvariate(mu, sigma)
        if noisy_angle < angle_min:
            noisy_angle = angle_min*1.
        elif noisy_angle > angle_max:
            noisy_angle = angle_max*1.

        #print "  noisy angle", noisy_angle
        return noisy_angle





    def map_slp_to_rand_angles(self, original_pose, alter_angles = True):
        angle_type = "angle_axis"
        #angle_type = "euler"

        dircos_limit = {}
        dircos_limit['hip0_L'] = -1.8620680158061373
        dircos_limit['hip0_U'] = 0.3928991379790361
        dircos_limit['hip1_L'] = -0.3002682177448498
        dircos_limit['hip1_U'] = 0.4312268766210817
        dircos_limit['hip2_L'] = -0.5252221939422439
        dircos_limit['hip2_U'] = 0.9361197251810272
        dircos_limit['knee_L'] = -0.00246317350132912
        dircos_limit['knee_U'] = 2.5615940356353004

        dircos_limit['shd00_L'] = -0.44788772883633016
        dircos_limit['shd00_U'] = 0.20496654360962563
        dircos_limit['shd01_L'] = -1.0991885726395296
        dircos_limit['shd01_U'] = 0.6828417105678483
        dircos_limit['shd02_L'] = -0.6177946204522845
        dircos_limit['shd02_U'] = 0.7288264054368747

        dircos_limit['shd10_L'] = -0.8121612985394123
        dircos_limit['shd10_U'] = 0.7203413993648013
        dircos_limit['shd11_L'] = -1.3427142449685556
        dircos_limit['shd11_U'] = 0.4865822031689829
        dircos_limit['shd12_L'] = -1.292990223497471
        dircos_limit['shd12_U'] = 0.7428419209098167

        dircos_limit['elbow_L'] = -2.656264518131647
        dircos_limit['elbow_U'] = 0.1873184747130497

        print('alter angs',alter_angles)

        while True:
            self.reset_pose = False

            R_root = libKinematics.matrix_from_dir_cos_angles(original_pose[0:3])

            flip_root_euler = np.pi
            flip_root_R = libKinematics.eulerAnglesToRotationMatrix([flip_root_euler, 0.0, 0.0])

            root_rot_rand_R = libKinematics.eulerAnglesToRotationMatrix([0.0,  random.normalvariate(0.0, np.pi/12),  random.normalvariate(0.0, np.pi/12)]) #randomize the root rotation
            #root_rot_rand_R = libKinematics.eulerAnglesToRotationMatrix([0.0,  0.0, 0.0]) #randomize the root rotation

            dir_cos_root = libKinematics.dir_cos_angles_from_matrix(np.matmul(root_rot_rand_R, np.matmul(R_root, flip_root_R)))


            #R_root2 = libKinematics.matrix_from_dir_cos_angles([original_pose[0]-4*np.pi, original_pose[1], original_pose[2]])
            #dir_cos_root2 = libKinematics.dir_cos_angles_from_matrix(R_root2)
            #print('eulers2', libKinematics.rotationMatrixToEulerAngles(R_root2))


            self.m.pose[0] = dir_cos_root[0]
            self.m.pose[1] = -dir_cos_root[1]
            self.m.pose[2] = dir_cos_root[2]

            #print(original_pose[0:3])
            #print(dir_cos_root)
            #print(dir_cos_root2)




            self.m.pose[3] = generator.get_noisy_angle_hard_limit(original_pose[3], dircos_limit['hip0_L'], dircos_limit['hip0_U'])
            self.m.pose[4] = generator.get_noisy_angle_hard_limit(original_pose[4], dircos_limit['hip1_L'], dircos_limit['hip1_U'])
            self.m.pose[5] = generator.get_noisy_angle_hard_limit(original_pose[5], dircos_limit['hip2_L'], dircos_limit['hip2_U'])
            self.m.pose[12] = generator.get_noisy_angle_hard_limit(original_pose[12], dircos_limit['knee_L'], dircos_limit['knee_U'])
            self.m.pose[13] = original_pose[13]
            self.m.pose[14] = original_pose[14]



            self.m.pose[6] = generator.get_noisy_angle_hard_limit(original_pose[6], dircos_limit['hip0_L'], dircos_limit['hip0_U'])
            self.m.pose[7] = generator.get_noisy_angle_hard_limit(original_pose[7], -dircos_limit['hip1_U'], -dircos_limit['hip1_L'])
            self.m.pose[8] = generator.get_noisy_angle_hard_limit(original_pose[8], -dircos_limit['hip2_U'], -dircos_limit['hip2_L'])
            self.m.pose[15] = generator.get_noisy_angle_hard_limit(original_pose[15], dircos_limit['knee_L'], dircos_limit['knee_U'])
            self.m.pose[16] = original_pose[16]
            self.m.pose[17] = original_pose[17]



            self.m.pose[9] = original_pose[9] #stomach
            self.m.pose[10] = original_pose[10] #stomach
            self.m.pose[11] = original_pose[11] #stomach


            self.m.pose[18] = original_pose[18]#chest
            self.m.pose[19] = original_pose[19]#chest
            self.m.pose[20] = original_pose[20]#chest
            self.m.pose[21] = original_pose[21]#l ankle
            self.m.pose[22] = original_pose[22]#l ankle
            self.m.pose[23] = original_pose[23]#l ankle
            self.m.pose[24] = original_pose[24]#r ankle
            self.m.pose[25] = original_pose[25]#r ankle
            self.m.pose[26] = original_pose[26]#r ankle
            self.m.pose[27] = original_pose[27]#sternum
            self.m.pose[28] = original_pose[28]#sternum
            self.m.pose[29] = original_pose[29]#stermum
            self.m.pose[30] = original_pose[30]#l foot
            self.m.pose[31] = original_pose[31]#l foot
            self.m.pose[32] = original_pose[32]#l foot
            self.m.pose[33] = original_pose[33]#r foot
            self.m.pose[34] = original_pose[34]#r foot
            self.m.pose[35] = original_pose[35]#r foot
            self.m.pose[36] = original_pose[36]#neck
            self.m.pose[37] = original_pose[37]#neck
            self.m.pose[38] = original_pose[38]#neck

            self.m.pose[45] = original_pose[45]#head
            self.m.pose[46] = original_pose[46]#head
            self.m.pose[47] = original_pose[47]#head


            self.m.pose[39] = generator.get_noisy_angle_hard_limit(original_pose[39], dircos_limit['shd00_L'], dircos_limit['shd00_U'])
            self.m.pose[40] = generator.get_noisy_angle_hard_limit(original_pose[40], dircos_limit['shd01_L'], dircos_limit['shd01_U'])
            self.m.pose[41] = generator.get_noisy_angle_hard_limit(original_pose[41], dircos_limit['shd02_L'], dircos_limit['shd02_U'])
            self.m.pose[42] = generator.get_noisy_angle_hard_limit(original_pose[42], dircos_limit['shd00_L'], dircos_limit['shd00_U'])
            self.m.pose[43] = generator.get_noisy_angle_hard_limit(original_pose[43], -dircos_limit['shd01_U'], -dircos_limit['shd01_L'])
            self.m.pose[44] = generator.get_noisy_angle_hard_limit(original_pose[44], -dircos_limit['shd02_U'], -dircos_limit['shd02_L'])
            self.m.pose[54] = original_pose[54]
            self.m.pose[55] = generator.get_noisy_angle_hard_limit(original_pose[55], dircos_limit['elbow_L'], dircos_limit['elbow_U'])
            self.m.pose[56] = original_pose[56]


            self.m.pose[48] = generator.get_noisy_angle_hard_limit(original_pose[48], dircos_limit['shd10_L'], dircos_limit['shd10_U'])
            self.m.pose[49] = generator.get_noisy_angle_hard_limit(original_pose[49], dircos_limit['shd11_L'], dircos_limit['shd11_U'])
            self.m.pose[50] = generator.get_noisy_angle_hard_limit(original_pose[50], dircos_limit['shd12_L'], dircos_limit['shd12_U'])
            self.m.pose[51] = generator.get_noisy_angle_hard_limit(original_pose[51], dircos_limit['shd10_L'], dircos_limit['shd10_U'])
            self.m.pose[52] = generator.get_noisy_angle_hard_limit(original_pose[52], -dircos_limit['shd11_U'], -dircos_limit['shd11_L'])
            self.m.pose[53] = generator.get_noisy_angle_hard_limit(original_pose[53], -dircos_limit['shd12_U'], -dircos_limit['shd12_L'])
            self.m.pose[57] = original_pose[57]
            self.m.pose[58] = generator.get_noisy_angle_hard_limit(original_pose[58], -dircos_limit['elbow_U'], -dircos_limit['elbow_L'])
            self.m.pose[59] = original_pose[59]

            for i in range(60, 72):
                self.m.pose[i] = original_pose[i]


            print "stuck in loop", self.reset_pose
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





    def generate_dataset(self, gender, some_subject, num_samp_per_slp_pose):
        #NEED FOR DATASET: pose Nx72, shape Nx10
        shape_pose_vol_list = []
        #contact_check_bns = [1, 2, 4, 5, 7, 8, 14, 15, 16, 17, 18, 19]
        contact_check_bns = [4, 5, 7, 8, 16, 17, 18, 19]
        contact_exceptions = [[9, 14],[9, 15]]



        for pose_num in range(1, 46):#45 pose per participant.
            #here load some subjects joint angle data within danaLab and found by SMPLIFY

            try:
                original_pose_data = load_pickle('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
            except:
                continue
            #print(original_pose_data)
            original_pose = original_pose_data['pose']

            for i in range(num_samp_per_slp_pose):
                shape_pose_vol = [[],[],[],[],[],[],[]]

                #root_rot = np.random.uniform(-np.pi / 16, np.pi / 16)
                shift_side = np.random.uniform(-0.2, 0.2)  # in meters
                shift_ud = np.random.uniform(-0.2, 0.2)  # in meters
                shape_pose_vol[3] = None #instead of root_rot
                shape_pose_vol[4] = shift_side
                shape_pose_vol[5] = shift_ud

                generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)
                in_collision = True

                print('init m betas', self.m.betas)
                self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.
                dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture='lay', stiffness=None, check_only_distal = True, filepath_prefix=self.filepath_prefix, add_floor = False)
                print "dataset create type", DATASET_CREATE_TYPE
                #print self.m.pose
                volumes = dss.getCapsuleVolumes(mm_resolution = 1., dataset_num = DATASET_CREATE_TYPE)
                #volumes = 0
                #libRender.standard_render(self.m)
                print volumes
                shape_pose_vol[6] = volumes
                dss.world.reset()
                dss.world.destroy()





                while in_collision == True:

                    #print "GOT HERE"
                    #time.sleep(2)

                    self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.

                    m, capsules, joint2name, rots0 = generator.map_slp_to_rand_angles(original_pose)

                    #print "GOT HERE"
                    #time.sleep(2)

                    shape_pose_vol[0] = np.asarray(m.betas).tolist()

                    print "stepping", m.betas
                    dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture = 'lay', stiffness=None, check_only_distal = True, filepath_prefix=self.filepath_prefix, add_floor = True)

                    #print "stepping", m.pose
                    invalid_pose = False
                    #run a step to check for collisions
                    dss.run_sim_step()

                    dss.world.check_collision()
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
                                    #dss.run_simulation(1000)
                                    #libRender.standard_render(self.m)
                                    in_collision = True
                                    invalid_pose = True
                                break

                        if invalid_pose == False:
                            print "resampling shape and pose, collision not important."
                            #dss.run_simulation(1)
                            #libRender.standard_render(self.m)
                            in_collision = False
                    else: # no contacts anywhere.

                        #dss.run_simulation(1)
                        print "resampling shape and pose, no collision."
                        in_collision = False
                        #libRender.standard_render(self.m)



                    #dss.world.skeletons[0].remove_all_collision_pairs()

                    #libRender.standard_render(self.m)
                    dss.world.reset()
                    dss.world.destroy()

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
        np.save(self.filepath_prefix+"/data/init_poses/slp/valid_shape_pose_vol_"+some_subject+"_"+gender+"_"+str(len(shape_pose_vol_list))+".npy", np.array(shape_pose_vol_list))




    def fix_dataset(self, gender, num_data, filepath_prefix):


        filename = filepath_prefix+"/data/init_poses/elbow_under/valid_shape_pose_vol_"+gender+"_"+str(num_data)+".npy"

        old_pose_list = np.load(filename, allow_pickle=True).tolist()


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

                m, capsules, joint2name, rots0 = generator.map_shuffled_yash_to_smpl_angles(shift_ud, alter_angles = True)

                #print "stepping", m.pose
                dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture = 'lay', stiffness=None, filepath_prefix=self.filepath_prefix, add_floor = False)
                #print "stepping", m.pose

                invalid_pose = False
                #run a step to check for collisions
                dss.run_sim_step()

                dss.world.check_collision()

                print dss.world.collision_result.contact_sets
                if len(dss.world.collision_result.contacted_bodies) != 0:
                    for contact_set in dss.world.collision_result.contact_sets:
                        if contact_set[0] in contact_check_bns or contact_set[1] in contact_check_bns: #consider removing spine 3 and upper legs
                            if contact_set in contact_exceptions:
                                print "contact set is in exceptions"
                                pass

                            else:
                                #print "one of the limbs in contact"
                                print contact_set
                                #dss.run_simulation(1)

                                print "resampling pose from the same shape, invalid pose"
                                #dss.run_simulation(10000)
                                #libRender.standard_render(self.m)
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
        np.save(self.filepath_prefix+"/data/init_poses/valid_shape_pose_vol_"+gender+"_"+str(len(old_pose_list))+"_side_up.npy", np.array(shape_pose_vol_list))


    def doublecheck_prechecked_list(self, gender, filename):
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

            dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture='lay', stiffness=None,
                                            shiftSIDE=shape_pose_vol[4], shiftUD=shape_pose_vol[5],
                                            filepath_prefix=self.filepath_prefix, add_floor = False)
            dss.run_sim_step()
            dss.world.destroy()
        print np.mean(left_abd) #.15 .16
        print np.mean(right_abd) #-.18 -.13

                # print "one of the limbs in contact"
            #print dss.world.collision_result.contact_sets

    def graph_angles(self):

        hip0_L_list = []
        hip0_R_list = []
        hip1_L_list = []
        hip1_R_list = []
        hip2_L_list = []
        hip2_R_list = []

        knee0_L_list = []
        knee0_R_list = []

        should00_L_list = []
        should00_R_list = []
        should01_L_list = []
        should01_R_list = []
        should02_L_list = []
        should02_R_list = []

        should10_L_list = []
        should10_R_list = []
        should11_L_list = []
        should11_R_list = []
        should12_L_list = []
        should12_R_list = []

        elbow1_L_list = []
        elbow1_R_list = []
        for i in range(51,103):
            if i == 7: continue
            some_subject = '%05d' % (i)

            for pose_num in range(1, 46):#45 pose per participant.

                #here load some subjects joint angle data within danaLab and found by SMPLIFY

                try:
                    original_pose_data = load_pickle('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                    #print(original_pose_data)
                    original_pose = original_pose_data['pose']

                    if original_pose[3] > 0.4:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    if original_pose[6] > 0.4:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    if original_pose[15] < -0.05:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[12] < -0.05:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[55] > 0.2:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[58] < -0.2:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif  original_pose[55] < -3:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif  original_pose[58] > 3:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[49] > 0.5:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass
                    elif original_pose[52] < -0.5:
                        print('/home/henry/git/smplify_public/output_'+some_subject+'/%04d.pkl' % (pose_num))
                        pass


                    else:
                        hip0_L_list.append(original_pose[3])
                        hip1_L_list.append(original_pose[4])
                        hip2_L_list.append(original_pose[5])
                        hip0_R_list.append(original_pose[6])
                        hip1_R_list.append(-original_pose[7])
                        hip2_R_list.append(-original_pose[8])
                        knee0_L_list.append(original_pose[12])
                        knee0_R_list.append(original_pose[15])

                        should00_L_list.append(original_pose[39])
                        should01_L_list.append(original_pose[40])
                        should02_L_list.append(original_pose[41])
                        should00_R_list.append(original_pose[42])
                        should01_R_list.append(-original_pose[43])
                        should02_R_list.append(-original_pose[44])

                        should10_L_list.append(original_pose[48])
                        should11_L_list.append(original_pose[49])
                        should12_L_list.append(original_pose[50])
                        should10_R_list.append(original_pose[51])
                        should11_R_list.append(-original_pose[52])
                        should12_R_list.append(-original_pose[53])

                        elbow1_L_list.append(original_pose[55])
                        elbow1_R_list.append(-original_pose[58])



                except:
                    pass
                    #print('cannot load ', 'output_'+some_subject+'/%04d.pkl' % (pose_num))

        print(len(hip0_L_list))

        import matplotlib.pyplot as plt

        plt.plot(np.ones(len(hip0_L_list)), hip0_L_list, 'k.')
        plt.plot(np.ones(len(hip0_R_list))*2, hip0_R_list, 'k.')
        print('hip0 minmax', np.min([hip0_L_list+hip0_R_list]), np.max([hip0_L_list+hip0_R_list]))
        plt.plot(np.ones(len(hip1_L_list))*3, hip1_L_list, 'k.')
        plt.plot(np.ones(len(hip1_R_list))*4, hip1_R_list, 'k.')
        print('hip1 minmax', np.min([hip1_L_list+hip1_R_list]), np.max([hip1_L_list+hip1_R_list]))
        plt.plot(np.ones(len(hip2_L_list))*5, hip2_L_list, 'k.')
        plt.plot(np.ones(len(hip2_R_list))*6, hip2_R_list, 'k.')
        print('hip2 minmax', np.min([hip2_L_list+hip2_R_list]), np.max([hip2_L_list+hip2_R_list]))

        plt.plot(np.ones(len(knee0_L_list))*7, knee0_L_list, 'b.')
        plt.plot(np.ones(len(knee0_R_list))*8, knee0_R_list, 'b.')
        print('knee minmax', np.min([knee0_L_list+knee0_R_list]), np.max([knee0_L_list+knee0_R_list]))

        plt.plot(np.ones(len(should00_L_list))*9, should00_L_list, 'r.')
        plt.plot(np.ones(len(should00_R_list))*10, should00_R_list, 'r.')
        print('should00 minmax', np.min([should00_L_list+should00_R_list]), np.max([should00_L_list+should00_R_list]))
        plt.plot(np.ones(len(should01_L_list))*11, should01_L_list, 'r.')
        plt.plot(np.ones(len(should01_R_list))*12, should01_R_list, 'r.')
        print('should01 minmax', np.min([should01_L_list+should01_R_list]), np.max([should01_L_list+should01_R_list]))
        plt.plot(np.ones(len(should02_L_list))*13, should02_L_list, 'r.')
        plt.plot(np.ones(len(should02_R_list))*14, should02_R_list, 'r.')
        print('should02 minmax', np.min([should02_L_list+should02_R_list]), np.max([should02_L_list+should02_R_list]))

        plt.plot(np.ones(len(should10_L_list))*15, should10_L_list, 'r.')
        plt.plot(np.ones(len(should10_R_list))*16, should10_R_list, 'r.')
        print('should10 minmax', np.min([should10_L_list+should10_R_list]), np.max([should10_L_list+should10_R_list]))
        plt.plot(np.ones(len(should11_L_list))*17, should11_L_list, 'r.')
        plt.plot(np.ones(len(should11_R_list))*18, should11_R_list, 'r.')
        print('should11 minmax', np.min([should11_L_list+should11_R_list]), np.max([should11_L_list+should11_R_list]))
        plt.plot(np.ones(len(should12_L_list))*19, should12_L_list, 'r.')
        plt.plot(np.ones(len(should12_R_list))*20, should12_R_list, 'r.')
        print('should12 minmax', np.min([should12_L_list+should12_R_list]), np.max([should12_L_list+should12_R_list]))

        plt.plot(np.ones(len(elbow1_L_list))*21, elbow1_L_list, 'g.')
        plt.plot(np.ones(len(elbow1_R_list))*22, elbow1_R_list, 'g.')
        print('elbow minmax', np.min([elbow1_L_list+elbow1_R_list]), np.max([elbow1_L_list+elbow1_R_list]))

        plt.grid()
        plt.show()

    def resave_individual_files(self):

        import os


        #starting_subj = 51
        for starting_subj in [51, 61, 71, 81, 91]:

            posture_list = [["lay", 0], ["ls", 15],['rs', 30]]
            for posture in posture_list:

                for gender in ["f", "m"]:

                    new_valid_shape_pose_vol_list = []

                    #for i in range(51,103):
                    for i in range(starting_subj,starting_subj+10):
                        if i == 7: continue
                        some_subject = '%05d' % (i)


                        onlyfiles = next(os.walk('/home/henry/git/smplify_public/output_'+some_subject))[2]
                        number_files = str(int(len(onlyfiles)*15/2))

                        subject_new_samples = np.load(self.filepath_prefix + "/data/init_poses/slp/orig_bins/valid_shape_pose_vol_" +some_subject+ "_"+gender + "_"+number_files+".npy", allow_pickle=True)
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
                                    for j in range(15):
                                        new_valid_shape_pose_vol_list.append(subject_new_samples[within_list_pose_ct+j])

                                        print(some_subject, 'appending: ', pose_num, '  within list ct:', within_list_pose_ct+j)

                                within_list_pose_ct += 15

                                #here make an if statement

                            except:
                                #pass
                                print('pose ',pose_num,' is missing for subject:', some_subject)

                        print('poses within: ', within_list_pose_ct, '    poses appended: ', len(new_valid_shape_pose_vol_list))

                    np.save(self.filepath_prefix + "/data/init_poses/slp/valid_shape_pose_vol_" + posture[0] + "_" + gender + "_" +str(starting_subj) + "to" +str(starting_subj+9) + "_" +  str(len(new_valid_shape_pose_vol_list)) + ".npy", new_valid_shape_pose_vol_list)




    def fix_root(self):

        import os



        for gender in ["f", "m"]:

            #for i in range(51,103):
            for i in range(52, 103):
                if i == 7: continue
                some_subject = '%05d' % (i)


                onlyfiles = next(os.walk('/home/henry/git/smplify_public/output_'+some_subject))[2]
                print(len(onlyfiles))
                number_files = str(int(len(onlyfiles)*15/2))

                new_valid_shape_pose_vol_list = []

                try:
                    subject_new_samples = np.load(self.filepath_prefix + "/data/init_poses/slp/orig_bins/valid_shape_pose_vol_" +some_subject+ "_"+gender + "_"+number_files+".npy", allow_pickle=True)
                except:
                    continue

                print(len(subject_new_samples))

                #for pose_num in range(1, 46):#45 pose per participant.
                for pose_num in range(0, 45):#45 pose per participant.

                    #here load some subjects joint angle data within danaLab and found by SMPLIFY
                    try:
                        subject_new_samples[pose_num][2][1] = -float(subject_new_samples[pose_num][2][1])

                    except:
                        #pass
                        print(i,pose_num,' is missing for subject:', some_subject)


                np.save(self.filepath_prefix + "/data/init_poses/slp/orig_bins/valid_shape_pose_vol_" +some_subject+ "_"+gender + "_"+number_files+".npy", subject_new_samples)




if __name__ == "__main__":


    '''some_subject = "00076"
    gender = "m"

    valid_shape_pos = np.load('/home/henry/data/init_poses/slp/valid_shape_pose_vol_' + some_subject + "_" + gender + '_1260.npy', allow_pickle=True)
    print(len(valid_shape_pos), "LENGTH")
    valid_shape_pos_half = []
    for i in range(len(valid_shape_pos)):
        if i%2 == 0:
            pass
        else:
            valid_shape_pos_half.append(valid_shape_pos[i])
    print(len(valid_shape_pos_half))
    np.save("/home/henry/data/init_poses/slp/valid_shape_pose_vol_" + some_subject + "_" + gender + "_" + str(
        len(valid_shape_pos_half)) + ".npy", np.array(valid_shape_pos_half))'''







    some_subject = '00017'
    filepath_prefix = "/home/henry"



    phys_arr = np.load('../../data/SLP/danaLab/physiqueData.npy')
    phys_arr[:, [2, 0]] = phys_arr[:, [0, 2]]
    gender_bin = phys_arr[int(some_subject) - 1][2]
    if int(gender_bin) == 1:
        gender = "f"
    else:
        gender = "m"

    #DATASET_CREATE_TYPE = 1



    generator = GeneratePose(gender, filepath_prefix)

    generator.graph_angles()

    #generator.resave_individual_files()
    #generator.fix_root()
    #generator.generate_dataset(gender, some_subject = some_subject, num_samp_per_slp_pose = 15)
    #generator.fix_dataset(gender = "m", num_data = 3000, filepath_prefix = filepath_prefix)
    #generator.doublecheck_prechecked_list(gender, filepath_prefix+"/data/init_poses/valid_shape_pose_"+gender+"_"+str(num_data)+".npy")

