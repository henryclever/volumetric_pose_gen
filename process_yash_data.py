import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model

import lib_visualization as libVisualization
import lib_kinematics as libKinematics

#ROS
#import rospy
#import tf

import tensorflow as tensorflow
import cPickle as pickle
import random


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL



class ProcessYashData():
    def __init__(self):
        self.dependencies = {}
        for side in ['r', 'l']:
            self.dependencies['exclude_'+side+'G_ext'] = [side+'G_yaw', side+'G_abd', side+'K']
            self.dependencies['exclude_'+side+'G_abd'] = [side+'G_ext', side+'G_yaw', side+'K']
            self.dependencies['exclude_'+side+'G_yaw'] = [side+'G_ext', side+'G_abd', side+'K']
            self.dependencies['exclude_'+side+'K'] = [side+'G_ext', side+'G_abd', side+'G_yaw']

            self.dependencies['exclude_'+side+'G_ext_'+side+'G_abd'] = [side+'G_yaw', side+'K']
            self.dependencies['exclude_'+side+'G_ext_'+side+'G_yaw'] = [side+'G_abd', side+'K']
            self.dependencies['exclude_'+side+'G_ext_'+side+'K'] = [side+'G_yaw', side+'G_abd']
            self.dependencies['exclude_'+side+'G_abd_'+side+'G_ext'] = [side+'G_yaw', side+'K']
            self.dependencies['exclude_'+side+'G_abd_'+side+'G_yaw'] = [side+'G_ext', side+'K']
            self.dependencies['exclude_'+side+'G_abd_'+side+'K'] = [side+'G_ext', side+'G_yaw']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'G_ext'] = [side+'G_abd', side+'K']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'G_abd'] = [side+'G_ext', side+'K']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'K'] = [side+'G_ext', side+'G_abd']
            self.dependencies['exclude_'+side+'K_'+side+'G_ext'] = [side+'G_abd', side+'G_yaw']
            self.dependencies['exclude_'+side+'K_'+side+'G_abd'] = [side+'G_ext', side+'G_yaw']
            self.dependencies['exclude_'+side+'K_'+side+'G_yaw'] = [side+'G_ext', side+'G_abd']

            self.dependencies['exclude_'+side+'G_ext_'+side+'G_abd_'+side+'G_yaw'] = [side+'K']
            self.dependencies['exclude_'+side+'G_ext_'+side+'G_abd_'+side+'K'] = [side+'G_yaw']
            self.dependencies['exclude_'+side+'G_ext_'+side+'G_yaw_'+side+'G_abd'] = [side+'K']
            self.dependencies['exclude_'+side+'G_ext_'+side+'G_yaw_'+side+'K'] = [side+'G_abd']
            self.dependencies['exclude_'+side+'G_ext_'+side+'K_'+side+'G_yaw'] = [side+'G_abd']
            self.dependencies['exclude_'+side+'G_ext_'+side+'K_'+side+'G_abd'] = [side+'G_yaw']
            self.dependencies['exclude_'+side+'G_abd_'+side+'G_ext_'+side+'G_yaw'] = [side+'K']
            self.dependencies['exclude_'+side+'G_abd_'+side+'G_ext_'+side+'K'] = [side+'G_yaw']
            self.dependencies['exclude_'+side+'G_abd_'+side+'G_yaw_'+side+'G_ext'] = [side+'K']
            self.dependencies['exclude_'+side+'G_abd_'+side+'G_yaw_'+side+'K'] = [side+'G_ext']
            self.dependencies['exclude_'+side+'G_abd_'+side+'K_'+side+'G_ext'] = [side+'G_yaw']
            self.dependencies['exclude_'+side+'G_abd_'+side+'K_'+side+'G_yaw'] = [side+'G_ext']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'G_ext_'+side+'K'] = [side+'G_abd']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'G_ext_'+side+'G_abd'] = [side+'K']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'G_abd_'+side+'K'] = [side+'G_ext']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'G_abd_'+side+'G_ext'] = [side+'K']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'K_'+side+'G_ext'] = [side+'G_abd']
            self.dependencies['exclude_'+side+'G_yaw_'+side+'K_'+side+'G_abd'] = [side+'G_ext']
            self.dependencies['exclude_'+side+'K_'+side+'G_ext_'+side+'G_abd'] = [side+'G_yaw']
            self.dependencies['exclude_'+side+'K_'+side+'G_ext_'+side+'G_yaw'] = [side+'G_abd']
            self.dependencies['exclude_'+side+'K_'+side+'G_abd_'+side+'G_ext'] = [side+'G_yaw']
            self.dependencies['exclude_'+side+'K_'+side+'G_abd_'+side+'G_yaw'] = [side+'G_ext']
            self.dependencies['exclude_'+side+'K_'+side+'G_yaw_'+side+'G_ext'] = [side+'G_abd']
            self.dependencies['exclude_'+side+'K_'+side+'G_yaw_'+side+'G_abd'] = [side+'G_ext']


            self.dependencies['exclude_'+side+'S_roll'] = [side+'S_yaw', side+'S_pitch', side+'E']
            self.dependencies['exclude_'+side+'S_pitch'] = [side+'S_roll', side+'S_yaw', side+'E']
            self.dependencies['exclude_'+side+'S_yaw'] = [side+'S_roll', side+'S_pitch', side+'E']
            self.dependencies['exclude_'+side+'E'] = [side+'S_roll', side+'S_pitch', side+'S_yaw']

            self.dependencies['exclude_'+side+'S_roll_'+side+'S_pitch'] = [side+'S_yaw', side+'E']
            self.dependencies['exclude_'+side+'S_roll_'+side+'S_yaw'] = [side+'S_pitch', side+'E']
            self.dependencies['exclude_'+side+'S_roll_'+side+'E'] = [side+'S_yaw', side+'S_pitch']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'S_roll'] = [side+'S_yaw', side+'E']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'S_yaw'] = [side+'S_roll', side+'E']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'E'] = [side+'S_roll', side+'S_yaw']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'S_roll'] = [side+'S_pitch', side+'E']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'S_pitch'] = [side+'S_roll', side+'E']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'E'] = [side+'S_roll', side+'S_pitch']
            self.dependencies['exclude_'+side+'E_'+side+'S_roll'] = [side+'S_pitch', side+'S_yaw']
            self.dependencies['exclude_'+side+'E_'+side+'S_pitch'] = [side+'S_roll', side+'S_yaw']
            self.dependencies['exclude_'+side+'E_'+side+'S_yaw'] = [side+'S_roll', side+'S_pitch']

            self.dependencies['exclude_'+side+'S_roll_'+side+'S_pitch_'+side+'S_yaw'] = [side+'E']
            self.dependencies['exclude_'+side+'S_roll_'+side+'S_pitch_'+side+'E'] = [side+'S_yaw']
            self.dependencies['exclude_'+side+'S_roll_'+side+'S_yaw_'+side+'S_pitch'] = [side+'E']
            self.dependencies['exclude_'+side+'S_roll_'+side+'S_yaw_'+side+'E'] = [side+'S_pitch']
            self.dependencies['exclude_'+side+'S_roll_'+side+'E_'+side+'S_yaw'] = [side+'S_pitch']
            self.dependencies['exclude_'+side+'S_roll_'+side+'E_'+side+'S_pitch'] = [side+'S_yaw']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'S_roll_'+side+'S_yaw'] = [side+'E']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'S_roll_'+side+'E'] = [side+'S_yaw']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'S_yaw_'+side+'S_roll'] = [side+'E']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'S_yaw_'+side+'E'] = [side+'S_roll']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'E_'+side+'S_roll'] = [side+'S_yaw']
            self.dependencies['exclude_'+side+'S_pitch_'+side+'E_'+side+'S_yaw'] = [side+'S_roll']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'S_roll_'+side+'E'] = [side+'S_pitch']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'S_roll_'+side+'S_pitch'] = [side+'E']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'S_pitch_'+side+'E'] = [side+'S_roll']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'S_pitch_'+side+'S_roll'] = [side+'E']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'E_'+side+'S_roll'] = [side+'S_pitch']
            self.dependencies['exclude_'+side+'S_yaw_'+side+'E_'+side+'S_pitch'] = [side+'S_roll']
            self.dependencies['exclude_'+side+'E_'+side+'S_roll_'+side+'S_pitch'] = [side+'S_yaw']
            self.dependencies['exclude_'+side+'E_'+side+'S_roll_'+side+'S_yaw'] = [side+'S_pitch']
            self.dependencies['exclude_'+side+'E_'+side+'S_pitch_'+side+'S_roll'] = [side+'S_yaw']
            self.dependencies['exclude_'+side+'E_'+side+'S_pitch_'+side+'S_yaw'] = [side+'S_roll']
            self.dependencies['exclude_'+side+'E_'+side+'S_yaw_'+side+'S_roll'] = [side+'S_pitch']
            self.dependencies['exclude_'+side+'E_'+side+'S_yaw_'+side+'S_pitch'] = [side+'S_roll']




    def solve_ik_tree_yashdata(self, filename, subject, verbose = True):

        mat_transform_file = "/media/henry/multimodal_data_2/pressure_mat_pose_data/mat_axes.p"

        bedangle=0.

        jointLimbFiller = libKinematics.JointLimbFiller()

        with open(filename, 'rb') as fp:
            data = pickle.load(fp)

        with open(mat_transform_file, 'rb') as mp:
            [p_world_mat, R_world_mat] = pickle.load(mp)

        print("*************************** SOLVING IK TREE ON YASH DATA ****************************")
        print len(data)
        new_data = []

        for entry in data:
            print len(new_data)
            p_mat, targets_raw, other = entry
            angles = {}

            targets = libKinematics.world_to_mat(targets_raw, p_world_mat, R_world_mat)

            lengths, pseudotargets = jointLimbFiller.get_lengths_pseudotargets(targets, subject, bedangle)

            Jtc = np.concatenate((np.array(targets-targets[1, :]), np.array(pseudotargets-targets[1,:])), axis = 0)
            #print Jtc
            Jtc_origins = np.copy(Jtc)
            Jtc_origins[0, 0] = Jtc[10, 0]
            Jtc_origins[0, 1] = Jtc[10, 1] + lengths['neck_head']
            Jtc_origins[0, 2] = Jtc[10, 2]
            Jtc_origins[2, 0] = Jtc[11, 0] - lengths['r_shoulder_elbow']
            Jtc_origins[2, 1:] = Jtc[11, 1:]
            Jtc_origins[3, 0] = Jtc[12, 0] + lengths['l_shoulder_elbow']
            Jtc_origins[3, 1:] = Jtc[12, 1:]
            Jtc_origins[4, 0] = Jtc_origins[2, 0] - lengths['r_elbow_wrist']
            Jtc_origins[4, 1:] = Jtc_origins[2, 1:]
            Jtc_origins[5, 0] = Jtc_origins[3, 0] + lengths['l_elbow_wrist']
            Jtc_origins[5, 1:] = Jtc_origins[3, 1:]
            Jtc_origins[6, 0] = Jtc[13, 0]
            Jtc_origins[6, 1] = Jtc[13, 1] - lengths['r_glute_knee']
            Jtc_origins[6, 2] = Jtc[13, 2]
            Jtc_origins[7, 0] = Jtc[14, 0]
            Jtc_origins[7, 1] = Jtc[14, 1] - lengths['l_glute_knee']
            Jtc_origins[7, 2] = Jtc[14, 2]
            Jtc_origins[8, 0] = Jtc_origins[6, 0]
            Jtc_origins[8, 1] = Jtc_origins[6, 1] - lengths['r_knee_ankle']
            Jtc_origins[8, 2] = Jtc_origins[6, 2]
            Jtc_origins[9, 0] = Jtc_origins[7, 0]
            Jtc_origins[9, 1] = Jtc_origins[7, 1] - lengths['l_knee_ankle']
            Jtc_origins[9, 2] = Jtc_origins[7, 2]

            #print Jtc_origins

            # grab the joint positions
            r_leg_pos_origins = np.array([np.array(Jtc_origins[1, :]), np.array(Jtc_origins[13, :]),
                                          np.array(Jtc_origins[6, :]), np.array(Jtc_origins[8, :])])
            r_leg_pos_current = np.array([np.array(Jtc[1, :]), np.array(Jtc[13, :]),
                                          np.array(Jtc[6, :]), np.array(Jtc[8, :])])

            # get the IKPY solution
            r_omega_hip, r_knee_chain, IK_RK, r_ankle_chain, IK_RA = libKinematics.ikpy_leg(r_leg_pos_origins,
                                                                                            r_leg_pos_current)

            # get the RPH values
            r_hip_angles = [IK_RK[1], IK_RA[2], IK_RK[2]]
            r_knee_angles = [IK_RA[4], 0, 0]

            if verbose == True:
                print IK_RK, 'IK_RK'
                print IK_RA, "IK_RA"
                print "right hip Angle-axis est:", r_omega_hip
                print "right hip Euler est: ", r_hip_angles
                # print "right knee Angle-axis est:", r_omega_knee
                print "right knee Euler est: ", r_knee_angles
            angles['r_hip_angle_axis'] = r_omega_hip
            angles['r_hip_euler'] = r_hip_angles
            angles['r_knee_angle_axis'] = r_knee_angles


            # grab the joint positions
            l_leg_pos_origins = np.array([np.array(Jtc_origins[1, :]), np.array(Jtc_origins[14, :]),
                                          np.array(Jtc_origins[7, :]), np.array(Jtc_origins[9, :])])
            l_leg_pos_current = np.array([np.array(Jtc[1, :]), np.array(Jtc[14, :]),
                                          np.array(Jtc[7, :]), np.array(Jtc[9, :])])

            # get the IKPY solution
            l_omega_hip, l_knee_chain, IK_LK, l_ankle_chain, IK_LA = libKinematics.ikpy_leg(l_leg_pos_origins,
                                                                                            l_leg_pos_current)

            # get the RPH values
            l_hip_angles = [IK_LK[1], IK_LA[2], IK_LK[2]]
            l_knee_angles = [IK_LA[4], 0, 0]

            if verbose == True:
                print IK_LK, 'IK_LK'
                print IK_LA, "IK_LA"
                print "left hip Angle-axis est:", l_omega_hip
                print "left hip Euler est: ", l_hip_angles
                # print "right knee Angle-axis est:", r_omega_knee
                print "right knee Euler est: ", l_knee_angles
            angles['l_hip_angle_axis'] = l_omega_hip
            angles['l_hip_euler'] = l_hip_angles
            angles['l_knee_angle_axis'] = l_knee_angles

            # grab the joint positions
            r_arm_pos_origins = np.array([np.array(Jtc_origins[1, :]), np.array(Jtc_origins[10, :]), np.array(Jtc_origins[11, :]),
                                          np.array(Jtc_origins[2, :]), np.array(Jtc_origins[4, :])])
            r_arm_pos_current = np.array([np.array(Jtc[1, :]), np.array(Jtc[10, :]), np.array(Jtc[11, :]),
                                          np.array(Jtc[2, :]), np.array(Jtc[4, :])])

            # get the IK solution
            r_omega_shoulder, r_elbow_chain, IK_RE, r_wrist_chain, IK_RW = libKinematics.ikpy_arm(r_arm_pos_origins,
                                                                                                  r_arm_pos_current)

            # get the RPH values
            r_shoulder_angles = [IK_RW[2], IK_RE[2], IK_RE[3]]
            r_elbow_angles = [0, IK_RW[5], 0]

            if verbose == True:
                print IK_RE, 'IK_RE'
                print IK_RW, 'IK_RW'
                print "right shoulder Angle-axis est: ", r_omega_shoulder
                print "right shoulder Euler est: ", r_shoulder_angles
                print "right elbow Euler est: ", r_elbow_angles
            angles['r_shoulder_angle_axis'] = r_omega_shoulder
            angles['r_shoulder_euler'] = r_shoulder_angles
            angles['r_elbow_angle_axis'] = r_elbow_angles

            # grab the joint positions
            l_arm_pos_origins = np.array([np.array(Jtc_origins[1, :]), np.array(Jtc_origins[10, :]), np.array(Jtc_origins[12, :]),
                                          np.array(Jtc_origins[3, :]), np.array(Jtc_origins[5, :])])
            l_arm_pos_current = np.array([np.array(Jtc[1, :]), np.array(Jtc[10, :]), np.array(Jtc[12, :]),
                                          np.array(Jtc[3, :]), np.array(Jtc[5, :])])

            # get the IK solution
            l_omega_shoulder, l_elbow_chain, IK_LE, l_wrist_chain, IK_LW = libKinematics.ikpy_arm(l_arm_pos_origins,
                                                                                                  l_arm_pos_current)

            # get the RPH values
            l_shoulder_angles = [IK_LW[2], IK_LE[2], IK_LE[3]]
            l_elbow_angles = [0, IK_LW[5], 0]

            if verbose == True:
                print IK_LE, 'IK_LE'
                print IK_LW, 'IK_LW'
                print "left shoulder Angle-axis est: ", l_omega_shoulder
                print "left shoulder Euler est: ", l_shoulder_angles
                print "left elbow Euler est: ", l_elbow_angles
            angles['l_shoulder_angle_axis'] = l_omega_shoulder
            angles['l_shoulder_euler'] = l_shoulder_angles
            angles['l_elbow_angle_axis'] = l_elbow_angles

            plot_first_joint = False
            plot_all_joints = True
            if plot_first_joint:
                r_knee_chain.plot(IK_RK, self.ax)
                l_knee_chain.plot(IK_LK, self.ax)
                r_elbow_chain.plot(IK_RE, self.ax)
                l_elbow_chain.plot(IK_LE, self.ax)
                #head_chain.plot(IK_H, self.ax)

            if plot_all_joints:
                r_ankle_chain.plot(IK_RA, self.ax)
                l_ankle_chain.plot(IK_LA, self.ax)
                r_wrist_chain.plot(IK_RW, self.ax)
                l_wrist_chain.plot(IK_LW, self.ax)
                #head_chain.plot(IK_H, self.ax)


            #self.ax.plot(Jtc_origins[:, 0], Jtc_origins[:, 1], Jtc_origins[:, 2], markerfacecolor='k', markeredgecolor='k', marker='o',  markersize=5, alpha=0.5)
            #self.ax.plot(Jtc[:, 0], Jtc[:, 1], Jtc[:, 2], markerfacecolor='k', markeredgecolor='k', marker='o',  markersize=5, alpha=0.5)
            #self.ax.set_xlim(-1, 1)
            #self.ax.set_ylim(-1, 1)
            #self.ax.set_zlim(-1, 1)
            #plt.show()

            new_entry = angles
            new_data.append(new_entry)


        return new_data


    def save_yash_data_with_angles(self):
        movements = ['LL', 'RL', 'LH1', 'LH2', 'LH3', 'RH1', 'RH2', 'RH3']
        subjects = ['1YJAG']#['WE2SZ', 'WFGW9', 'WM9KJ', 'ZV7TE']

        for subject in subjects:
            for movement in movements:

                filename = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/" + movement + ".p"
                filename_save = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/" + movement + "_angles.p"

                new_data = generator.solve_ik_tree_yashdata(filename, subject, verbose = False)

                print len(new_data), 'length of new data'
                with open(filename_save, 'wb') as fp:
                    pickle.dump(new_data, fp)
                print 'done saving ', filename_save




    def create_all_angle_lists(self, all_angle_types, all_leg):
        print 'creating leg all angle lists'
        for angle_type in all_angle_types:
            all_leg[angle_type] = []
            for segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                for dependent_angle in self.dependencies['exclude_'+angle_type]:
                    all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle] = []
                    for next_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                        for dependent_angle2 in self.dependencies['exclude_'+angle_type+'_'+dependent_angle]:
                            all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2] = []
                            for last_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                                for dependent_angle3 in self.dependencies['exclude_'+angle_type+'_'+dependent_angle+'_'+dependent_angle2]:
                                    all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3] = []
        return all_leg


    def fill_initial_leg_all_angle_lists(self, subjects, movements, all_leg, r_or_l):
        print 'filling initial leg all angle lists'
        for subject in subjects:
            for movement in movements:
                print "subject: ", subject, " movement: ", movement
                filename = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/" + movement + "_angles.p"
                with open(filename, 'rb') as fp:
                    angles_data = pickle.load(fp)
                for entry in angles_data:
                    if entry[r_or_l+'_knee_angle_axis'][0] > 0.0:
                        all_leg[r_or_l+'G_ext'].append(entry[r_or_l+'_hip_angle_axis'][0])
                        all_leg[r_or_l+'G_yaw'].append(entry[r_or_l+'_hip_angle_axis'][1])
                        all_leg[r_or_l+'G_abd'].append(entry[r_or_l+'_hip_angle_axis'][2])
                        all_leg[r_or_l+'K'].append(entry[r_or_l+'_knee_angle_axis'][0])
        return all_leg

    def fill_initial_arm_all_angle_lists(self, subjects, movements, all_arm, r_or_l):
        print 'filling initial arm all angle lists'
        for subject in subjects:
            for movement in movements:
                print "subject: ", subject, " movement: ", movement
                filename = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/" + movement + "_angles.p"
                with open(filename, 'rb') as fp:
                    angles_data = pickle.load(fp)
                for entry in angles_data:
                    if r_or_l == 'r':
                        if entry[r_or_l+'_elbow_angle_axis'][1] > 0.0:
                            all_arm[r_or_l+'S_roll'].append(entry[r_or_l+'_shoulder_angle_axis'][0])
                            all_arm[r_or_l+'S_yaw'].append(entry[r_or_l+'_shoulder_angle_axis'][1])
                            all_arm[r_or_l+'S_pitch'].append(entry[r_or_l+'_shoulder_angle_axis'][2])
                            all_arm[r_or_l+'E'].append(entry[r_or_l+'_elbow_angle_axis'][1])
                    elif r_or_l == 'l':
                        if entry[r_or_l+'_elbow_angle_axis'][1] < 0.0:
                            all_arm[r_or_l+'S_roll'].append(entry[r_or_l+'_shoulder_angle_axis'][0])
                            all_arm[r_or_l+'S_yaw'].append(entry[r_or_l+'_shoulder_angle_axis'][1])
                            all_arm[r_or_l+'S_pitch'].append(entry[r_or_l+'_shoulder_angle_axis'][2])
                            all_arm[r_or_l+'E'].append(entry[r_or_l+'_elbow_angle_axis'][1])
        return all_arm


    def fill_initial_range_angle_lists(self, all_leg_angle_types, all_leg, select_leg):
        print 'filling initial leg range angle lists'
        for angle_type in all_leg_angle_types:
            print('min of',angle_type,': ', min(all_leg[angle_type]))
            print('max of',angle_type,': ', max(all_leg[angle_type]))
            select_leg[angle_type] = [min(all_leg[angle_type]), max(all_leg[angle_type]), len(all_leg[angle_type])]
        return select_leg

    def fill_conditioned_all_angle_lists(self, all_leg_angle_types, all_leg, select_leg):
        print 'building lists of conditioned ranges on leg'
        #Build lists of conditioned ranges on right leg
        for index in range(0, len(all_leg[all_leg_angle_types[0]])): #iterate over every angle in the set
            for segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]: #break up into ten segments
                for angle_type in all_leg_angle_types: #the angle we are first conditioning on
                    if all_leg[angle_type][index] > select_leg[angle_type][0]+(segment*0.01)*(select_leg[angle_type][1] - select_leg[angle_type][0]) and \
                        all_leg[angle_type][index] <= select_leg[angle_type][0]+(segment*0.01+0.1)* (select_leg[angle_type][1] - select_leg[angle_type][0]):
                        for dependent_angle in self.dependencies['exclude_'+angle_type]: #the next dependent angle
                            all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle].append(all_leg[dependent_angle][index])
                            for next_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                                if all_leg[dependent_angle][index] > select_leg[dependent_angle][0]+(next_segment * 0.01) * (select_leg[dependent_angle][1] - select_leg[dependent_angle][0]) and \
                                    all_leg[dependent_angle][index] <= select_leg[dependent_angle][0]+(next_segment * 0.01+0.1) * (select_leg[dependent_angle][1] - select_leg[dependent_angle][0]):
                                    for dependent_angle2 in self.dependencies['exclude_'+angle_type+'_'+dependent_angle]:
                                        all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2].append(all_leg[dependent_angle2][index])
                                        for last_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                                            if all_leg[dependent_angle2][index] > select_leg[dependent_angle2][0]+(last_segment * 0.01) * (select_leg[dependent_angle2][1] - select_leg[dependent_angle2][0]) and \
                                                all_leg[dependent_angle2][index] <= select_leg[dependent_angle2][0]+(last_segment * 0.01+0.1) * (select_leg[dependent_angle2][1] - select_leg[dependent_angle2][0]):
                                                for dependent_angle3 in self.dependencies['exclude_'+angle_type+'_'+dependent_angle+'_'+dependent_angle2]:
                                                    all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+ \
                                                        str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3].append(all_leg[dependent_angle3][index])
        return all_leg

    def fill_conditioned_range_angle_lists(self, all_leg_angle_types, all_leg, select_leg):
        print 'assigning conditioned mins and maxes on leg'
        #Assign the new mins and maxes to a file we will save
        for segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            for angle_type in all_leg_angle_types:  # the angle we are first conditioning on
                for dependent_angle in self.dependencies['exclude_'+angle_type]:
                    try: select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle] = \
                        [min(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle]),
                         max(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle]),
                         len(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle])]
                    except: select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle] = [None, None, 0]
                    for next_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                        for dependent_angle2 in self.dependencies['exclude_'+angle_type+'_'+dependent_angle]:
                            try: select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2] = \
                                    [min(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2]),
                                     max(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2]),
                                     len(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2])]
                            except: select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2] = [None, None, 0]

                            for last_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                                for dependent_angle3 in self.dependencies['exclude_'+angle_type+'_'+dependent_angle+'_'+dependent_angle2]:
                                    try: select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3] = \
                                            [min(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3]),
                                             max(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3]),
                                             len(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3])]
                                    except: select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3] = [None, None, 0]
        return select_leg

    def print_conditioned_range_tree(self, all_leg_angle_types, all_leg, select_leg):
        #print things out to check them
        for angle_type in all_leg_angle_types:  # the angle we are first conditioning on
            for dependent_angle in self.dependencies['exclude_'+angle_type]:
                print('________________________________________________________________________________________________')
                print(len(all_leg[angle_type]), angle_type, 'range: ',select_leg[angle_type])
                for segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                    print('____________________________________________')
                    print(angle_type+'_'+str(segment)+'_'+str(segment+10), len(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle]),
                        dependent_angle, 'range: ',  select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle])
                    for dependent_angle2 in self.dependencies['exclude_' + angle_type+'_'+dependent_angle]:
                        for next_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                            print('______________________')
                            print(angle_type+'_'+str(segment)+'_'+str(segment+10)+'_'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10),
                                len(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2]),
                                dependent_angle2, 'range: ',  select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2])
                            for dependent_angle3 in self.dependencies['exclude_' + angle_type + '_' + dependent_angle + '_' + dependent_angle2]:
                                for last_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                                    print(angle_type+'_'+str(segment)+'_'+str(segment+10)+'_'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'_'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10),
                                        len(all_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3]),
                                        dependent_angle3, 'range: ',  select_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3])




    def map_yash_to_axis_angle(self, verbose = True):

        subjects = ['40ESJ', '4ZZZQ', '5LDJG', 'A4G4Y','G55Q1','GF5Q3', 'GRTJK', 'RQCLC', 'TSQNA', 'TX887', 'WCNOM', 'WE2SZ', 'WFGW9', 'WM9KJ', 'ZV7TE']
        movements = ['LL', 'RL', 'LH1', 'LH2', 'LH3', 'RH1', 'RH2', 'RH3']


        select_r_leg = {}
        select_l_leg = {}
        select_r_arm = {}
        select_l_arm = {}
        all_r_leg = {}
        all_l_leg = {}
        all_r_arm = {}
        all_l_arm = {}


        all_r_leg_angle_types = ['rG_ext', 'rG_abd', 'rG_yaw', 'rK']
        all_l_leg_angle_types = ['lG_ext', 'lG_abd', 'lG_yaw', 'lK']
        all_r_arm_angle_types = ['rS_roll', 'rS_yaw', 'rS_pitch', 'rE']
        all_l_arm_angle_types = ['lS_roll', 'lS_yaw', 'lS_pitch', 'lE']


        all_r_leg = self.create_all_angle_lists(all_r_leg_angle_types, all_r_leg)
        all_l_leg = self.create_all_angle_lists(all_l_leg_angle_types, all_l_leg)
        all_r_arm = self.create_all_angle_lists(all_r_arm_angle_types, all_r_arm)
        all_l_arm = self.create_all_angle_lists(all_l_arm_angle_types, all_l_arm)


        all_r_leg = self.fill_initial_leg_all_angle_lists(subjects, movements, all_r_leg, 'r')
        all_l_leg = self.fill_initial_leg_all_angle_lists(subjects, movements, all_l_leg, 'l')
        all_r_arm = self.fill_initial_arm_all_angle_lists(subjects, movements, all_r_arm, 'r')
        all_l_arm = self.fill_initial_arm_all_angle_lists(subjects, movements, all_l_arm, 'l')

        select_r_leg = self.fill_initial_range_angle_lists(all_r_leg_angle_types, all_r_leg, select_r_leg)
        select_l_leg = self.fill_initial_range_angle_lists(all_l_leg_angle_types, all_l_leg, select_l_leg)
        select_r_arm = self.fill_initial_range_angle_lists(all_r_arm_angle_types, all_r_arm, select_r_arm)
        select_l_arm = self.fill_initial_range_angle_lists(all_l_arm_angle_types, all_l_arm, select_l_arm)

        all_r_leg = self.fill_conditioned_all_angle_lists(all_r_leg_angle_types, all_r_leg, select_r_leg)
        all_l_leg = self.fill_conditioned_all_angle_lists(all_l_leg_angle_types, all_l_leg, select_l_leg)
        all_r_arm = self.fill_conditioned_all_angle_lists(all_r_arm_angle_types, all_r_arm, select_r_arm)
        all_l_arm = self.fill_conditioned_all_angle_lists(all_l_arm_angle_types, all_l_arm, select_l_arm)


        select_r_leg = self.fill_conditioned_range_angle_lists(all_r_leg_angle_types, all_r_leg, select_r_leg)
        select_l_leg = self.fill_conditioned_range_angle_lists(all_l_leg_angle_types, all_l_leg, select_l_leg)
        select_r_arm = self.fill_conditioned_range_angle_lists(all_r_arm_angle_types, all_r_arm, select_r_arm)
        select_l_arm = self.fill_conditioned_range_angle_lists(all_l_arm_angle_types, all_l_arm, select_l_arm)

        #self.print_conditioned_range_tree(all_r_leg_angle_types, all_r_leg, select_r_leg)
        #self.print_conditioned_range_tree(all_l_leg_angle_types, all_l_leg, select_l_leg)


        filename_save = "/media/henry/multimodal_data_2/pressure_mat_pose_data/select_r_leg.p"
        with open(filename_save, 'wb') as fp:
            pickle.dump(select_r_leg, fp)
        print 'done saving ', filename_save

        filename_save = "/media/henry/multimodal_data_2/pressure_mat_pose_data/select_l_leg.p"
        with open(filename_save, 'wb') as fp:
            pickle.dump(select_l_leg, fp)
        print 'done saving ', filename_save

        filename_save = "/media/henry/multimodal_data_2/pressure_mat_pose_data/select_r_arm.p"
        with open(filename_save, 'wb') as fp:
            pickle.dump(select_r_arm, fp)
        print 'done saving ', filename_save

        filename_save = "/media/henry/multimodal_data_2/pressure_mat_pose_data/select_l_arm.p"
        with open(filename_save, 'wb') as fp:
            pickle.dump(select_l_arm, fp)
        print 'done saving ', filename_save


    def check_found_limits(self):

        filename = "/media/henry/multimodal_data_2/pressure_mat_pose_data/select_r_leg.p"

        with open(filename, 'rb') as fp:
            select_r_leg = pickle.load(fp)
            # print things out to check them


        #print things out to check them
        for angle_type in ['rG_ext', 'rG_abd', 'rG_yaw', 'rK']:  # the angle we are first conditioning on
            for dependent_angle in self.dependencies['exclude_'+angle_type]:
                print('________________________________________________________________________________________________')
                print(angle_type, 'range: ',select_r_leg[angle_type])
                for segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                    print('____________________________________________')
                    print(angle_type+'_'+str(segment)+'_'+str(segment+10), dependent_angle, 'range: ',
                        select_r_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle])
                    for dependent_angle2 in self.dependencies['exclude_' + angle_type+'_'+dependent_angle]:
                        for next_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                            print('______________________')
                            print(angle_type+'_'+str(segment)+'_'+str(segment+10)+'_'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10),
                                dependent_angle2, 'range: ',  select_r_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2])
                            for dependent_angle3 in self.dependencies['exclude_' + angle_type + '_' + dependent_angle + '_' + dependent_angle2]:
                                for last_segment in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
                                    print(angle_type+'_'+str(segment)+'_'+str(segment+10)+'_'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'_'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10),
                                        dependent_angle3, 'range: ',  select_r_leg[angle_type+'_'+str(segment)+'_'+str(segment+10)+'__'+dependent_angle+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+dependent_angle2+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+dependent_angle3])

    def round_down(self, num, divisor):
        return int(num - (num % divisor))

    def sample_angles(self, limb):
        filename = "/home/henry/git/volumetric_pose_gen/select_"+limb+".p"

        with open(filename, 'rb') as fp:
            select = pickle.load(fp)
            # print things out to check them

        if limb == 'r_leg':
            angles = ['rG_ext', 'rG_abd', 'rG_yaw', 'rK']
        elif limb == 'l_leg':
            angles = ['lG_ext', 'lG_abd', 'lG_yaw', 'lK']
        elif limb == 'r_arm':
            angles = ['rS_roll', 'rS_yaw', 'rS_pitch', 'rE']
        elif limb == 'l_arm':
            angles = ['lS_roll', 'lS_yaw', 'lS_pitch', 'lE']
        else:
            return "ERROR"

        angles = random.sample(angles, len(angles))


        selection = {}

        first_angle_select = angles.pop()
        second_angle_select = angles.pop()
        third_angle_select = angles.pop()
        fourth_angle_select = angles.pop()

        print first_angle_select, select[first_angle_select]

        while True:
            try:
                print first_angle_select, "min: ", select[first_angle_select][0], "max: ",  select[first_angle_select][1]
                selection[first_angle_select] = random.uniform(select[first_angle_select][0],  select[first_angle_select][1])

                segment = 100*(selection[first_angle_select] - select[first_angle_select][0])/(select[first_angle_select][1] -  select[first_angle_select][0])
                segment = self.round_down(segment, 10)
                print 'selected',first_angle_select,'value of',selection[first_angle_select],' a value within', str(segment), 'and', str(segment+10)


                print second_angle_select, select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select]

                while True:
                    try:
                        selection[second_angle_select] = random.uniform(select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select][0],
                                                                       select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select][1])

                        next_segment = 100*(selection[second_angle_select] - select[second_angle_select][0])/(select[second_angle_select][1] -  select[second_angle_select][0])
                        next_segment = self.round_down(next_segment, 10)
                        print 'selected',second_angle_select,'value of',selection[second_angle_select],' a value within', str(next_segment), 'and', str(next_segment+10)


                        print third_angle_select, select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+third_angle_select]
                        while True:
                            try:
                                selection[third_angle_select] = random.uniform(select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+third_angle_select][0],
                                                                               select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+third_angle_select][1])

                                last_segment = 100*(selection[third_angle_select] - select[third_angle_select][0])/(select[third_angle_select][1] -  select[third_angle_select][0])
                                last_segment = self.round_down(last_segment, 10)
                                print 'selected',third_angle_select,'value of',selection[third_angle_select],' a value within', str(last_segment), 'and', str(last_segment+10)




                                print fourth_angle_select, select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+third_angle_select+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+fourth_angle_select]
                                selection[fourth_angle_select] = random.uniform(select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+third_angle_select+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+fourth_angle_select][0],
                                                                               select[first_angle_select+'_'+str(segment)+'_'+str(segment+10)+'__'+second_angle_select+'_'+str(next_segment)+'_'+str(next_segment+10)+'__'+third_angle_select+'_'+str(last_segment)+'_'+str(last_segment+10)+'__'+fourth_angle_select][1])

                                last_last_segment = 100*(selection[fourth_angle_select] - select[fourth_angle_select][0])/(select[fourth_angle_select][1] -  select[fourth_angle_select][0])
                                last_last_segment = self.round_down(last_last_segment, 10)
                                print 'selected',fourth_angle_select,'value of',selection[fourth_angle_select],' a value within', str(last_last_segment), 'and', str(last_last_segment+10)
                                return selection
                            except:
                                print('repeating inner loop')
                                break
                        break
                    except:
                        print('repeating middle loop')
                        break
            except:
                pass
