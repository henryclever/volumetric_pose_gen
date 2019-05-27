import numpy as np
import random
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model

#volumetric pose gen libraries
import lib_visualization as libVisualization
import lib_kinematics as libKinematics
from process_yash_data import ProcessYashData
import dart_skel_sim

#ROS
try:
    import rospy
    import tf
except:
    pass

import tensorflow as tensorflow
import cPickle as pickle


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL



class GeneratePose():
    def __init__(self, sampling = "NORMAL", sigma = 0, one_side_range = 0, gender="m"):
        ## Load SMPL model (here we load the female model)
        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
        self.m = load_model(model_path)

        ## Assign random pose and shape parameters

        self.m.betas[:] = np.random.rand(self.m.betas.size) * .0
        #self.m.betas[5] = 20.

        self.m.pose[0] = 0 #pitch rotation of the person in space. 0 means the person is upside down facing back. pi is standing up facing forward
        self.m.pose[1] = 0 #roll of the person in space. -pi/2 means they are tilted to their right side
        self.m.pose[2] = 0#-np.pi/4 #yaw of the person in space, like turning around normal to the ground

        self.m.pose[3] = 0#-np.pi/4 #left hip extension (i.e. leg bends back for np.pi/2)
        self.m.pose[4] = 0#np.pi/8 #left leg yaw about hip, where np.pi/2 makes bowed leg
        self.m.pose[5] = 0.#np.pi/4 #left leg abduction (POS) /adduction (NEG)

        self.m.pose[6] = 0#-np.pi/4 #right hip extension (i.e. leg bends back for np.pi/2)
        self.m.pose[7] = 0#-np.pi/8
        self.m.pose[8] = 0.#-np.pi/4 #right leg abduction (NEG) /adduction (POS)

        self.m.pose[9] = 0 #bending of spine at hips. np.pi/2 means person bends down to touch the ground
        self.m.pose[10] = 0 #twisting of spine at hips. body above spine yaws normal to the ground
        self.m.pose[11] = 0 #bending of spine at hips. np.pi/2 means person bends down sideways to touch the ground 3

        self.m.pose[12] = 0#np.pi/4 #left knee extension. (i.e. knee bends back for np.pi/2)
        self.m.pose[13] = 0 #twisting of knee normal to ground. KEEP AT ZERO
        self.m.pose[14] = 0 #bending of knee sideways. KEEP AT ZERO

        self.m.pose[15] = 0#np.pi/4 #right knee extension (i.e. knee bends back for np.pi/2)

        self.m.pose[18] = 0 #bending at mid spine. makes person into a hunchback for positive values
        self.m.pose[19] = 0#twisting of midspine. body above midspine yaws normal to the ground
        self.m.pose[20] = 0 #bending of midspine, np.pi/2 means person bends down sideways to touch ground 6

        self.m.pose[21] = 0 #left ankle flexion/extension
        self.m.pose[22] = 0 #left ankle yaw about leg
        self.m.pose[23] = 0 #left ankle twist KEEP CLOSE TO ZERO

        self.m.pose[24] = 0 #right ankle flexion/extension
        self.m.pose[25] = 0 #right ankle yaw about leg
        self.m.pose[26] = 0#np.pi/4 #right ankle twist KEEP CLOSE TO ZERO

        self.m.pose[27] = 0 #bending at upperspine. makes person into a hunchback for positive values
        self.m.pose[28] = 0#twisting of upperspine. body above upperspine yaws normal to the ground
        self.m.pose[29] = 0 #bending of upperspine, np.pi/2 means person bends down sideways to touch ground 9

        self.m.pose[30] = 0 #flexion/extension of left ankle midpoint

        self.m.pose[33] = 0 #flexion/extension of right ankle midpoint

        self.m.pose[36] = 0#np.pi/2 #flexion/extension of neck. i.e. whiplash 12
        self.m.pose[37] = 0#-np.pi/2 #yaw of neck
        self.m.pose[38] = 0#np.pi/4  #tilt head side to side

        self.m.pose[39] = 0 #left inner shoulder roll
        self.m.pose[40] = 0 #left inner shoulder yaw, negative moves forward
        self.m.pose[41] = 0.#np.pi/4 #left inner shoulder pitch, positive moves up

        self.m.pose[42] = 0
        self.m.pose[43] = 0 #right inner shoulder yaw, positive moves forward
        self.m.pose[44] = 0.#-np.pi/4 #right inner shoulder pitch, positive moves down

        self.m.pose[45] = 0 #flexion/extension of head 15

        self.m.pose[48] = 0#-np.pi/4 #left outer shoulder roll
        self.m.pose[49] = 0#-np.pi/4
        self.m.pose[50] = 0#np.pi/4 #left outer shoulder pitch

        self.m.pose[51] = 0#-np.pi/4 #right outer shoulder roll
        self.m.pose[52] = 0#np.pi/4
        self.m.pose[53] = 0#-np.pi/4

        self.m.pose[54] = 0 #left elbow roll KEEP AT ZERO
        self.m.pose[55] = 0#np.pi/3 #left elbow flexion/extension. KEEP NEGATIVE
        self.m.pose[56] = 0 #left elbow KEEP AT ZERO

        self.m.pose[57] = 0
        self.m.pose[58] = 0#np.pi/4 #right elbow flexsion/extension KEEP POSITIVE

        self.m.pose[60] = 0 #left wrist roll

        self.m.pose[63] = 0 #right wrist roll
        #self.m.pose[65] = np.pi/5

        self.m.pose[66] = 0 #left hand roll

        self.m.pose[69] = 0 #right hand roll
        #self.m.pose[71] = np.pi/5 #right fist


        mu = 0



        for i in range(10):
            if sampling == "NORMAL":
                self.m.betas[i] = random.normalvariate(mu, sigma)
            elif sampling == "UNIFORM":
                self.m.betas[i]  = np.random.uniform(-one_side_range, one_side_range)
        #self.m.betas[0] = random.normalvariate(mu, sigma) #overall body size. more positive number makes smaller, negative makes larger with bigger belly
        #self.m.betas[1] = random.normalvariate(mu, sigma) #positive number makes person very skinny, negative makes fat
        #self.m.betas[2] = random.normalvariate(mu, sigma) #muscle mass. higher makes person less physically fit
        #self.m.betas[3] = random.normalvariate(mu, sigma) #proportion for upper vs lower bone lengths. more negative number makes legs much bigger than arms
        #self.m.betas[4] = random.normalvariate(mu, sigma) #neck. more negative seems to make neck longer and body more skinny
        #self.m.betas[5] = random.normalvariate(mu, sigma) #size of hips. larger means bigger hips
        #self.m.betas[6] = random.normalvariate(mu, sigma) #proportion of belly with respect to rest of the body. higher number is larger belly
        #self.m.betas[7] = random.normalvariate(mu, sigma)
        #self.m.betas[8] = random.normalvariate(-3, 3)
        #self.m.betas[9] = random.normalvariate(-3, 3)

        #print self.m.pose.shape
        #print self.m.pose, 'pose'
        #print self.m.betas, 'betas'




    def get_volumetric_surface_points_closest(self):
        print self.m.pose
        print self.m.betas
        print self.m.J_transformed
        vertices = np.array(self.m.r)
        print vertices[0:5, :]

        Torso_mid_joint = np.mean([np.array(self.m.J_transformed[3, :]), np.array(self.m.J_transformed[6, :])], axis = 0)
        #print np.array(self.m.J_transformed[3, :]), np.array(self.m.J_transformed[6, :])
        #find closest point to x and y positions i.e. the first two columns. Find higher of the z position i.e. the 3rd.
        print "TORSO: ", Torso_mid_joint
        dist_from_torso = list(vertices - Torso_mid_joint)
        euclid_dist_from_torso = np.square(dist_from_torso)
        euclid_dist_from_torso = list(np.sqrt(euclid_dist_from_torso[:, 0] + euclid_dist_from_torso[:, 1]))
        for i in range(10):
            print np.argmin(euclid_dist_from_torso), dist_from_torso[np.argmin(euclid_dist_from_torso)] + Torso_mid_joint
            euclid_dist_from_torso[np.argmin(euclid_dist_from_torso)] += 100.
            #torso is 1325


        head_mid_joint = np.array(self.m.J_transformed[15, :]) + np.array([0.0, 0.123, 0.0])
        #measure of man and woman, Dreyfuss:
        #50% male distance from top spine joint to top of head: 185
        #50% male distance from top of head to eyes: 112 (thus forehead is 56)
        #50% female distance from top spine joint to top of head: 173
        #50% female distance from top of head to eyes: 112 (thus forehead is 56)
        #average distance from spine to forehead: np.mean(185-56, 173-56) = 123 or 12.3 cm


        #print np.array(self.m.J_transformed[3, :]), np.array(self.m.J_transformed[6, :])
        #find closest point to x and y positions i.e. the first two columns. Find higher of the z position i.e. the 3rd.
        print "head: ", head_mid_joint
        dist_from_head = list(vertices - head_mid_joint)
        euclid_dist_from_head = np.square(dist_from_head)
        euclid_dist_from_head = list(np.sqrt(euclid_dist_from_head[:, 0] + euclid_dist_from_head[:, 1]))
        for i in range(10):
            print np.argmin(euclid_dist_from_head), dist_from_head[np.argmin(euclid_dist_from_head)] + head_mid_joint
            euclid_dist_from_head[np.argmin(euclid_dist_from_head)] += 100.
            #head is 336

        L_knee_joint = np.array(self.m.J_transformed[4, :])
        L_knee_joint[1] += 0.05
        print "L KNEE: ", L_knee_joint
        dist_from_L_knee = list(vertices - L_knee_joint)
        euclid_dist_from_L_knee = np.square(dist_from_L_knee)
        euclid_dist_from_L_knee = list(np.sqrt(euclid_dist_from_L_knee[:, 0] + euclid_dist_from_L_knee[:, 1]))
        for i in range(10):
            print np.argmin(euclid_dist_from_L_knee), dist_from_L_knee[np.argmin(euclid_dist_from_L_knee)] + L_knee_joint
            euclid_dist_from_L_knee[np.argmin(euclid_dist_from_L_knee)] += 100.
            #L knee is 1046
            #5cm up: 1032

        R_knee_joint = np.array(self.m.J_transformed[5, :])
        R_knee_joint[1] += 0.05
        print "R KNEE: ", R_knee_joint
        dist_from_R_knee = list(vertices - R_knee_joint)
        euclid_dist_from_R_knee = np.square(dist_from_R_knee)
        euclid_dist_from_R_knee = list(np.sqrt(euclid_dist_from_R_knee[:, 0] + euclid_dist_from_R_knee[:, 1]))
        for i in range(10):
            print np.argmin(euclid_dist_from_R_knee), dist_from_R_knee[np.argmin(euclid_dist_from_R_knee)] + R_knee_joint
            euclid_dist_from_R_knee[np.argmin(euclid_dist_from_R_knee)] += 100.
            #R knee is 4530
            #5cm up: 4515

        L_ankle_joint = np.array(self.m.J_transformed[7, :])
        L_ankle_joint[1] += 0.10
        print "L ankle: ", L_ankle_joint
        dist_from_L_ankle = list(vertices - L_ankle_joint)
        euclid_dist_from_L_ankle = np.square(dist_from_L_ankle)
        euclid_dist_from_L_ankle = list(np.sqrt(euclid_dist_from_L_ankle[:, 0] + euclid_dist_from_L_ankle[:, 1]))
        for i in range(10):
            print np.argmin(euclid_dist_from_L_ankle), dist_from_L_ankle[np.argmin(euclid_dist_from_L_ankle)] + L_ankle_joint
            euclid_dist_from_L_ankle[np.argmin(euclid_dist_from_L_ankle)] += 100.
            #L ankle is 3333
            #5cm up: 3319
            #10cm up: 1374

        R_ankle_joint = np.array(self.m.J_transformed[8, :])
        R_ankle_joint[1] += 0.10
        print "R ankle: ", R_ankle_joint
        dist_from_R_ankle = list(vertices - R_ankle_joint)
        euclid_dist_from_R_ankle = np.square(dist_from_R_ankle)
        euclid_dist_from_R_ankle = list(np.sqrt(euclid_dist_from_R_ankle[:, 0] + euclid_dist_from_R_ankle[:, 1]))
        for i in range(10):
            print np.argmin(euclid_dist_from_R_ankle), dist_from_R_ankle[np.argmin(euclid_dist_from_R_ankle)] + R_ankle_joint
            euclid_dist_from_R_ankle[np.argmin(euclid_dist_from_R_ankle)] += 100.
            #R knee is 6732
            #5cm up: 6720
            #10cm up: 4848


        L_elbow_joint = np.array(self.m.J_transformed[18, :])
        L_elbow_joint[0] -= 0.025
        print "L elbow: ", L_elbow_joint
        dist_from_L_elbow = list(vertices - L_elbow_joint)
        euclid_dist_from_L_elbow = np.square(dist_from_L_elbow)
        euclid_dist_from_L_elbow = list(np.sqrt(euclid_dist_from_L_elbow[:, 0] + euclid_dist_from_L_elbow[:, 2]))
        for i in range(10):
            print np.argmin(euclid_dist_from_L_elbow), dist_from_L_elbow[np.argmin(euclid_dist_from_L_elbow)] + L_elbow_joint
            euclid_dist_from_L_elbow[np.argmin(euclid_dist_from_L_elbow)] += 100.
            #L elbow is 1664
            #L elbow from side is 1620
            #L elbow from side and 2.5cm toward body is 1739
            #L elbow from side and 5cm toward body is 1681

        R_elbow_joint = np.array(self.m.J_transformed[19, :])
        R_elbow_joint[0] += 0.025
        print "R elbow: ", R_elbow_joint
        dist_from_R_elbow = list(vertices - R_elbow_joint)
        euclid_dist_from_R_elbow = np.square(dist_from_R_elbow)
        euclid_dist_from_R_elbow = list(np.sqrt(euclid_dist_from_R_elbow[:, 0] + euclid_dist_from_R_elbow[:, 2]))
        for i in range(10):
            print np.argmin(euclid_dist_from_R_elbow), dist_from_R_elbow[np.argmin(euclid_dist_from_R_elbow)] + R_elbow_joint
            euclid_dist_from_R_elbow[np.argmin(euclid_dist_from_R_elbow)] += 100.
            #R elbow is 5121
            #R elbow from side is 5091
            #R elbow from side and 2.5cm toward body is 5209
            #R elbow from side and 5cm toward body is 5150


        L_wrist_joint = np.array(self.m.J_transformed[20, :])
        L_wrist_joint[0] -= 0.05
        print "L wrist: ", L_wrist_joint
        dist_from_L_wrist = list(vertices - L_wrist_joint)
        euclid_dist_from_L_wrist = np.square(dist_from_L_wrist)
        euclid_dist_from_L_wrist = list(np.sqrt(euclid_dist_from_L_wrist[:, 0] + euclid_dist_from_L_wrist[:, 2]))
        for i in range(10):
            print np.argmin(euclid_dist_from_L_wrist), dist_from_L_wrist[np.argmin(euclid_dist_from_L_wrist)] + L_wrist_joint
            euclid_dist_from_L_wrist[np.argmin(euclid_dist_from_L_wrist)] += 100.
            # L wrist is 2208
            # L wrist 5cm toward body is 1960

        R_wrist_joint = np.array(self.m.J_transformed[21, :])
        R_wrist_joint[0] += 0.05
        print "R wrist: ", R_wrist_joint
        dist_from_R_wrist = list(vertices - R_wrist_joint)
        euclid_dist_from_R_wrist = np.square(dist_from_R_wrist)
        euclid_dist_from_R_wrist = list(np.sqrt(euclid_dist_from_R_wrist[:, 0] + euclid_dist_from_R_wrist[:, 2]))
        for i in range(10):
            print np.argmin(euclid_dist_from_R_wrist), dist_from_R_wrist[np.argmin(euclid_dist_from_R_wrist)] + R_wrist_joint
            euclid_dist_from_R_wrist[np.argmin(euclid_dist_from_R_wrist)] += 100.
            #R wrist is 5669
            #R wrist 5 cm toward body is 5423




if __name__ == "__main__":
    generator = GeneratePose(sampling = "UNIFORM", sigma = 0, one_side_range = 0)
    generator.get_volumetric_surface_points_closest()
