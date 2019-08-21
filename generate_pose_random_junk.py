import numpy as np
import random
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model as load_smpl_model

from keras.models import load_model as load_keras_model

#volumetric pose gen libraries
import lib_visualization as libVisualization
import lib_kinematics as libKinematics
#import lib_render as libRender
from process_yash_data import ProcessYashData
try:
    import dart_skel_sim
except:
    pass

#ROS
try:
    import rospy
    import tf
except:
    pass

#ROS libs
import rospkg
import roslib
import tf.transformations as tft
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import tensorflow as tensorflow
import cPickle as pickle


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL



class GeneratePose():
    def __init__(self, sampling = "NORMAL", sigma = 0, one_side_range = 0, gender="m"):
        ## Load SMPL model (here we load the female model)
        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
        self.m = load_smpl_model(model_path)

        self.filepath_prefix = '/home/henry'

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
        self.load_yifeng_data()



    def sample_body_shape(self, sampling, sigma, one_side_range):
        mu = 0
        for i in range(10):
            if sampling == "NORMAL":
                self.m.betas[i] = random.normalvariate(mu, sigma)
            elif sampling == "UNIFORM":
                self.m.betas[i]  = np.random.uniform(-one_side_range, one_side_range)



    def load_yifeng_data(self):

        arm_mat_contents = sio.loadmat('/home/henry/git/realistic_human_joint_limits/randomsin_arm_left_q_big.mat')
        for content in arm_mat_contents:
            print content

        arm_xyall = arm_mat_contents['qTrain_ba']
        print(arm_xyall.shape)

        np.random.shuffle(arm_xyall)

        self.yifeng_y_arm = arm_xyall[:, 4].reshape(-1, 1).astype(int)
        self.yifeng_X_arm = arm_xyall[:, :4]


        leg_mat_contents = sio.loadmat('/home/henry/git/realistic_human_joint_limits/randomsin_leg_left_q_big.mat')
        for content in leg_mat_contents:
            print content

        leg_xyall = leg_mat_contents['qTrain_ba']
        print(leg_xyall.shape)

        np.random.shuffle(leg_xyall)

        self.yifeng_y_leg = leg_xyall[:, 6].reshape(-1, 1).astype(int)
        self.yifeng_X_leg = leg_xyall[:, :4]

        self.curr_yifeng_arm_idx = 0
        self.curr_yifeng_leg_idx = 0



    def read_precomp_set(self):
        precomp_data = np.load('/home/henry/data/init_poses/all_yifeng_f_lay_100.npy', allow_pickle = True)
        precomp_data = list(precomp_data)
        print len(precomp_data)
        #print precomp_data[7]

        for i in range(35,36):

            pose_ind = precomp_data[i][1]
            pose_angs = precomp_data[i][2]
            validity = [1,0]#precomp_data[i][7]
            gender = 'f'
            posture = 'lay'

            print len(pose_ind), len(pose_angs)

            for idx in range(len(pose_ind)):
                m_idx = pose_ind[idx]
                ang = pose_angs[idx]
                if m_idx not in [3,4,5, 12]: continue
                else: self.m.pose[m_idx] = ang


            print self.m.pose

            joints = np.array(self.m.J_transformed)

            print joints
            print validity
            rospy.init_node("smpl_viz", anonymous=False)

            if int(validity[0]) == 1:
                r=1.0
                g=1.0
                b=0.0
                a = 0.3
            else:
                r=1.0
                g=0.0
                b=0.0
                a = 0.3

            time.sleep(1)


            libVisualization.rviz_publish_output(joints, r=r, g=g, b=b, a = a)
            libVisualization.rviz_publish_output_limbs_direct(joints, r=r, g=g, b=b, a = a)




    def generate_rand_dir_cos(self, gender, posture, num_data):

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

            generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)


            self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.

            #m, capsules, joint2name, rots0 = generator.map_nom_limited_random_selection_to_smpl_angles()
            m, capsules, joint2name, rots0, is_valid_pose = generator.map_yifeng_random_selection_to_smpl_angles(get_new=True)

            #print "GOT HERE"
            #time.sleep(2)

            shape_pose_vol[0] = np.asarray(m.betas).tolist()


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
        np.save(self.filepath_prefix+"/data/init_poses/all_yifeng_"+gender+"_"+posture+"_"+str(num_data)+".npy", np.array(shape_pose_vol_list))


    def map_random_cartesian_ik_to_smpl_angles(self, shift, get_new, alter_angles = True):
        joint_locs = np.array(self.m.J_transformed)
        print joint_locs
        #joint_locs = joint_locs - joint_locs[0, :]
        #print joint_locs
        #joint_locs += shift
        print joint_locs

        #let's do the left leg first.
        len_thigh = np.linalg.norm(joint_locs[4, :] - joint_locs[1, :])
        len_calf = np.linalg.norm(joint_locs[7, :] - joint_locs[4, :])
        len_leg = len_thigh+len_calf

        #pick anything in the rectangular prism box that is above the bed height or at least as high as the hip and lower than it on the bed
        has_found_valid_pose = False
        while has_found_valid_pose == False:
            x = np.random.uniform(joint_locs[1,0]-len_leg, joint_locs[1,0]+len_leg)
            y = np.random.uniform(joint_locs[1,1]-len_leg, joint_locs[1,1])
            z = np.random.uniform(joint_locs[1,2], joint_locs[1,2]+len_leg)

            x = joint_locs[1, 0] + len_leg - 0.01
            y = joint_locs[0, 1]# + len_leg# + 0.2
            z = joint_locs[0, 2]

            cart_len_from_hip = np.linalg.norm(np.array([x,y,z]) - joint_locs[1,:])
            print "got here", cart_len_from_hip, len_leg
            if cart_len_from_hip > len_leg: continue

            print "got here2", cart_len_from_hip, len_leg
            l_leg_pos_origins = np.array([np.array(self.m.J[0, :]), np.array(self.m.J[1, :]), np.array(self.m.J[4, :]),
                                          np.array(self.m.J[7, :])])


            l_leg_pos_current = np.array([x,y,z])
            libKinematics.ikpy_leg_rand_cart(l_leg_pos_origins, l_leg_pos_current)




        return self.m


    def map_yifeng_random_selection_to_smpl_angles(self, get_new, alter_angles=True):


        if alter_angles == True:
            print '\n'
            if get_new == True:
                self.arm_limb_angs = []
                for arm_side in ['left','right']:
                    is_within_human_manifold = False

                    while is_within_human_manifold == False:

                        i = self.curr_yifeng_arm_idx
                        zxy_angs = np.squeeze(self.yifeng_X_arm[i:i + 1, :])
                        is_valid_pose = np.squeeze(self.yifeng_y_arm[i:i + 1, :])

                        print np.array(zxy_angs), is_valid_pose, 'orig'

                        for i in range(3):
                            if zxy_angs[i] > np.pi:
                                zxy_angs[i] = zxy_angs[i] - 2 * np.pi

                        #print np.array(zxy_angs), prediction, 'orig'
                        R = libKinematics.ZXYeulerAnglesToRotationMatrix(zxy_angs)
                        #print R

                        eulers, eulers2 = libKinematics.rotationMatrixToZXYEulerAngles(R)  # use THESE rather than the originals

                        #print R - libKinematics.ZXYeulerAnglesToRotationMatrix(eulers)
                        #print R - libKinematics.ZXYeulerAnglesToRotationMatrix(eulers2)

                        #print eulers, 'recomp eul solution 1'
                        #print eulers2, 'recomp eul solution 2'

                        dircos = libKinematics.dir_cos_angles_from_matrix(R)
                        print dircos, 'recomp dircos 1 arm', arm_side, self.curr_yifeng_arm_idx
                        #print dircos2, 'recomp dircos 2'
                        self.curr_yifeng_arm_idx += 1
                        if int(is_valid_pose) == 1:
                            is_within_human_manifold = True

                    self.arm_limb_angs.append([dircos[0], dircos[1], dircos[2], zxy_angs[3]])



            self.m.pose[39] = self.arm_limb_angs[0][0]*1/3
            self.m.pose[40] = self.arm_limb_angs[0][1]*1/3
            self.m.pose[41] = self.arm_limb_angs[0][2]*1/3

            self.m.pose[48] = self.arm_limb_angs[0][0]*2/3
            self.m.pose[49] = self.arm_limb_angs[0][1]*2/3
            self.m.pose[50] = self.arm_limb_angs[0][2]*2/3

            self.m.pose[55] = -self.arm_limb_angs[0][3]


            self.m.pose[42] = self.arm_limb_angs[1][0]*1/3
            self.m.pose[43] = -self.arm_limb_angs[1][1]*1/3
            self.m.pose[44] = -self.arm_limb_angs[1][2]*1/3

            self.m.pose[51] = self.arm_limb_angs[1][0]*2/3
            self.m.pose[52] = -self.arm_limb_angs[1][1]*2/3
            self.m.pose[53] = -self.arm_limb_angs[1][2]*2/3

            self.m.pose[58] = self.arm_limb_angs[1][3]
            print '\n'



            if get_new == True:
                self.leg_limb_angs = []
                for leg_side in ['left','right']:
                    is_within_human_manifold = False

                    while is_within_human_manifold == False:

                        i = self.curr_yifeng_leg_idx
                        zxy_angs = np.squeeze(self.yifeng_X_leg[i:i + 1, :])
                        is_valid_pose = np.squeeze(self.yifeng_y_leg[i:i + 1, :])

                        print np.array(zxy_angs), is_valid_pose, 'orig'

                        for i in range(3):
                            if zxy_angs[i] > np.pi:
                                zxy_angs[i] = zxy_angs[i] - 2 * np.pi

                        #print np.array(zxy_angs), prediction, 'orig'
                        R = libKinematics.ZXYeulerAnglesToRotationMatrix(zxy_angs)
                        #print R

                        #eulers, eulers2 = libKinematics.rotationMatrixToZXYEulerAngles(R)  # use THESE rather than the originals

                        #print R - libKinematics.ZXYeulerAnglesToRotationMatrix(eulers)
                        #print R - libKinematics.ZXYeulerAnglesToRotationMatrix(eulers2)

                        #print eulers, 'recomp eul solution 1'
                        #print eulers2, 'recomp eul solution 2'

                        dircos = libKinematics.dir_cos_angles_from_matrix(R)
                        #print R-libKinematics.matrix_from_dir_cos_angles(dircos)
                        #print dircos2, 'recomp dircos 2'
                        print dircos, 'recomp dircos 1 leg', leg_side, self.curr_yifeng_leg_idx
                        #print dircos2, 'recomp dircos 2'
                        self.curr_yifeng_leg_idx += 1
                        if int(is_valid_pose) == 1:
                            is_within_human_manifold = True

                    self.leg_limb_angs.append([dircos[0], dircos[1], dircos[2], zxy_angs[3]])

            self.m.pose[3] = self.leg_limb_angs[0][0]
            self.m.pose[4] = self.leg_limb_angs[0][1]
            self.m.pose[5] = self.leg_limb_angs[0][2]

            self.m.pose[12] = self.leg_limb_angs[0][3]

            self.m.pose[6] = self.leg_limb_angs[1][0]
            self.m.pose[7] = -self.leg_limb_angs[1][1]
            self.m.pose[8] = -self.leg_limb_angs[1][2]

            self.m.pose[15] = self.leg_limb_angs[1][3]


            print '\n'







        #self.m.pose[51] = selection_r
        from capsule_body import get_capsules, joint2name, rots0
        capsules = get_capsules(self.m)
        joint2name = joint2name
        rots0 = rots0

        return self.m, capsules, joint2name, rots0, is_valid_pose



    def map_nom_limited_random_selection_to_smpl_angles(self, alter_angles=True):
        if alter_angles == True:

            #self.m.pose[3] = np.random.uniform(np.deg2rad(-132.1), np.deg2rad(17.8))
            self.m.pose[3] = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(17.8))
            self.m.pose[4] = np.random.uniform(np.deg2rad(-33.7), np.deg2rad(32.6))
            self.m.pose[5] = np.random.uniform(np.deg2rad(-30.5), np.deg2rad(38.6))

            self.m.pose[12] = np.random.uniform(np.deg2rad(-1.3), np.deg2rad(139.9))



            self.m.pose[6] = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(17.8))
            self.m.pose[7] = np.random.uniform(np.deg2rad(-32.6), np.deg2rad(33.7))
            self.m.pose[8] = np.random.uniform(np.deg2rad(-38.6), np.deg2rad(30.5))

            self.m.pose[15] = np.random.uniform(np.deg2rad(-1.3), np.deg2rad(139.9))

            ls_roll = np.random.uniform(np.deg2rad(-88.9), np.deg2rad(81.4))
            ls_yaw = np.random.uniform(np.deg2rad(-140.7), np.deg2rad(43.7))
            ls_pitch = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(80.4))

            self.m.pose[39] = ls_roll*1/3
            self.m.pose[40] = ls_yaw*1/3
            self.m.pose[41] = ls_pitch*1/3
            self.m.pose[48] = ls_roll*2/3
            self.m.pose[49] = ls_yaw*2/3
            self.m.pose[50] = ls_pitch*2/3

            self.m.pose[55] = np.random.uniform(np.deg2rad(-147.3), np.deg2rad(2.8))

            rs_roll = np.random.uniform(np.deg2rad(-88.9), np.deg2rad(81.4))
            rs_yaw = np.random.uniform(np.deg2rad(-43.7), np.deg2rad(140.7))
            rs_pitch = np.random.uniform(np.deg2rad(-80.4), np.deg2rad(90.0))

            self.m.pose[42] = rs_roll*1/3
            self.m.pose[43] = rs_yaw*1/3
            self.m.pose[44] = rs_pitch*1/3
            self.m.pose[51] = rs_roll*2/3
            self.m.pose[52] = rs_yaw*2/3
            self.m.pose[53] = rs_pitch*2/3

            self.m.pose[58] = np.random.uniform(np.deg2rad(-2.8), np.deg2rad(147.3))


        #self.m.pose[51] = selection_r
        from capsule_body import get_capsules, joint2name, rots0
        capsules = get_capsules(self.m)
        joint2name = joint2name
        rots0 = rots0
        print len(capsules)

        #put these capsules into dart based on these angles. Make Dart joints only as necessary.
        #Use the positions found in dart to update positions in FleX. Do not use angles in Flex
        #repeat: do not need a forward kinematics model in FleX! Just need the capsule positions and radii. Can potentially get rotation from the Capsule end positions.
        #Find IK solution at the very end.

        return self.m, capsules, joint2name, rots0







if __name__ == "__main__":
    generator = GeneratePose(sampling = "UNIFORM", sigma = 0, one_side_range = 0)
    #libRender.standard_render(generator.m)
    #generator.ax = plt.figure().add_subplot(111, projection='3d')
    #generator.solve_ik_tree_smpl()

    posture = "lay"
    #generator.read_precomp_set()
    generator.generate_rand_dir_cos(gender='f', posture='lay', num_data=100)

    #generator.save_yash_data_with_angles(posture)
    #generator.map_euler_angles_to_axis_angle()
    #generator.check_found_limits()

    #processYashData = ProcessYashData()
    #processYashData.map_yash_to_axis_angle(verbose=False)
    #processYashData.check_found_limits()

    #processYashData.get_r_leg_angles()



    #m, capsules, joint2name, rots0 = generator.map_random_selection_to_smpl_angles(alter_angles = True)

    #dss = dart_skel_sim.DartSkelSim(render=True, m=m, capsules=capsules, joint_names=joint2name, initial_rots=rots0)

    #generator.standard_render()
    #dss.run_simulation()