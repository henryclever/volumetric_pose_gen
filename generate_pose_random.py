import numpy as np
import random

from keras.models import load_model as load_keras_model

from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints


from smpl.smpl_webuser.serialization import load_model as load_smpl_model



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
        print "changed body to: ",gender

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
        print self.m.J_transformed, 'here'

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
        self.yifeng_X_leg = leg_xyall[:, :6]

        self.curr_yifeng_arm_idx = 0
        self.curr_yifeng_leg_idx = 0

        self.arm_model = load_keras_model('/home/henry/git/realistic_human_joint_limits/arm_left_limits.h5')
        self.leg_model = load_keras_model('/home/henry/git/realistic_human_joint_limits/leg_left_limits_big.h5')


    def read_precomp_set(self, gender):
        precomp_data = np.load('/home/henry/data/init_poses/random/all_rand_nom_endhtbicheck_rollpi_'+gender+'_lay_1500_set1.npy', allow_pickle = True)
        precomp_data = list(precomp_data)
        print len(precomp_data)
        #print precomp_data[7]

        shape_pose_17joints_list = []

        num_samples = 50000

        for i in range(num_samples):
            print i
            #if i == num_samples/2: GeneratePose(sampling="UNIFORM", sigma=0, one_side_range=0, gender='m')
            shape_pose_17joints = []
            #betas = precomp_data[i][0]
            #pose_ind = precomp_data[i][1]
            #pose_angs = precomp_data[i][2]
            #validity = [1,0]#precomp_data[i][7]
            #posture = 'lay'

            #print len(pose_ind), len(pose_angs), i

            #for idx in range(len(betas)):
                #self.m.betas[idx] = betas[idx]


            #for idx in range(len(pose_ind)):
            #    m_idx = pose_ind[idx]
            #    ang = pose_angs[idx]
            #    self.m.pose[m_idx] = ang
            #print self.m.pose



            generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)

            #self.m.pose[3] = np.random.uniform(np.deg2rad(-180.), np.deg2rad(90.))
            self.m.pose[3] = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(17.8))
            #self.m.pose[4] = np.random.uniform(np.deg2rad(-90.), np.deg2rad(90.))
            self.m.pose[4] = np.random.uniform(np.deg2rad(-33.7), np.deg2rad(32.6))
            #self.m.pose[5] = np.random.uniform(np.deg2rad(-90.), np.deg2rad(90.))
            self.m.pose[5] = np.random.uniform(np.deg2rad(-30.5), np.deg2rad(38.6))
            #self.m.pose[12] = np.random.uniform(np.deg2rad(-1.0), np.deg2rad(180.))
            self.m.pose[12] = np.random.uniform(np.deg2rad(-1.3), np.deg2rad(139.9))

            #self.m.pose[6] = np.random.uniform(np.deg2rad(-180.), np.deg2rad(90.))
            self.m.pose[6] = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(17.8))
            #self.m.pose[7] = np.random.uniform(np.deg2rad(-90.), np.deg2rad(90.))
            self.m.pose[7] = np.random.uniform(np.deg2rad(-32.6), np.deg2rad(33.7))
            #self.m.pose[8] = np.random.uniform(np.deg2rad(-90.), np.deg2rad(90.))
            self.m.pose[8] = np.random.uniform(np.deg2rad(-38.6), np.deg2rad(30.5))

            #self.m.pose[15] = np.random.uniform(np.deg2rad(-0.0), np.deg2rad(180.))
            self.m.pose[15] = np.random.uniform(np.deg2rad(-1.3), np.deg2rad(139.9))

            #ls_roll = np.random.uniform(np.deg2rad(-90.), np.deg2rad(90.))
            ls_roll = np.random.uniform(np.deg2rad(-(88.9+45.)), np.deg2rad(81.4+45.))
            #ls_yaw = np.random.uniform(np.deg2rad(-180.), np.deg2rad(90.))
            ls_yaw = np.random.uniform(np.deg2rad(-140.7), np.deg2rad(43.7))
            #ls_pitch = np.random.uniform(np.deg2rad(-180.), np.deg2rad(90.))
            ls_pitch = np.random.uniform(np.deg2rad(-135.0), np.deg2rad(80.4)) #first should be -90

            self.m.pose[39] = ls_roll * 1 / 3
            self.m.pose[40] = ls_yaw * 1 / 3
            self.m.pose[41] = ls_pitch * 1 / 3
            self.m.pose[48] = ls_roll * 2 / 3
            self.m.pose[49] = ls_yaw * 2 / 3
            self.m.pose[50] = ls_pitch * 2 / 3

            #self.m.pose[55] = np.random.uniform(np.deg2rad(-180.), np.deg2rad(0.0))
            self.m.pose[55] = np.random.uniform(np.deg2rad(-147.3), np.deg2rad(2.8))

            #rs_roll = np.random.uniform(np.deg2rad(-90.), np.deg2rad(90.))
            rs_roll = np.random.uniform(np.deg2rad(-(88.9+45.)), np.deg2rad(81.4+45.))
            #rs_yaw = np.random.uniform(np.deg2rad(-90.), np.deg2rad(180.))
            rs_yaw = np.random.uniform(np.deg2rad(-43.7), np.deg2rad(140.7))
            #rs_pitch = np.random.uniform(np.deg2rad(-90.), np.deg2rad(180.))
            rs_pitch = np.random.uniform(np.deg2rad(-80.4), np.deg2rad(135.0)) #2nd should be 90

            self.m.pose[42] = rs_roll * 1 / 3
            self.m.pose[43] = rs_yaw * 1 / 3
            self.m.pose[44] = rs_pitch * 1 / 3
            self.m.pose[51] = rs_roll * 2 / 3
            self.m.pose[52] = rs_yaw * 2 / 3
            self.m.pose[53] = rs_pitch * 2 / 3

            #self.m.pose[58] = np.random.uniform(np.deg2rad(0.0), np.deg2rad(180.))
            self.m.pose[58] = np.random.uniform(np.deg2rad(-2.8), np.deg2rad(147.3))

            joints = np.array(self.m.J_transformed)
            joints = joints - joints[0, :] + np.array([0.0, -0.4, 0.0])

            joints[:, 2] *= -1

            joints_pose_cond_check = np.stack((joints[0, :],
                                               joints[12, :],
                                               joints[17, :],
                                               joints[19, :],
                                               joints[21, :],
                                               joints[16, :],
                                               joints[18, :],
                                               joints[20, :],
                                               joints[15, :],
                                               joints[2, :],
                                               joints[5, :],
                                               joints[8, :],
                                               joints[11, :],
                                               joints[1, :],
                                               joints[4, :],
                                               joints[7, :],
                                               joints[10, :],))

            #print ((joints_pose_cond_check*100).astype(int)).astype(float)/100


            shape_pose_17joints.append(np.array(self.m.betas))
            shape_pose_17joints.append(0)
            shape_pose_17joints.append(np.array(self.m.pose))
            shape_pose_17joints.append(joints_pose_cond_check)

            shape_pose_17joints_list.append(shape_pose_17joints)


            #print validity
            #rospy.init_node("smpl_viz", anonymous=False)

            #if int(validity[0]) == 1:
            #    r=1.0
            #    g=1.0
            #    b=0.0
            #    a = 0.3
            #else:
            #    r=1.0
            #    g=0.0
            #    b=0.0
            #    a = 0.3


            #libVisualization.rviz_publish_output(joints, r=r, g=g, b=b, a = a)
            #libVisualization.rviz_publish_output_limbs_direct(joints, r=r, g=g, b=b, a = a)


            #dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture=posture, stiffness=None,
            #                                check_only_distal=True, filepath_prefix=self.filepath_prefix, add_floor=False)




            # run a step to check for collisions
            #dss.run_sim_step()

            #dss.world.check_collision()
            #print "checked collisions"
            # print dss.world.CollisionResult()
            #print "is valid pose?", is_valid_pose
            #print dss.world.collision_result.contacted_bodies

            #dss.run_simulation(1)

            #for i in range(100):
            #    print precomp_data[i][7]


        #only_17_joints = np.array(shape_pose_17joints_list[:][3])

        shape_pose_17joints_list = np.array(shape_pose_17joints_list)

        print np.shape(shape_pose_17joints_list)

        only_17_joints = np.array(list(shape_pose_17joints_list[:, 3]))
        print only_17_joints.shape
        #print only_17_joints

        import numpy, scipy.io

        #arr = numpy.arange(10)
        #arr = arr.reshape((3, 3))  # 2d array of 3x3

        #scipy.io.savemat('/home/henry/data/init_poses/random/generous_limit_'+gender+'_'+str(num_samples)+'_samp_lay.mat', mdict={'arr':only_17_joints})
        #np.save('/home/henry/data/init_poses/random/generous_limit_'+gender+'_'+str(num_samples)+'_samp_lay.npy', np.array(shape_pose_17joints_list))
        scipy.io.savemat('/home/henry/data/init_poses/random/nom_limit_'+gender+'_'+str(num_samples)+'_samp_lay.mat', mdict={'arr':only_17_joints})
        np.save('/home/henry/data/init_poses/random/nom_limit_'+gender+'_'+str(num_samples)+'_samp_lay.npy', np.array(shape_pose_17joints_list))

    def generate_rand_dir_cos(self, gender, posture, num_data, roll_person, set, prevent_limb_overhang, hands_behind_head):

        #NEED FOR DATASET: pose Nx72, shape Nx10
        shape_pose_vol_list = []
        #contact_check_bns = [1, 2, 4, 5, 7, 8, 14, 15, 16, 17, 18, 19]
        if hands_behind_head == False:
            contact_check_bns = [4, 5, 7, 8, 16, 17, 18, 19]
        else:
            contact_check_bns = [4, 5, 7, 8, 16, 17]

        contact_exceptions = [[9, 14],[9, 15]]





        for i in range(num_data):
            shape_pose_vol = [[],[],[],[],[],[],[],[]]

            #root_rot = np.random.uniform(-np.pi / 16, np.pi / 16)

            generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)
            in_collision = True

            #m, capsules, joint2name, rots0 = generator.map_nom_limited_random_selection_to_smpl_angles()
            #m, capsules, joint2name, rots0, is_valid_pose = generator.map_yifeng_random_selection_to_smpl_angles(get_new = True)
            #m = generator.map_random_cartesian_ik_to_smpl_angles([shift_side, shift_ud, 0.0], get_new = True)

            self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.

            dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, gender=gender, posture=posture, stiffness=None, check_only_distal = True, filepath_prefix=self.filepath_prefix, add_floor = False)

            #print "dataset create type", DATASET_CREATE_TYPE
            #print self.m.pose
            volumes = dss.getCapsuleVolumes(mm_resolution = 2.)

            #libRender.standard_render(self.m)
            #print volumes
            shape_pose_vol[6] = volumes
            dss.world.reset()
            dss.world.destroy()

            if hands_behind_head == True:
                arm_choices = 4
            else:
                arm_choices = 8

            self.left_arm_block = int(np.random.randint(arm_choices))
            self.right_arm_block = int(np.random.randint(arm_choices))
            self.left_leg_block = int(np.random.randint(4))
            self.right_leg_block = int(np.random.randint(4))
            self.num_collisions = 0

            while in_collision == True:

                if self.num_collisions > 20: self.num_collisions = 0
                if self.num_collisions == 0:
                    generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)
                    shift_side = np.random.uniform(-0.2, 0.2)  # in meters
                    shift_ud = np.random.uniform(-0.2, 0.2)  # in meters
                    #shape_pose_vol[3] = root_rot
                    shape_pose_vol[4] = shift_side
                    shape_pose_vol[5] = shift_ud

                    self.m.pose[0:3] = np.random.rand(3) * 0.
                    if roll_person == True:
                        self.m.pose[1] = np.random.uniform(-np.pi, np.pi)
                    self.m.pose[2] = np.random.uniform(-np.pi/6, np.pi/6)


                #m, capsules, joint2name, rots0 = generator.map_random_selection_to_smpl_angles(alter_angles = True)

                print "GOT HERE", len(shape_pose_vol_list)
                #time.sleep(2)

                self.m.pose[3:] = np.random.rand(self.m.pose.size - 3) * 0.

                m, capsules, joint2name, rots0 = generator.map_nom_limited_random_selection_to_smpl_angles(alter_angles= True, roll_person = roll_person,
                                                                                                           shift=np.array([shift_side, shift_ud, 0.0]),
                                                                                                           prevent_limb_overhang = prevent_limb_overhang,
                                                                                                           hands_behind_head = hands_behind_head)
                #m, capsules, joint2name, rots0 = generator.map_yifeng_random_selection_to_smpl_angles(get_new=True, alter_angles= True, roll_person = roll_person)

                shape_pose_vol[3] = np.copy(m.pose[2])
                shape_pose_vol[7] = np.copy(m.pose[1])


                print "GOT HERE2, num collision tries: ", self.num_collisions
                #time.sleep(2)

                shape_pose_vol[0] = np.asarray(m.betas).tolist()

                #print "stepping", m.pose
                dss = dart_skel_sim.DartSkelSim(render=True, m=m, gender=gender, posture = posture, stiffness=None, check_only_distal = True, filepath_prefix=self.filepath_prefix, add_floor = False)

                #print "stepping", m.pose
                invalid_pose = False
                #run a step to check for collisions
                dss.run_sim_step()

                dss.world.check_collision()
                print "checked collisions"
                #print dss.world.CollisionResult()
                #print "is valid pose?", is_valid_pose
                print dss.world.collision_result.contacted_bodies

                #dss.run_simulation(1)

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

                        #libRender.standard_render(self.m)
                        in_collision = False
                else: # no contacts anywhere.

                    print "resampling shape and pose, no collision."
                    in_collision = False
                    #libRender.standard_render(self.m)



                #shape_pose_vol[7] = [is_valid_pose, in_collision]
                #in_collision = False

                #dss.world.skeletons[0].remove_all_collision_pairs()

                #libRender.standard_render(self.m)
                dss.world.reset()
                dss.world.destroy()

                self.num_collisions += 1

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

        if prevent_limb_overhang == True:
            plo = "_plo"
        else:
            plo = ""

        if roll_person == True:
            rp = "pi"
        else:
            rp = "0"

        if hands_behind_head == True:
            hbh = "_hbh"
        else:
            hbh = ""

        np.save(self.filepath_prefix+"/data/init_poses/all_rand_nom_endhtbicheck_roll"+rp+plo+hbh+"_"+gender+"_"+posture+"_set"+str(set)+"_"+str(num_data)+".npy", np.array(shape_pose_vol_list))



    def check_height(self, root_z_loc, distal_z_loc, hands_behind_head = False):
        if hands_behind_head == False:
            if distal_z_loc < (root_z_loc - 0.2) or distal_z_loc > (root_z_loc + 0.2):
                is_ht_ok = False
            else:
                is_ht_ok = True
        elif hands_behind_head == True: #make sure the hands are actually behind the head!
            if distal_z_loc < (root_z_loc - 0.4) or distal_z_loc > (root_z_loc + 0.0):
                is_ht_ok = False
            else:
                is_ht_ok = True
        return is_ht_ok

    def check_leg_block(self, leg_block, hip_y_loc, ankle_y_loc, leg_length):
        if leg_block == 0:
            if ankle_y_loc > hip_y_loc - leg_length * 1 / 4: is_leg_chosen = True
            else: is_leg_chosen = False
        elif leg_block == 1:
            if ankle_y_loc <= hip_y_loc - leg_length * 1 / 4 and ankle_y_loc > hip_y_loc - leg_length * 1 / 2: is_leg_chosen = True
            else: is_leg_chosen = False
        elif leg_block == 2:
            if ankle_y_loc <= hip_y_loc - leg_length * 1 / 2 and ankle_y_loc > hip_y_loc - leg_length * 3 / 4: is_leg_chosen = True
            else: is_leg_chosen = False
        elif leg_block == 3:
            if ankle_y_loc <= hip_y_loc - leg_length * 3 / 4: is_leg_chosen = True
            else: is_leg_chosen = False
        else:
            is_leg_chosen = False
        return is_leg_chosen

    def check_arm_block(self, arm_block, should_x_loc, wrist_x_loc, should_y_loc, wrist_y_loc, arm_length, side, hands_behind_head):
        if arm_block == 0:
            if wrist_y_loc > should_y_loc + arm_length*3/4: is_arm_chosen = True
            else: is_arm_chosen = False
        elif arm_block == 1:
            if wrist_y_loc <= should_y_loc + arm_length*3/4 and wrist_y_loc > should_y_loc + arm_length*1/2: is_arm_chosen = True
            else: is_arm_chosen = False
        elif arm_block == 2:
            if wrist_y_loc <= should_y_loc + arm_length*1/2 and wrist_y_loc > should_y_loc + arm_length*1/4: is_arm_chosen = True
            else: is_arm_chosen = False
        elif arm_block == 3:
            if wrist_y_loc <= should_y_loc + arm_length*1/4 and wrist_y_loc > should_y_loc: is_arm_chosen = True
            else: is_arm_chosen = False
        elif arm_block == 4:
            if wrist_y_loc <= should_y_loc and wrist_y_loc > should_y_loc - arm_length*1/4: is_arm_chosen = True
            else: is_arm_chosen = False
        elif arm_block == 5:
            if wrist_y_loc <= should_y_loc - arm_length*1/4 and wrist_y_loc > should_y_loc - arm_length*1/2: is_arm_chosen = True
            else: is_arm_chosen = False
        elif arm_block == 6:
            if wrist_y_loc <= should_y_loc - arm_length*1/2 and wrist_y_loc > should_y_loc - arm_length*3/4: is_arm_chosen = True
            else: is_arm_chosen = False
        elif arm_block == 7:
            if wrist_y_loc <= should_y_loc - arm_length*3/4: is_arm_chosen = True
            else: is_arm_chosen = False
        else:
            is_arm_chosen = False

        if hands_behind_head == True:
            if side == 'left':
                if wrist_x_loc < should_x_loc: pass
                else: is_arm_chosen = False
            if side == 'right':
                if wrist_x_loc > should_x_loc: pass
                else: is_arm_chosen = False


        return is_arm_chosen


    def check_limb_overhang(self, mJ, mJtransformed_red, shift, limb_tag, hands_behind_head = False):
        #print mJtransformed_red
        #print mJtransformed_red - mJ[0, :]
        #print mJtransformed_red - mJ[0, :] + np.array(shift)
        #print (mJtransformed_red - mJ[0, :] + np.array(shift))*2.58872
        global_joint_pos = (mJtransformed_red - mJ[0, :] + np.array(shift))*2.58872 + np.array([1.185, 2.55, 0.0])
        #print global_joint_pos, 'final mesh tree'

        if hands_behind_head == True:
            arm_choices = 4
        else:
            arm_choices = 8

        #print np.min(global_joint_pos[:, 0]), np.max(global_joint_pos[:, 0])
        #print np.min(global_joint_pos[:, 1]), np.max(global_joint_pos[:, 1])
        if np.min(global_joint_pos[:, 0]) > 0.0 and np.max(global_joint_pos[:, 0]) < 1.185*2 and np.min(global_joint_pos[:, 1]) > 0.0 and np.max(global_joint_pos[:, 1]) < 2.55*2:
            is_limb_chosen = True
        else:
            is_limb_chosen = False
            #we also have to see if the block itself is valid: i.e. pick a new block if the arm is up too high or the leg is down too low.
            if limb_tag == 'left_leg':
                if np.min(global_joint_pos[:, 1]) <= 0.0 or self.try_idx > 20:
                    print 'old left leg block: ', self.left_leg_block,
                    self.left_leg_block = int(np.random.randint(4))
                    print '   new block: ', self.left_leg_block
                    self.try_idx = 0
            elif limb_tag == 'right_leg':
                if np.min(global_joint_pos[:, 1]) <= 0.0 or self.try_idx > 20:
                    print 'old right leg block: ', self.right_leg_block,
                    self.right_leg_block = int(np.random.randint(4))
                    print '   new block: ', self.right_leg_block
                    self.try_idx = 0
            elif limb_tag == 'left_arm':
                if np.max(global_joint_pos[:, 1]) >= 2.55*2 or self.try_idx > 20:
                    print 'old left arm block: ', self.left_arm_block,
                    self.left_arm_block = int(np.random.randint(arm_choices))
                    print '   new block: ', self.left_arm_block
                    self.try_idx = 0
            elif limb_tag == 'right_arm':
                if np.max(global_joint_pos[:, 1]) >= 2.55*2 or self.try_idx > 20:
                    print 'old right arm block: ', self.right_arm_block,
                    self.right_arm_block = int(np.random.randint(arm_choices))
                    print '   new block: ', self.right_arm_block
                    self.try_idx = 0



        print is_limb_chosen, self.try_idx, self.m.betas[0], self.num_collisions
        self.try_idx += 1
        return is_limb_chosen


    def map_nom_limited_random_selection_to_smpl_angles(self, alter_angles, roll_person, shift, prevent_limb_overhang, hands_behind_head):
        if alter_angles == True:
            #print self.m.r
            #print mJtransformed, 'MJTRANS'
            #print self.m.pose[2], 'yaw of person in space'


            mJtransformed = np.array(self.m.J_transformed)

            leg_length = np.linalg.norm(np.array(mJtransformed[7, :])-np.array(mJtransformed[4, :]))+np.linalg.norm(np.array(mJtransformed[4, :])-np.array(mJtransformed[1, :]))

            print 'picking left leg...', self.left_leg_block
            left_leg_chosen = False
            self.try_idx = 0
            while left_leg_chosen == False:
                # self.m.pose[3] = np.random.uniform(np.deg2rad(-132.1), np.deg2rad(17.8))
                self.m.pose[3] = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(17.8))
                self.m.pose[4] = np.random.uniform(np.deg2rad(-33.7), np.deg2rad(32.6))
                self.m.pose[5] = np.random.uniform(np.deg2rad(-30.5), np.deg2rad(38.6))
                self.m.pose[12] = np.random.uniform(np.deg2rad(-1.3), np.deg2rad(139.9))

                mJtransformed = np.array(self.m.J_transformed)
                mJ = np.array(self.m.J)

                #check if the ankle falls in the correct block to ensure more even distribution across the space
                left_leg_chosen = self.check_height(root_z_loc=mJtransformed[1, 2], distal_z_loc=mJtransformed[7, 2])

                if left_leg_chosen == True:
                    #if prevent_limb_overhang == False:
                    left_leg_chosen = self.check_leg_block(leg_block=self.left_leg_block,
                                                           hip_y_loc=mJtransformed[1, 1],
                                                           ankle_y_loc=mJtransformed[7, 1],
                                                           leg_length=leg_length)

                    if left_leg_chosen == True and prevent_limb_overhang == True: #only check off the edges if we know the block is OK
                        mJtransformed_red = np.stack((mJtransformed[4, :], mJtransformed[7, :], mJtransformed[10, :]))
                        left_leg_chosen = self.check_limb_overhang(mJ=mJ, mJtransformed_red=mJtransformed_red, shift=shift, limb_tag='left_leg', hands_behind_head = hands_behind_head)


            print mJtransformed[7, 2] - mJtransformed[1, 2],'   picking right leg...', self.right_leg_block
            right_leg_chosen = False
            self.try_idx = 0
            while right_leg_chosen == False:
                self.m.pose[6] = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(17.8))
                self.m.pose[7] = np.random.uniform(np.deg2rad(-32.6), np.deg2rad(33.7))
                self.m.pose[8] = np.random.uniform(np.deg2rad(-38.6), np.deg2rad(30.5))

                self.m.pose[15] = np.random.uniform(np.deg2rad(-1.3), np.deg2rad(139.9))

                mJtransformed = np.array(self.m.J_transformed)
                mJ = np.array(self.m.J)


                right_leg_chosen = self.check_height(root_z_loc=mJtransformed[2, 2], distal_z_loc=mJtransformed[8, 2])

                if right_leg_chosen == True:
                    #if prevent_limb_overhang == False:
                    right_leg_chosen = self.check_leg_block(leg_block=self.right_leg_block,
                                                                hip_y_loc=mJtransformed[2, 1],
                                                                ankle_y_loc=mJtransformed[8, 1],
                                                                leg_length=leg_length)

                    if right_leg_chosen == True and prevent_limb_overhang == True: #only check off the edges if we know the block is OK
                        mJtransformed_red = np.stack((mJtransformed[5, :], mJtransformed[8, :], mJtransformed[11, :]))
                        right_leg_chosen = self.check_limb_overhang(mJ=mJ, mJtransformed_red=mJtransformed_red, shift=shift, limb_tag='right_leg')


            arm_length = np.linalg.norm(np.array(mJtransformed[20, :])-np.array(mJtransformed[18, :]))+np.linalg.norm(np.array(mJtransformed[18, :])-np.array(mJtransformed[16, :]))
            print mJtransformed[8, 2] - mJtransformed[2, 2],'   picking left arm...', self.left_arm_block
            left_arm_chosen = False
            self.try_idx = 0
            while left_arm_chosen == False:
                if hands_behind_head == True:
                    ls_roll = np.random.uniform(np.deg2rad(-88.9), np.deg2rad(81.4+45.))
                    ls_pitch = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(80.4+45.))
                else:
                    ls_roll = np.random.uniform(np.deg2rad(-88.9), np.deg2rad(81.4))
                    ls_pitch = np.random.uniform(np.deg2rad(-90.0), np.deg2rad(80.4))
                ls_yaw = np.random.uniform(np.deg2rad(-140.7), np.deg2rad(43.7))

                self.m.pose[39] = ls_roll*1/3
                self.m.pose[40] = ls_yaw*1/3
                self.m.pose[41] = ls_pitch*1/3
                self.m.pose[48] = ls_roll*2/3
                self.m.pose[49] = ls_yaw*2/3
                self.m.pose[50] = ls_pitch*2/3

                self.m.pose[55] = np.random.uniform(np.deg2rad(-147.3), np.deg2rad(2.8))


                mJtransformed = np.array(self.m.J_transformed)
                mJ = np.array(self.m.J)

                left_arm_chosen = self.check_height(root_z_loc=mJtransformed[16, 2], distal_z_loc=mJtransformed[20, 2], hands_behind_head = hands_behind_head)

                if left_arm_chosen == True:
                    #if prevent_limb_overhang == False:
                    left_arm_chosen = self.check_arm_block(arm_block=self.left_arm_block,
                                                           should_x_loc=mJtransformed[16, 0],
                                                           wrist_x_loc=mJtransformed[20, 0],
                                                           should_y_loc=mJtransformed[16, 1],
                                                           wrist_y_loc=mJtransformed[20, 1],
                                                           arm_length=arm_length,
                                                           side = 'left',
                                                           hands_behind_head=hands_behind_head)

                    if left_arm_chosen == True and prevent_limb_overhang == True: #only check off the edges if we know the block is OK
                        mJtransformed_red = np.stack((mJtransformed[18, :], mJtransformed[20, :], mJtransformed[22, :]))
                        left_arm_chosen = self.check_limb_overhang(mJ=mJ, mJtransformed_red=mJtransformed_red, shift=shift, limb_tag='left_arm')


            print mJtransformed[20, 2] - mJtransformed[16, 2],'   picking right arm...', self.right_arm_block
            right_arm_chosen = False
            self.try_idx = 0
            while right_arm_chosen == False:
                if hands_behind_head == True:
                    rs_roll = np.random.uniform(np.deg2rad(-88.9), np.deg2rad(81.4+45.))
                    rs_pitch = np.random.uniform(np.deg2rad(-(80.4+45.)), np.deg2rad(90.0))
                else:
                    rs_roll = np.random.uniform(np.deg2rad(-88.9), np.deg2rad(81.4))
                    rs_pitch = np.random.uniform(np.deg2rad(-80.4), np.deg2rad(90.0))
                rs_yaw = np.random.uniform(np.deg2rad(-43.7), np.deg2rad(140.7))

                self.m.pose[42] = rs_roll*1/3
                self.m.pose[43] = rs_yaw*1/3
                self.m.pose[44] = rs_pitch*1/3
                self.m.pose[51] = rs_roll*2/3
                self.m.pose[52] = rs_yaw*2/3
                self.m.pose[53] = rs_pitch*2/3

                self.m.pose[58] = np.random.uniform(np.deg2rad(-2.8), np.deg2rad(147.3))

                mJtransformed = np.array(self.m.J_transformed)
                mJ = np.array(self.m.J)

                right_arm_chosen = self.check_height(root_z_loc=mJtransformed[17, 2], distal_z_loc=mJtransformed[21, 2], hands_behind_head = hands_behind_head)

                if right_arm_chosen == True:
                    #if prevent_limb_overhang == False:
                    right_arm_chosen = self.check_arm_block(arm_block=self.right_arm_block,
                                                           should_x_loc=mJtransformed[17, 0],
                                                           wrist_x_loc=mJtransformed[21, 0],
                                                           should_y_loc=mJtransformed[17, 1],
                                                           wrist_y_loc=mJtransformed[21, 1],
                                                           arm_length=arm_length,
                                                           side = 'right',
                                                           hands_behind_head=hands_behind_head)

                    if right_arm_chosen == True and prevent_limb_overhang == True: #only check off the edges if we know the block is OK
                        mJtransformed_red = np.stack((mJtransformed[19, :], mJtransformed[21, :], mJtransformed[23, :]))
                        right_arm_chosen = self.check_limb_overhang(mJ=mJ, mJtransformed_red=mJtransformed_red, shift=shift, limb_tag='right_arm')


        #self.m.pose[51] = selection_r
        from capsule_body import get_capsules, joint2name, rots0
        capsules = get_capsules(self.m)
        joint2name = joint2name
        rots0 = rots0
        #print len(capsules)

        #put these capsules into dart based on these angles. Make Dart joints only as necessary.
        #Use the positions found in dart to update positions in FleX. Do not use angles in Flex
        #repeat: do not need a forward kinematics model in FleX! Just need the capsule positions and radii. Can potentially get rotation from the Capsule end positions.
        #Find IK solution at the very end.

        return self.m, capsules, joint2name, rots0







if __name__ == "__main__":
    gender = 'm'


    generator = GeneratePose(sampling = "UNIFORM", sigma = 0, one_side_range = 0, gender=gender)
    #libRender.standard_render(generator.m)
    #generator.ax = plt.figure().add_subplot(111, projection='3d')
    #generator.solve_ik_tree_smpl()

    #generator.read_precomp_set(gender=gender)
    generator.generate_rand_dir_cos(gender=gender, posture='lay', num_data=250, roll_person = False, set = 15, prevent_limb_overhang = True, hands_behind_head=True)

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
