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
import rospy
import tf

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
        self.m.pose[:] = np.random.rand(self.m.pose.size) * 0.
        self.m.betas[:] = np.random.rand(self.m.betas.size) * .0
        #self.m.betas[5] = 20.

        self.m.pose[0] = 0 #pitch rotation of the person in space. 0 means the person is upside down facing back. pi is standing up facing forward
        self.m.pose[1] = 0 #roll of the person in space. -pi/2 means they are tilted to their right side
        self.m.pose[2] = 0 #-np.pi/4 #yaw of the person in space, like turning around normal to the ground

        self.m.pose[3] = 0#-np.pi/4 #left hip extension (i.e. leg bends back for np.pi/2)
        self.m.pose[4] = 0#np.pi/8 #left leg yaw about hip, where np.pi/2 makes bowed leg
        self.m.pose[5] = 0#-np.pi/8 #left leg abduction (POS) /adduction (NEG)

        self.m.pose[6] = 0#-np.pi/4 #right hip extension (i.e. leg bends back for np.pi/2)
        self.m.pose[7] = 0#-np.pi/2
        self.m.pose[8] = 0#-np.pi/4 #right leg abduction (NEG) /adduction (POS)

        self.m.pose[9] = 0 #bending of spine at hips. np.pi/2 means person bends down to touch the ground
        self.m.pose[10] = 0 #twisting of spine at hips. body above spine yaws normal to the ground
        self.m.pose[11] = 0 #bending of spine at hips. np.pi/2 means person bends down sideways to touch the ground 3

        self.m.pose[12] = 0#np.pi/3 #left knee extension. (i.e. knee bends back for np.pi/2)
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
        self.m.pose[41] = 0 #left inner shoulder pitch, positive moves up

        self.m.pose[42] = 0#np.pi/4
        self.m.pose[43] = 0 #right inner shoulder yaw, positive moves forward
        self.m.pose[44] = 0 #right inner shoulder pitch, positive moves down

        self.m.pose[45] = 0 #flexion/extension of head 15

        self.m.pose[48] = 0 #left outer shoulder roll
        self.m.pose[49] = 0#-np.pi/4
        self.m.pose[50] = 0#np.pi/4 #left outer shoulder pitch

        self.m.pose[51] = 0#-np.pi/3 #right outer shoulder roll
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




    def standard_render(self):

        ## Create OpenDR renderer
        rn = ColoredRenderer()

        ## Assign attributes to renderer
        w, h = (640, 480)

        rn.camera = ProjectPoints(v=self.m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
        rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
        rn.set(v=self.m, f=self.m.f, bgcolor=np.zeros(3))

        ## Construct point light source
        rn.vc = LambertianPointLight(
            f=self.m.f,
            v=rn.v,
            num_verts=len(self.m),
            light_pos=np.array([-1000,-1000,-2000]),
            vc=np.ones_like(self.m)*.9,
            light_color=np.array([1., 1., 1.]))


        ## Show it using OpenCV
        import cv2
        cv2.imshow('render_SMPL', rn.r)
        print ('..Print any key while on the display window')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        ## Could also use matplotlib to display
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.imshow(rn.r)
        # plt.show()
        # import pdb; pdb.set_trace()




    def solve_ik_tree_smpl(self):

        # print the origin
        print self.m.J - self.m.J[0, :]


        #grab the joint positions
        r_leg_pos_origins = np.array([np.array(self.m.J[0, :]), np.array(self.m.J[2, :]), np.array(self.m.J[5, :]), np.array(self.m.J[8, :])])
        r_leg_pos_current = np.array([np.array(self.m.J_transformed[0, :]), np.array(self.m.J_transformed[2, :]), np.array(self.m.J_transformed[5, :]), np.array(self.m.J_transformed[8, :])])


        #get the IKPY solution
        r_omega_hip, r_knee_chain, IK_RK, r_ankle_chain, IK_RA = libKinematics.ikpy_leg(r_leg_pos_origins, r_leg_pos_current)


        #get the RPH values
        r_hip_angles = [IK_RK[1], IK_RA[2], IK_RK[2]]
        r_knee_angle = [IK_RA[4], 0 , 0]

        #grab the joint angle ground truth
        r_hip_angles_GT = self.m.pose[6:9]
        r_knee_angle_GT = self.m.pose[15]

        print IK_RK, 'IK_RK'
        print IK_RA, "IK_RA"


        print "right hip Angle-axis GT: ", r_hip_angles_GT
        print "right hip Angle-axis est:", r_omega_hip
        print "right hip Euler est: ", r_hip_angles
        print "right knee Angle-axis GT: ", r_knee_angle_GT
        #print "right knee Angle-axis est:", r_omega_knee
        print "right knee Euler est: ", r_knee_angle


        #grab the joint positions
        l_leg_pos_origins = np.array([np.array(self.m.J[0, :]), np.array(self.m.J[1, :]), np.array(self.m.J[4, :]), np.array(self.m.J[7, :])])
        l_leg_pos_current = np.array([np.array(self.m.J_transformed[0, :]), np.array(self.m.J_transformed[1, :]), np.array(self.m.J_transformed[4, :]), np.array(self.m.J_transformed[7, :])])

        #get the IK solution
        l_omega_hip, l_knee_chain, IK_LK, l_ankle_chain, IK_LA  = libKinematics.ikpy_leg(l_leg_pos_origins, l_leg_pos_current)

        #get the RPH values
        l_hip_angles = [IK_LK[1], IK_LA[2], IK_LK[2]]
        l_knee_angle = [IK_LA[4], 0, 0]

        #grab the joint angle ground truth
        l_hip_angles_GT = self.m.pose[3:6]
        l_knee_angle_GT = self.m.pose[12]

        print IK_LK, "IK_LK"
        print IK_LA, "IK_LA"

        print "left hip Angle-axis GT: ", l_hip_angles_GT
        print "left hip Angle-axis est: ", l_omega_hip
        print "left hip Euler estimation: ", l_hip_angles
        print "left knee Angle-axis GT: ", l_knee_angle_GT
        #print "left knee Angle-axis est: ", l_omega_knee
        print "left knee Euler est: ", l_knee_angle



        #grab the joint positions
        r_arm_pos_origins = np.array([np.array(self.m.J[0, :]), np.array(self.m.J[12, :]), np.array(self.m.J[17, :]), np.array(self.m.J[19, :]), np.array(self.m.J[21, :]) ])
        r_arm_pos_current = np.array([np.array(self.m.J_transformed[0, :]), np.array(self.m.J_transformed[12, :]), np.array(self.m.J_transformed[17, :]), np.array(self.m.J_transformed[19, :]), np.array(self.m.J_transformed[21, :])])

        #get the IK solution
        r_omega_shoulder, r_elbow_chain, IK_RE, r_wrist_chain, IK_RW = libKinematics.ikpy_right_arm(r_arm_pos_origins, r_arm_pos_current)

        #get the RPH values
        r_shoulder_angles = [IK_RW[2], IK_RE[2], IK_RE[3]]
        r_elbow_angle = [0, IK_RW[5], 0]

        #grab the joint angle ground truth
        r_shoulder_angles_GT = self.m.pose[51:54]
        r_elbow_angle_GT = self.m.pose[58]

        print IK_RE
        print IK_RW

        print "right shoulder Angle-axis GT: ", r_shoulder_angles_GT
        print "right shoulder Angle-axis est: ", r_omega_shoulder
        print "right shoulder Euler est: ", r_shoulder_angles
        print "right elbow Angle-axis GT: ", r_elbow_angle_GT
        print "right elbow Euler est: ", r_elbow_angle




        #grab the joint positions
        l_arm_pos_origins = np.array([np.array(self.m.J[0, :]), np.array(self.m.J[12, :]), np.array(self.m.J[16, :]), np.array(self.m.J[18, :]), np.array(self.m.J[20, :]) ])
        l_arm_pos_current = np.array([np.array(self.m.J_transformed[0, :]), np.array(self.m.J_transformed[12, :]), np.array(self.m.J_transformed[16, :]), np.array(self.m.J_transformed[18, :]), np.array(self.m.J_transformed[20, :])])

        #get the IK solution
        l_omega_shoulder, l_elbow_chain, IK_LE, l_wrist_chain, IK_LW = libKinematics.ikpy_left_arm(l_arm_pos_origins, l_arm_pos_current)

        #get the RPH values
        l_shoulder_angles = [IK_LW[2], IK_LE[2], IK_LE[3]]
        l_elbow_angle = [0, IK_LW[5], 0]

        #grab the joint angle ground truth
        l_shoulder_angles_GT = self.m.pose[48:51]
        l_elbow_angle_GT = self.m.pose[55]

        print IK_LE
        print IK_LW

        print "left shoulder Angle-axis GT: ", l_shoulder_angles_GT
        print "left shoulder Angle-axis est: ", l_omega_shoulder
        print "left shoulder Euler est: ", l_shoulder_angles
        print "left elbow Angle-axis GT: ", l_elbow_angle_GT
        print "left elbow Euler est: ", l_elbow_angle


        #grab the joint positions
        neck_pos_origins = np.array([np.array(self.m.J[0, :]), np.array(self.m.J[12, :]), np.array(self.m.J[15, :]) ])
        neck_pos_current = np.array([np.array(self.m.J_transformed[0, :]), np.array(self.m.J_transformed[12, :]), np.array(self.m.J_transformed[15, :]) ])

        #get the IK solution
        omega_neck, head_chain, IK_H = libKinematics.ikpy_head(neck_pos_origins, neck_pos_current)

        #get the RPH values
        neck_angles = [IK_H[1], 0, IK_H[2]]

        #grab the joint angle ground truth
        neck_angles_GT = self.m.pose[36:39]

        print IK_H

        print "neck Angle-axis GT: ", neck_angles_GT
        print "neck Angle-axis est: ", omega_neck
        print "neck Euler est: ", neck_angles


        plot_first_joint = False
        plot_all_joints = False
        if plot_first_joint:
            r_knee_chain.plot(IK_RK, self.ax)
            l_knee_chain.plot(IK_LK, self.ax)
            r_elbow_chain.plot(IK_RE, self.ax)
            l_elbow_chain.plot(IK_LE, self.ax)
            head_chain.plot(IK_H, self.ax)

        if plot_all_joints:
            r_ankle_chain.plot(IK_RA, self.ax)
            l_ankle_chain.plot(IK_LA, self.ax)
            r_wrist_chain.plot(IK_RW, self.ax)
            l_wrist_chain.plot(IK_LW, self.ax)
            head_chain.plot(IK_H, self.ax)


        Jt = np.array(self.m.J_transformed-self.m.J_transformed[0,:])
        #print Jt
        #self.ax.plot(Jt[:,0],Jt[:,1],Jt[:,2], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.5)

        #with tensorflow.Session() as sess:
        #    smpl = SMPL(model_path)
        #    J_transformed = smpl(self.m.betas, self.m.pose)

        #print J_transformed
        #print self.m.J_transformed



    def render_rviz(self):

        print self.m.J_transformed
        rospy.init_node('smpl_model')

        shift_sideways = np.zeros((24,3))
        shift_sideways[:, 0] = 1.0

        for i in range(0, 10):
            #libVisualization.rviz_publish_output(None, np.array(self.m.J_transformed))
            time.sleep(0.5)

            concatted = np.concatenate((np.array(self.m.J_transformed), np.array(self.m.J_transformed) + shift_sideways), axis = 0)
            #print concatted
            #libVisualization.rviz_publish_output(None, np.array(self.m.J_transformed) + shift_sideways)
            libVisualization.rviz_publish_output(None, concatted)
            time.sleep(0.5)


    def solve_ik_tree_yashdata(self, filename, subject, movement, verbose = True):

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

            #print "dir cos is", libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(r_hip_angles))
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
            r_omega_shoulder, r_elbow_chain, IK_RE, r_wrist_chain, IK_RW = libKinematics.ikpy_right_arm(r_arm_pos_origins,
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

            if r_elbow_angles[1] < 0:
                print "****************************", subject, movement, "***************************************************"

            # grab the joint positions
            l_arm_pos_origins = np.array([np.array(Jtc_origins[1, :]), np.array(Jtc_origins[10, :]), np.array(Jtc_origins[12, :]),
                                          np.array(Jtc_origins[3, :]), np.array(Jtc_origins[5, :])])
            l_arm_pos_current = np.array([np.array(Jtc[1, :]), np.array(Jtc[10, :]), np.array(Jtc[12, :]),
                                          np.array(Jtc[3, :]), np.array(Jtc[5, :])])

            # get the IK solution
            l_omega_shoulder, l_elbow_chain, IK_LE, l_wrist_chain, IK_LW = libKinematics.ikpy_left_arm(l_arm_pos_origins,
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

            if l_elbow_angles[1] > 0: #40ESJ LH1 #40ESJ LH2
                print "*********************************", subject, movement, "********************************************************"

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


            self.ax.plot(Jtc_origins[:, 0], Jtc_origins[:, 1], Jtc_origins[:, 2], markerfacecolor='k', markeredgecolor='k', marker='o',  markersize=5, alpha=0.5)
            self.ax.plot(Jtc[:, 0], Jtc[:, 1], Jtc[:, 2], markerfacecolor='k', markeredgecolor='k', marker='o',  markersize=5, alpha=0.5)
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(-1, 1)
            plt.show()

            angles['r_hip_euler'] = angles['r_hip_angle_axis']
            angles['r_hip_angle_axis'] = libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(angles['r_hip_euler']))
            angles['l_hip_euler'] = angles['l_hip_angle_axis']
            angles['l_hip_angle_axis'] = libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(angles['l_hip_euler']))
            angles['r_shoulder_euler'] = angles['r_shoulder_angle_axis']
            angles['r_shoulder_angle_axis'] = libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(angles['r_shoulder_euler']))
            angles['l_shoulder_euler'] = angles['l_shoulder_angle_axis']
            angles['l_shoulder_angle_axis'] = libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(angles['l_shoulder_euler']))

            new_entry = angles
            new_data.append(new_entry)


        return new_data


    def save_yash_data_with_angles(self):
        movements = ['LL', 'RL', 'LH1', 'LH2', 'LH3', 'RH1', 'RH2', 'RH3']
        subjects = ['40ESJ', 'GRTJK', 'TX887', 'WFGW9', 'WM9KJ', 'ZV7TE', 'FMNGQ']

        for subject in subjects:
            for movement in movements:

                filename = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/" + movement + ".p"
                filename_save = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/" + movement + "_angles.p"
                filename_save2 = "/home/henry/pressure_mat_angles/subject_" + subject + "/" + movement + "_angles.p"

                new_data = generator.solve_ik_tree_yashdata(filename, subject, movement, verbose = False)

                #sprint len(new_data), 'length of new data'
                with open(filename_save, 'wb') as fp:
                    pickle.dump(new_data, fp)
                print 'done saving ', filename_save

                with open(filename_save2, 'wb') as fp:
                    pickle.dump(new_data, fp)
                print 'done saving ', filename_save2




    def map_euler_angles_to_axis_angle(self):
        movements = ['LL', 'RL', 'LH1', 'LH2', 'LH3', 'RH1', 'RH2', 'RH3']
        #subjects = ['FMNGQ', '40ESJ', '4ZZZQ', '5LDJG', 'A4G4Y','G55Q1','GF5Q3', 'GRTJK', 'RQCLC', 'TSQNA', 'TX887', 'WCNOM', 'WE2SZ', 'WFGW9', 'WM9KJ', 'ZV7TE']

        subjects = ['40ESJ', 'GRTJK', 'TX887', 'WFGW9', 'WM9KJ', 'ZV7TE', 'FMNGQ']
        for subject in subjects:
            for movement in movements:
                print "subject: ", subject, " movement: ", movement
                filename = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/euler_angles/" + movement + "_angles.p"
                #filename_save = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/" + movement + "_angles.p"
                filename_save = "/home/henry/pressure_mat_angles/subject_" + subject + "/" + movement + "_angles.p"

                with open(filename, 'rb') as fp:
                    angles_data = pickle.load(fp)

                new_angles_data = []
                for entry in angles_data:
                    #print "euler", entry['r_hip_angle_axis']
                    entry['r_hip_euler'] = entry['r_hip_angle_axis']
                    entry['r_hip_angle_axis'] = libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(entry['r_hip_euler']))
                    entry['l_hip_euler'] = entry['l_hip_angle_axis']
                    entry['l_hip_angle_axis'] = libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(entry['l_hip_euler']))
                    entry['r_shoulder_euler'] = entry['r_shoulder_angle_axis']
                    entry['r_shoulder_angle_axis'] = libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(entry['r_shoulder_euler']))
                    entry['l_shoulder_euler'] = entry['l_shoulder_angle_axis']
                    entry['l_shoulder_angle_axis'] = libKinematics.dir_cos_angles_from_matrix(libKinematics.eulerAnglesToRotationMatrix(entry['l_shoulder_euler']))
                    new_angles_data.append(entry)
                    #print "euler", entry['r_hip_euler']
                    #print "angleax", entry['r_hip_angle_axis']

                with open(filename_save, 'wb') as fp:
                    pickle.dump(new_angles_data, fp)

    def map_yash_to_smpl_angles(self, verbose = True):

        movements = ['LL', 'RL', 'LH1', 'LH2', 'LH3', 'RH1', 'RH2', 'RH3']
        subjects = ['FMNGQ']#['40ESJ', '4ZZZQ', '5LDJG', 'A4G4Y','G55Q1','GF5Q3', 'GRTJK', 'RQCLC', 'TSQNA', 'TX887', 'WCNOM', 'WE2SZ', 'WFGW9', 'WM9KJ', 'ZV7TE']


        for subject in subjects:
            for movement in movements:
                print "subject: ", subject, " movement: ", movement
                filename = "/media/henry/multimodal_data_2/pressure_mat_pose_data/subject_" + subject + "/" + movement + "_angles.p"

                with open(filename, 'rb') as fp:
                    angles_data = pickle.load(fp)

                for entry in angles_data:


                    #entry = angles_data[50]

                    self.m.pose[6] = entry['r_hip_angle_axis'][0]
                    self.m.pose[7] = entry['r_hip_angle_axis'][1]/2
                    self.m.pose[8] = entry['r_hip_angle_axis'][2]
                    if verbose == True: print 'r hip', self.m.pose[6:9]

                    self.m.pose[15] = entry['r_knee_angle_axis'][0]
                    self.m.pose[16] = entry['r_hip_angle_axis'][1]/2
                    if verbose == True: print 'r knee', self.m.pose[15:18]


                    self.m.pose[3] = entry['l_hip_angle_axis'][0]
                    self.m.pose[4] = entry['l_hip_angle_axis'][1]/2
                    self.m.pose[5] = entry['l_hip_angle_axis'][2]
                    if verbose == True: print 'l hip', self.m.pose[3:6]

                    self.m.pose[12] = entry['l_knee_angle_axis'][0]
                    self.m.pose[13] = entry['l_hip_angle_axis'][1]/2
                    if verbose == True: print 'l knee', self.m.pose[12:15]


                    self.m.pose[51] = entry['r_shoulder_angle_axis'][0]*2/3
                    self.m.pose[52] = entry['r_shoulder_angle_axis'][1]*2/3
                    self.m.pose[53] = entry['r_shoulder_angle_axis'][2]*2/3
                    self.m.pose[42] = entry['r_shoulder_angle_axis'][0]*1/3
                    self.m.pose[43] = entry['r_shoulder_angle_axis'][1]*1/3
                    self.m.pose[44] = entry['r_shoulder_angle_axis'][2]*1/3
                    if verbose == True: print 'r shoulder', self.m.pose[51:54] + self.m.pose[42:45]

                    self.m.pose[58] = entry['r_elbow_angle_axis'][1]
                    if verbose == True: print 'r elbow', self.m.pose[57:60]


                    self.m.pose[48] = entry['l_shoulder_angle_axis'][0]*2/3
                    self.m.pose[49] = entry['l_shoulder_angle_axis'][1]*2/3
                    self.m.pose[50] = entry['l_shoulder_angle_axis'][2]*2/3
                    self.m.pose[39] = entry['l_shoulder_angle_axis'][0]*1/3
                    self.m.pose[40] = entry['l_shoulder_angle_axis'][1]*1/3
                    self.m.pose[41] = entry['l_shoulder_angle_axis'][2]*1/3
                    if verbose == True: print 'l shoulder', self.m.pose[48:51] + self.m.pose[39:42]

                    self.m.pose[55] = entry['l_elbow_angle_axis'][1]
                    if verbose == True: print 'l elbow', self.m.pose[54:57]

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
        print len(capsules)

        #put these capsules into dart based on these angles. Make Dart joints only as necessary.
        #Use the positions found in dart to update positions in FleX. Do not use angles in Flex
        #repeat: do not need a forward kinematics model in FleX! Just need the capsule positions and radii. Can potentially get rotation from the Capsule end positions.
        #Find IK solution at the very end.


        return self.m, capsules, joint2name, rots0



if __name__ == "__main__":
    generator = GeneratePose(sampling = "UNIFORM", sigma = 0, one_side_range = 3)
    generator.ax = plt.figure().add_subplot(111, projection='3d')
    #generator.solve_ik_tree_smpl()
    generator.save_yash_data_with_angles()
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
