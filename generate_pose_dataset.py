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
    def __init__(self, sampling = "NORMAL", sigma = 0, one_side_range = 0):
        ## Load SMPL model (here we load the female model)

        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
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

        self.sample_body_shape(sampling, sigma, one_side_range)



    def sample_body_shape(self, sampling, sigma, one_side_range):
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


        return self.m, capsules, joint2name, rots0



if __name__ == "__main__":

    generator = GeneratePose(sampling="UNIFORM", sigma=0, one_side_range=3)
    #NEED FOR DATASET: pose Nx72, shape Nx10

    for i in range(5):


        generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)
        in_collision = True

        while in_collision == True:
            m, capsules, joint2name, rots0 = generator.map_random_selection_to_smpl_angles(alter_angles = True)

            dss = dart_skel_sim.DartSkelSim(render=True, m=m, capsules=capsules, joint_names=joint2name, initial_rots=rots0)

            print "stepping"
            #run a step to check for collisions
            dss.run_sim_step()

            #print dss.world.CollisionResult()
            for colliding_object in dss.world.collision_result.contacted_bodies:
                if colliding_object.id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 17, 18, 19]:
                    print dss.world.collision_result.contacted_bodies
                    generator.standard_render()
                    #dss.run_simulation(1)

                    break
            #

            in_collision = False

            dss.world.reset()
            dss.world.remove_all_constraints()
            dss.world.destroy()