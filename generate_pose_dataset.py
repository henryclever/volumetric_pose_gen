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
    def __init__(self, gender):
        ## Load SMPL model (here we load the female model)

        if gender == "male":
            model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.m = load_model(model_path)



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


        return self.m, capsules, joint2name, rots0


    def generate_dataset(self, gender, num_data):
        #NEED FOR DATASET: pose Nx72, shape Nx10
        shape_pose_list = []
        contact_check_bns = [1, 2, 4, 5, 7, 8, 14, 15, 16, 17, 18, 19]
        contact_exceptions = [[9, 14],[9, 15]]


        for i in range(1000):
            shape_pose = [[],[],[]]


            generator.sample_body_shape(sampling = "UNIFORM", sigma = 0, one_side_range = 3)
            in_collision = True
            num_samplings = 0

            while in_collision == True:
                m, capsules, joint2name, rots0 = generator.map_random_selection_to_smpl_angles(alter_angles = True)

                shape_pose[0] = np.asarray(m.betas).tolist()

                dss = dart_skel_sim.DartSkelSim(render=True, m=m, capsules=capsules, joint_names=joint2name, initial_rots=rots0)

                print "stepping"
                invalid_pose = False
                #run a step to check for collisions
                dss.run_sim_step()

                #dss.world.check_collision()
                #print "checked collisions"
                #dss.run_simulation(1)
                #print dss.world.CollisionResult()
                #print dss.world.collision_result.contacted_bodies
                print dss.world.collision_result.contact_sets
                if len( dss.world.collision_result.contacted_bodies) != 0:
                    for contact_set in dss.world.collision_result.contact_sets:
                        if contact_set[0] in contact_check_bns or contact_set[1] in contact_check_bns: #consider removing spine 3 and upper legs
                            if contact_set in contact_exceptions:
                                pass

                            else:
                                #print "one of the limbs in contact"
                                #print contact_set
                                #dss.run_simulation(1)

                                print "resampling pose from the same shape, invalid pose"
                                #generator.standard_render()
                                in_collision = True
                                invalid_pose = True

                            break

                    if invalid_pose == False:
                        print "resampling shape and pose, collision not important."


                        #generator.standard_render()
                        in_collision = False
                else: # no contacts anywhere.

                    print "resampling shape and pose, no collision."
                    in_collision = False



                #dss.world.skeletons[0].remove_all_collision_pairs()


                dss.world.reset()
                dss.world.destroy()

            pose_indices = [3, 4, 5, 6, 7, 8, 12, 15, 39, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 53, 55, 58]
            pose_angles = []
            for index in pose_indices:
                pose_angles.append(float(m.pose[index]))

            shape_pose[1] = pose_indices
            shape_pose[2] = pose_angles


            shape_pose_list.append(shape_pose)

        print "SAVING! "
        #print shape_pose_list
        #pickle.dump(shape_pose_list, open("/home/henry/git/volumetric_pose_gen/valid_shape_pose_list1.pkl", "wb"))
        np.save("/home/henry/git/volumetric_pose_gen/valid_shape_pose_"+gender+"_list_"+str(num_data)+".npy", np.array(shape_pose_list))

    def generate_prechecked_pose(self, filename):
        prechecked_pose_list = np.load(filename).tolist()

        from capsule_body import get_capsules, joint2name, rots0


        print len(prechecked_pose_list)

        for shape_pose in prechecked_pose_list:
            #print shape_pose
            #print shape_pose[0]
            #print shape_pose[1]
            #print shape_pose[2]
            for idx in range(len(shape_pose[0])):
                #print shape_pose[0][idx]
                self.m.betas[idx] = shape_pose[0][idx]

            for idx in range(len(shape_pose[1])):
                print shape_pose[1][idx]
                #print self.m.pose[shape_pose[1][idx]]
                print shape_pose[2][idx]

                #self.m.pose[shape_pose[1][idx]] = shape_pose[2][idx]

            #print self.m.pose
            #print shape_pose[1]
            #print shape_pose[2]

            # self.m.pose[51] = selection_r
            capsules = get_capsules(self.m)
            joint2name = joint2name
            rots0 = rots0

            dss = dart_skel_sim.DartSkelSim(render=True, m=self.m, capsules=capsules, joint_names=joint2name, initial_rots=rots0)
            #dss.run_simulation(10000)

            generator.standard_render()

            #break





if __name__ == "__main__":

    gender = "male"
    num_data = "500"
    generator = GeneratePose(gender)
    generator.generate_dataset(gender, num_data = 25000)
    #generator.generate_prechecked_pose("/home/henry/git/volumetric_pose_gen/valid_shape_pose_"+gender+"_list_"+num_data+".npy")