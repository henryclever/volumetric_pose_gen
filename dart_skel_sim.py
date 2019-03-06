# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group
import numpy as np
import pydart2 as pydart
from pydart2 import skeleton_builder
from dart_opengl_window import GLUTWindow

from lib_dart_skel import LibDartSkel
from capsule_body import get_capsules, joint2name, rots0

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
from pydart2.gui.opengl.scene import OpenGLScene
from time import time
import scipy.signal as signal

#import pymrt.geometry

GRAVITY = -9.81
STARTING_HEIGHT = 1.0

K = 1269.0
B = K*2
FRICTION_COEFF = 0.5

NUM_CAPSULES = 20
DART_TO_FLEX_CONV = 2.58872


class DampingController(object):
    """ Add damping force to the skeleton """
    def __init__(self, skel):
        self.skel = skel

    def compute(self):
        damping = -0.01 * self.skel.dq
        damping[1::3] *= 0.1
        return damping

class DartSkelSim(object):
    def __init__(self, render, m, gender, posture, stiffness, shiftSIDE = 0.0, shiftUD = 0.0, check_only_distal = True):

        if gender == "n":
            regs = np.load('/home/henry/git/smplify_public/code/models/regressors_locked_normalized_hybrid.npz')
        elif gender == "f":
            regs = np.load('/home/henry/git/smplify_public/code/models/regressors_locked_normalized_female.npz')
        else:
            regs = np.load('/home/henry/git/smplify_public/code/models/regressors_locked_normalized_male.npz')
        length_regs = regs['betas2lens']
        rad_regs = regs['betas2rads']
        betas = m.betas

        capsules_median = get_capsules(m, betas*0, length_regs, rad_regs)
        capsules = get_capsules(m, betas, length_regs, rad_regs)
        joint_names = joint2name
        initial_rots = rots0
        self.num_steps = 10000
        self.render_dart = render
        self.ct = 0
        self.num_dart_steps = 4

        self.has_reset_velocity1 = False
        self.has_reset_velocity2 = False

        joint_ref = list(m.kintree_table[1]) #joints
        parent_ref = list(m.kintree_table[0]) #parent of each joint
        parent_ref[0] = -1

        self.capsules = capsules


        pydart.init(verbose=True)
        print('pydart initialization OK')

        self.world = pydart.World(0.0103/self.num_dart_steps, "EMPTY") #0.003, .0002 #0.002 is stable
        self.world.set_gravity([0, 0, GRAVITY])#([0, 0,  -9.81])
        self.world.set_collision_detector(detector_type=2)
        self.world.add_empty_skeleton(_skel_name="human")

        self.force_dir_list_prev = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        self.pmat_idx_list_prev = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        self.force_loc_list_prev = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

        joint_root_loc = np.asarray(np.transpose(capsules[0].t)[0])


        joint_locs = []
        capsule_locs = []
        joint_locs_abs = []
        joint_locs_trans_abs = []
        capsule_locs_abs = []

        mJ = np.asarray(m.J)
        mJ_transformed = np.asarray(m.J_transformed)

        shift = [shiftSIDE, shiftUD, 0.0]


        red_joint_ref = joint_ref[0:20] #joints
        red_parent_ref = parent_ref[0:20] #parent of each joint
        red_parent_ref[10] = 9 #fix neck
        red_parent_ref[11] = 9 #fix l inner shoulder
        red_parent_ref[13] = 10 #fix head
        red_parent_ref[14] = 11 #fix l outer shoulder
        red_parent_ref[15] = 12 #fix r outer shoulder
        red_parent_ref[16] = 14 #fix l elbow
        red_parent_ref[17] = 15 #fix r elbow

        head_ref = [10, 13]
        leg_cap_ref = [1, 2, 4, 5]
        foot_ref = [7, 8]
        l_arm_ref = [11, 14, 16, 18]
        r_arm_ref = [12, 15, 17, 19]

        self.red_joint_ref = red_joint_ref
        self.red_parent_ref = red_parent_ref


        #make lists of the locations of the joint locations and the smplify capsule initial ends
        for i in range(np.shape(mJ)[0]):
            if i == 0:
                joint_locs.append(list(mJ[0, :] - mJ[0, :] + shift))
                joint_locs_abs.append(list(mJ[0, :] - mJ[0, :]))
                joint_locs_trans_abs.append(list(mJ_transformed[0, :] - mJ_transformed[0, :]))
                if i < 20:
                    capsule_locs.append(list(np.asarray(np.transpose(capsules[i].t)[0]) - joint_root_loc))
                    capsule_locs_abs.append(list(np.asarray(np.transpose(capsules[i].t)[0]) - joint_root_loc))
            else:
                joint_locs.append(list(mJ[i, :] - mJ[parent_ref[i], :]))
                joint_locs_abs.append(list(mJ[i, :] - mJ[0, :]))
                joint_locs_trans_abs.append(list(mJ_transformed[i, :] - mJ_transformed[0, :]))
                if i < 20:
                    capsule_locs.append(list(np.asarray(np.transpose(capsules[i].t)[0]) - np.asarray(np.transpose(capsules[red_parent_ref[i]].t)[0])))
                    capsule_locs_abs.append(list(np.asarray(np.transpose(capsules[i].t)[0]) - joint_root_loc))
                    capsule_locs_abs[i][0] += np.abs(float(capsules[0].length[0])) / 2
                    if i in [1, 2]: #shift over the legs relative to where the pelvis mid capsule is
                        capsule_locs[i][0] += np.abs(float(capsules[0].length[0])) / 2
                    if i in [3, 6, 9]: #shift over the torso segments relative to their length and their parents length to match the mid capsule
                        capsule_locs[i][0] -= (np.abs(float(capsules[i].length[0]))-np.abs(float(capsules[red_parent_ref[i]].length[0])))/2
                    if i in [10, 11, 12]: #shift over the inner shoulders and neck to match the middle of the top spine capsule
                        capsule_locs[i][0] += np.abs(float(capsules[red_parent_ref[i]].length[0])) / 2
                    if i in [3, 6, 9]: #shift over everything in the abs list to match the root
                        capsule_locs_abs[i][0] -= np.abs(float(capsules[i].length[0])) / 2

        del(joint_locs[10])
        del(joint_locs[10])
        del(joint_locs_abs[10])
        del(joint_locs_abs[10])

        self.joint_locs = joint_locs
        self.capsule_locs_abs = capsule_locs_abs


        count = 0
        root_joint_type = "FREE"

        self.cap_offsets = []
        self.cap_init_rots = []
        lowest_points = []

        for capsule in capsules:
            print "************* Capsule No.",count, joint_names[count], " joint ref: ", red_joint_ref[count]," parent_ref: ", red_parent_ref[count]," ****************"
            cap_rad = np.abs(float(capsule.rad[0]))
            cap_len = np.abs(float(capsule.length[0]))
            cap_init_rot = list(np.asarray(initial_rots[count]))


            joint_loc = joint_locs[count]
            joint_loc_abs = joint_locs_abs[count]
            capsule_loc = capsule_locs[count]
            capsule_loc_abs = capsule_locs_abs[count]

            cap_offset = [0., 0., 0.]
            if count in leg_cap_ref:
                cap_offset[1] = -cap_len/2
            if count in foot_ref: cap_offset[2] = cap_len/2
            if count in l_arm_ref: cap_offset[0] = cap_len/2
            if count in r_arm_ref: cap_offset[0] = -cap_len/2
            #if count in head_ref: cap_offset[1] = cap_len/2

            cap_offset[0] += capsule_loc_abs[0] - joint_loc_abs[0]
            cap_offset[1] += capsule_loc_abs[1] - joint_loc_abs[1] - .04
            cap_offset[2] += capsule_loc_abs[2] - joint_loc_abs[2]
            self.cap_offsets.append(np.asarray(cap_offset))
            self.cap_init_rots.append(np.asarray(cap_init_rot))



            if count == 0:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type=root_joint_type, joint_name=joint_names[count])
            elif count == 4 or count == 5:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type="REVOLUTE_X", joint_name=joint_names[count])
            elif count == 16 or count == 17:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type="REVOLUTE_Y", joint_name=joint_names[count])
            else:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type="BALL", joint_name=joint_names[count])

            lowest_points.append(np.asarray(joint_locs_trans_abs)[count, 2] - np.abs(float(capsule.rad[0])))


            count += 1


        #print "pelvis cap",
        #print np.asarray(joint_locs_trans_abs)[:, 2]
        self.STARTING_HEIGHT = STARTING_HEIGHT - np.min(np.array(lowest_points))*DART_TO_FLEX_CONV


        #add a floor-STARTING_HEIGHT / DART_TO_FLEX_CONV
        self.world.add_weld_box(width = 10.0, length = 10.0, height = 0.2, joint_loc = [0.0, 0.0, -self.STARTING_HEIGHT/DART_TO_FLEX_CONV/2 - 0.05], box_rot=[0.0, 0.0, 0.0], joint_name = "floor") #-0.05

        if posture == "sit": #need to hack the 0.5 to the right spot
            self.world.add_weld_box(width = 10.0, length = 10.0, height = 0.2, joint_loc = [0.0, 8.37, 0.0], box_rot=[np.pi/3, 0.0, 0.0], joint_name = "headrest") #-0.05

        skel = self.world.add_built_skeleton(_skel_id=0, _skel_name="human")

        if check_only_distal == True:
            skel.set_self_collision_check(True)
            skel.set_collision_filter(True)
        else:
            skel.set_self_collision_check(True)
            skel.set_adjacent_body_check(True)



        skel = LibDartSkel().assign_init_joint_angles(skel, m, root_joint_type)

        skel = LibDartSkel().assign_joint_rest_and_stiffness(skel, m, STIFFNESS = stiffness, posture = posture)

        #skel = LibDartSkel().assign_joint_limits_and_damping(skel)

        #skel = LibDartSkel().assign_capsule_friction(skel, friction = 1000.0)






        #weight the capsules appropriately
        volume = []
        volume_median = []
        for body_ct in range(NUM_CAPSULES):
            #give the capsules a weight propertional to their volume
            cap_rad = np.abs(float(capsules[body_ct].rad[0]))
            cap_len = np.abs(float(capsules[body_ct].length[0]))

            cap_rad_median = np.abs(float(capsules_median[body_ct].rad[0]))
            cap_len_median = np.abs(float(capsules_median[body_ct].length[0]))

            volume.append(np.pi*np.square(cap_rad)*(cap_rad*4/3 + cap_len))
            volume_median.append(np.pi*np.square(cap_rad_median)*(cap_rad_median*4/3 + cap_len_median))

        self.volume = volume
        self.volume_median = volume_median

        volume_torso = volume[0] + volume[3] + volume[6] + volume[9] + volume[11] + volume[12]
        volume_head = volume[10] + volume[13]

        #Human Body Dynamics: Classical Mechanics and Human Movement by Aydin Tozeren, the average percentage of weight for each body part is as follows:
        #Trunk(Chest, back and abdomen)- 50.80,  Head - 7.30, Thigh - 9.88 x 2, Lower leg - 4.65 x 2, Foot - 1.45 x 2, Upper arm - 2.7 x 2, Forearm - 1.60 x 2, Hand - 0.66 x 2,
        #Trunk(Chest, back and abdomen) Women- 50.80,  Head - 9.40, Thigh - 8.30 x 2, Lower leg - 5.50 x 2, Foot - 1.20 x 2, Upper arm - 2.7 x 2, Forearm - 1.60 x 2, Hand - 0.50 x 2,
        #Trunk(Chest, back and abdomen) Men - 48.30,  Head - 7.10, Thigh - 10.50 x 2, Lower leg - 4.50 x 2, Foot - 1.50 x 2, Upper arm - 3.3 x 2, Forearm - 1.90 x 2, Hand - 0.60 x 2,
        if gender == "f":
            BODY_MASS = 54.4 #kg
            skel.bodynodes[0].set_mass(BODY_MASS * 0.5080 * (volume[0]/volume_torso) * (volume[0]/volume_median[0]))
            skel.bodynodes[1].set_mass(BODY_MASS * 0.0830 * (volume[1]/volume_median[1]))
            skel.bodynodes[2].set_mass(BODY_MASS * 0.0830 * (volume[2]/volume_median[2]))
            skel.bodynodes[3].set_mass(BODY_MASS * 0.5080 * (volume[3]/volume_torso) * (volume[3]/volume_median[3]))
            skel.bodynodes[4].set_mass(BODY_MASS * 0.0550 * (volume[4]/volume_median[4]))
            skel.bodynodes[5].set_mass(BODY_MASS * 0.0550 * (volume[5]/volume_median[5]))
            skel.bodynodes[6].set_mass(BODY_MASS * 0.5080 * (volume[6]/volume_torso) * (volume[6]/volume_median[6]))
            skel.bodynodes[7].set_mass(BODY_MASS * 0.0120 * (volume[7]/volume_median[7]))
            skel.bodynodes[8].set_mass(BODY_MASS * 0.0120 * (volume[8]/volume_median[8]))
            skel.bodynodes[9].set_mass(BODY_MASS * 0.5080 * (volume[9]/volume_torso) * (volume[9]/volume_median[9]))
            skel.bodynodes[10].set_mass(BODY_MASS * 0.0940 * (volume[10]/volume_head) * (volume[10]/volume_median[10]))
            skel.bodynodes[11].set_mass(BODY_MASS * 0.5080 * (volume[11]/volume_torso) * (volume[11]/volume_median[11]))
            skel.bodynodes[12].set_mass(BODY_MASS * 0.5080 * (volume[12]/volume_torso) * (volume[12]/volume_median[12]))
            skel.bodynodes[13].set_mass(BODY_MASS * 0.0940 * (volume[13]/volume_head) * (volume[13]/volume_median[13]))
            skel.bodynodes[14].set_mass(BODY_MASS * 0.0270 * (volume[14]/volume_median[14]))
            skel.bodynodes[15].set_mass(BODY_MASS * 0.0270 * (volume[15]/volume_median[15]))
            skel.bodynodes[16].set_mass(BODY_MASS * 0.0160 * (volume[16]/volume_median[16]))
            skel.bodynodes[17].set_mass(BODY_MASS * 0.0160 * (volume[17]/volume_median[17]))
            skel.bodynodes[18].set_mass(BODY_MASS * 0.0050 * (volume[18]/volume_median[18]))
            skel.bodynodes[19].set_mass(BODY_MASS * 0.0050 * (volume[19]/volume_median[19]))
        else:
            BODY_MASS = 72.6 #kg
            skel.bodynodes[0].set_mass(BODY_MASS * 0.4830 * (volume[0]/volume_torso) * (volume[0]/volume_median[0]))
            skel.bodynodes[1].set_mass(BODY_MASS * 0.1050 * (volume[1]/volume_median[1]))
            skel.bodynodes[2].set_mass(BODY_MASS * 0.1050 * (volume[2]/volume_median[2]))
            skel.bodynodes[3].set_mass(BODY_MASS * 0.4830 * (volume[3]/volume_torso) * (volume[3]/volume_median[3]))
            skel.bodynodes[4].set_mass(BODY_MASS * 0.0450 * (volume[4]/volume_median[4]))
            skel.bodynodes[5].set_mass(BODY_MASS * 0.0450 * (volume[5]/volume_median[5]))
            skel.bodynodes[6].set_mass(BODY_MASS * 0.4830 * (volume[6]/volume_torso) * (volume[6]/volume_median[6]))
            skel.bodynodes[7].set_mass(BODY_MASS * 0.0150 * (volume[7]/volume_median[7]))
            skel.bodynodes[8].set_mass(BODY_MASS * 0.0150 * (volume[8]/volume_median[8]))
            skel.bodynodes[9].set_mass(BODY_MASS * 0.4830 * (volume[9]/volume_torso) * (volume[9]/volume_median[9]))
            skel.bodynodes[10].set_mass(BODY_MASS * 0.0710 * (volume[10]/volume_head) * (volume[10]/volume_median[10]))
            skel.bodynodes[11].set_mass(BODY_MASS * 0.4830 * (volume[11]/volume_torso) * (volume[11]/volume_median[11]))
            skel.bodynodes[12].set_mass(BODY_MASS * 0.4830 * (volume[12]/volume_torso * (volume[12]/volume_median[12])))
            skel.bodynodes[13].set_mass(BODY_MASS * 0.0710 * (volume[13]/volume_head) * (volume[13]/volume_median[13]))
            skel.bodynodes[14].set_mass(BODY_MASS * 0.0330 * (volume[14]/volume_median[14]))
            skel.bodynodes[15].set_mass(BODY_MASS * 0.0330 * (volume[15]/volume_median[15]))
            skel.bodynodes[16].set_mass(BODY_MASS * 0.0190 * (volume[16]/volume_median[16]))
            skel.bodynodes[17].set_mass(BODY_MASS * 0.0190 * (volume[17]/volume_median[17]))
            skel.bodynodes[18].set_mass(BODY_MASS * 0.0060 * (volume[18]/volume_median[18]))
            skel.bodynodes[19].set_mass(BODY_MASS * 0.0060 * (volume[19]/volume_median[19]))

        body_mass = 0.0
        #set the mass moment of inertia matrices
        for body_ct in range(NUM_CAPSULES):
            radius = np.abs(float(capsules[body_ct].rad[0]))
            length = np.abs(float(capsules[body_ct].length[0]))
            radius2 = radius * radius
            length2 = length * length
            mass = skel.bodynodes[body_ct].m

            cap_init_rot = list(np.asarray(initial_rots[body_ct]))

            volumeCylinder = np.pi*radius2*length
            volumeSphere = np.pi*radius*radius*radius*4/3

            density = mass / (volumeCylinder + volumeSphere)
            massCylinder = density * volumeCylinder
            massSphere = density * volumeSphere
            Ixx = massCylinder * (length2 / 12.0 + radius2 / 4.0) + massSphere * (length2 + (3.0 / 8.0) * length * radius + (2.0 / 5.0) * radius2)
            Izz = massCylinder * (radius2 / 2.0) + massSphere * ((2.0 / 5.0) * radius2)

            RotMatInit = LibDartSkel().eulerAnglesToRotationMatrix([np.pi/2, 0.0, 0.0])
            RotMat = LibDartSkel().eulerAnglesToRotationMatrix(cap_init_rot)
            I = np.matmul(np.matmul(RotMatInit, RotMat), np.asarray([ Ixx, Izz, Ixx]))
            Ixx = np.abs(I[0])
            Iyy = np.abs(I[1])
            Izz = np.abs(I[2])
            #print body_ct, I

            skel.bodynodes[body_ct].set_inertia_entries(Ixx, Iyy, Izz)

            body_mass += skel.bodynodes[body_ct].m


        print "Body mass is: ", body_mass, "kg"
        self.body_node = 9 #need to solve for the body node that corresponds to a force using flex.
        self.force = np.asarray([0.0, 100.0, 100.0])
        self.offset_from_centroid = np.asarray([-0.15, 0.0, 0.0])


        self.pmat_red_all = np.load("/home/henry/git/volumetric_pose_gen/data/pmat_red.npy")
        self.force_dir_red_dart_all = np.load("/home/henry/git/volumetric_pose_gen/data/force_dir_red.npy")
        for element in range(len(self.force_dir_red_dart_all)):
            self.force_dir_red_dart_all[element] = (np.multiply(np.asarray(self.force_dir_red_dart_all[element]),np.expand_dims(np.asarray(self.pmat_red_all[element]), axis = 1)))
        self.force_loc_red_dart_all = np.load("/home/henry/git/volumetric_pose_gen/data/force_loc_red.npy").tolist()
        self.nearest_capsule_list_all = np.load("/home/henry/git/volumetric_pose_gen/data/nearest_capsule.npy").tolist()


        print('init pose = %s' % skel.q)
        skel.controller = DampingController(skel)


        #now setup the open GL window
        self.title = "GLUT Window"
        self.window_size = (1280, 720)
        self.scene = OpenGLScene(*self.window_size)

        self.mouseLastPos = None
        self.is_simulating = False
        self.is_animating = False
        self.frame_index = 0
        self.capture_index = 0

        self.force_application_count = 0
        self.count = 0



        self.zi = []
        self.b = []
        self.a = []
        for i in range(60):
            b, a = signal.butter(1, 0.05, analog=False)
            self.b.append(b)
            self.a.append(a)
            self.zi.append(signal.lfilter_zi(self.b[-1], self.a[-1]))



    def getCapsuleVolumes(self):

        self.run_sim_step()
        contacts = self.world.collision_result.contact_sets

        contacts_with_all_nodes = []

        for body_ct in range(NUM_CAPSULES):
            #give the capsules a weight propertional to their volume
            cap_rad = np.abs(float(self.capsules[body_ct].rad[0]))
            cap_len = np.abs(float(self.capsules[body_ct].length[0]))
            print body_ct, cap_len, cap_rad,  self.capsule_locs_abs[body_ct], self.volume[body_ct]

            contacts_with_node = []
            for contact in contacts: #check all possible contacts
                if body_ct in contact: #check if the contact set includes the node we're looping through
                    for bn_contact in contact: #get the body node that the node we're looping through is in contact with
                        if body_ct != bn_contact:
                            contacts_with_node.append(bn_contact)

            print contacts_with_node
            contacts_with_all_nodes.append(contacts_with_node)

        for body_ct in range(NUM_CAPSULES):
            cap_rad = np.abs(float(self.capsules[body_ct].rad[0]))
            cap_len = np.abs(float(self.capsules[body_ct].length[0]))


            if body_ct in [0, 3, 6, 9, 11, 12, 14, 15, 16, 17, 18, 19]:

                x_range = int((self.capsule_locs_abs[body_ct][0] - cap_len - cap_rad)*500), int((self.capsule_locs_abs[body_ct][0] + cap_len + cap_rad)*500)
                y_range = int((self.capsule_locs_abs[body_ct][1] - cap_rad)*500), int((self.capsule_locs_abs[body_ct][1] + cap_rad)*500)
                z_range = int((self.capsule_locs_abs[body_ct][2] - cap_rad)*500), int((self.capsule_locs_abs[body_ct][2] + cap_rad)*500)
                voxel_space = np.zeros((x_range[1]-x_range[0], y_range[1] - y_range[0], y_range[1] - y_range[0]))
                #voxel_space[]
                print np.shape(voxel_space)

                #voxel_space[:, :, :, 0] = [x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]]




        #get sphere
        radius = 4
        xx, yy, zz = np.mgrid[0:radius*2+1, 0:radius*2+1, 0:radius*2+1]
        sphere = (xx - radius) ** 2 + (yy - radius) ** 2 + (zz - radius) ** 2

        print sphere

        sphere[sphere <= radius**2] = 1
        sphere[sphere > radius**2] = 0
        print sphere

        #get cylinder


            #print pymrt.geometry.sphere(3, 1)


        print contacts

        return 0



        #print "joint locs",
    def destroyWorld(self):
        self.world.destroy()

    def initGL(self, w, h):
        self.scene.init()

    def resizeGL(self, w, h):
        self.scene.resize(w, h)

    def drawGL(self, ):
        self.scene.render(self.world)
        # GLUT.glutSolidSphere(0.3, 20, 20)  # Default object for debugging
        GLUT.glutSwapBuffers()

    # The function called whenever a key is pressed.
    # Note the use of Python tuples to pass in: (key, x, y)
    def keyPressed(self, key, x, y):
        keycode = ord(key)
        key = key.decode('utf-8')
        # print("key = [%s] = [%d]" % (key, ord(key)))

        # n = sim.num_frames()
        if keycode == 27:
            GLUT.glutDestroyWindow(self.window)
            sys.exit()
        elif key == ' ':
            self.is_simulating = not self.is_simulating
            self.is_animating = False
            print("self.is_simulating = %s" % self.is_simulating)
        elif key == 'a':
            self.is_animating = not self.is_animating
            self.is_simulating = False
            print("self.is_animating = %s" % self.is_animating)
        elif key == ']':
            self.frame_index = (self.frame_index + 1) % self.world.num_frames()
            print("frame = %d/%d" % (self.frame_index, self.world.num_frames()))
            if hasattr(self.world, "set_frame"):
                self.world.set_frame(self.frame_index)
        elif key == '[':
            self.frame_index = (self.frame_index - 1) % self.world.num_frames()
            print("frame = %d/%d" % (self.frame_index, self.world.num_frames()))
            if hasattr(self.world, "set_frame"):
                self.world.set_frame(self.frame_index)
        elif key == 'c':
            self.capture()

    def mouseFunc(self, button, state, x, y):
        if state == 0:  # Mouse pressed
            self.mouseLastPos = np.array([x, y])
        elif state == 1:
            self.mouseLastPos = None

    def motionFunc(self, x, y):
        dx = x - self.mouseLastPos[0]
        dy = y - self.mouseLastPos[1]
        modifiers = GLUT.glutGetModifiers()
        tb = self.scene.tb
        if modifiers == GLUT.GLUT_ACTIVE_SHIFT:
            tb.zoom_to(dx, -dy)
        elif modifiers == GLUT.GLUT_ACTIVE_CTRL:
            tb.trans_to(dx, -dy)
        else:
            tb.drag_to(x, y, dx, -dy)
        self.mouseLastPos = np.array([x, y])

    def idle(self):
        if self.world is None:
            return

        #if self.count == self.num_steps: self.is_simulating = False



        if self.is_simulating:
            self.count += 1
            self.world.step()
            print "did a step"
            self.world.check_collision()

            if self.count%200 == 1:
                self.world.skeletons[0].reset_momentum()

            skel = self.world.skeletons[0]
            #print skel.q

            #print len(self.force_loc_red_dart_all)
            #print len(self.force_loc_red_dart_all[10])
            #print len(self.force_loc_red_dart_all[10][0])

            #sum force on capsule at COM: sum of

            time_orig = time()



            print "appending time", time() - time_orig


            #LibDartSkel().impose_force(skel=skel, body_node=9, force=self.force,
            #                           offset_from_centroid = self.offset_from_centroid, cap_offsets = self.cap_offsets,
            #                           render=True, init=False)

            #LibDartSkel().impose_force(skel=skel, body_node=6, force=self.force,
            #                           offset_from_centroid = -self.offset_from_centroid, cap_offsets = self.cap_offsets,
            #                           render=True, init=False)
            self.force_application_count += 1


            # if self.world.frame % 10 == 0:
            #     self.capture()
        elif self.is_animating:
            self.frame_index = (self.frame_index + 1) % self.world.num_frames()
            if hasattr(self.world, "set_frame"):
                self.world.set_frame(self.frame_index)

    def renderTimer(self, timer):
        GLUT.glutPostRedisplay()
        GLUT.glutTimerFunc(20, self.renderTimer, 1)

    def capture(self, ):
        print("capture! index = %d" % self.capture_index)
        from PIL import Image
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        w, h = 1280, 720
        data = GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        img = Image.fromstring("RGBA", (w, h), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        filename = "./data/captures/capture%04d.png" % self.capture_index
        img.save(filename, 'png')
        self.capture_index += 1

    def run_sim_with_window(self):
        print("\n")
        print("space bar: simulation on/off")
        print("' ': run/stop simulation")
        print("'a': run/stop animation")
        print("'[' and ']': play one frame backward and forward")

        # Init glut
        GLUT.glutInit(())
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA |
                                 GLUT.GLUT_DOUBLE |
                                 GLUT.GLUT_MULTISAMPLE |
                                 GLUT.GLUT_ALPHA |
                                 GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(*self.window_size)
        GLUT.glutInitWindowPosition(0, 0)
        self.window = GLUT.glutCreateWindow(self.title)

        # Init functions
        # glutFullScreen()
        GLUT.glutDisplayFunc(self.drawGL)
        GLUT.glutIdleFunc(self.idle)
        GLUT.glutReshapeFunc(self.resizeGL)
        GLUT.glutKeyboardFunc(self.keyPressed)
        GLUT.glutMouseFunc(self.mouseFunc)
        GLUT.glutMotionFunc(self.motionFunc)
        GLUT.glutTimerFunc(25, self.renderTimer, 1)
        self.initGL(*self.window_size)

        # Run
        GLUT.glutMainLoop()


    def run_sim_step(self, pmat_red_list = [], force_loc_red_dart = [], force_dir_red_dart = [], pmat_idx_red_dart = [], nearest_capsule_list = [], kill_dart_vel = False):
        self.world.step()
        if kill_dart_vel == True:
            self.world.skeletons[0].reset_momentum()

        max_vel = 0.0
        max_acc = 0.0
        skel = self.world.skeletons[0]


        force_dir_red_dart = (np.multiply(np.asarray(force_dir_red_dart), np.expand_dims(np.asarray(pmat_red_list), axis=1)))/10


        nearest_capsules = nearest_capsule_list


        force_dir_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        pmat_idx_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        force_loc_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        force_vel_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for idx in range(len(nearest_capsules)):
            force_dir_list[nearest_capsules[idx]].append(force_dir_red_dart[idx])
            pmat_idx_list[nearest_capsules[idx]].append(pmat_idx_red_dart[idx])
            force_loc_list[nearest_capsules[idx]].append(force_loc_red_dart[idx])


        #filter the acceleration vectors
        accel_vectors = []
        for item in range(len(force_dir_list)):
            accel_vectors.append(skel.bodynodes[item].com_linear_acceleration())
        accel_vectors = np.array(accel_vectors).flatten()
        accel_vectors_filtered = np.copy(accel_vectors)
        for i in range(len(force_dir_list)*3):
            accel_vectors_filtered[i], self.zi[i] = signal.lfilter(self.b[i], self.a[i], [accel_vectors[i]], zi=self.zi[i])
        accel_vectors_filtered = accel_vectors_filtered.reshape(len(accel_vectors.tolist())/3, 3)

        #time3 = time() - time2 - time1 - time0



        active_bn_list = []
        active_force_resultant_COM_list = []
        active_moment_at_COM_list = []



        for item in range(len(force_dir_list)):
            #print "linear v", skel.bodynodes[item].com_linear_velocity()

            #if item not in max_vel_withhold and np.linalg.norm(skel.bodynodes[item].com_linear_velocity()) > max_vel:

            if np.linalg.norm(skel.bodynodes[item].com_linear_velocity()) > max_vel:
                max_vel = np.linalg.norm(skel.bodynodes[item].com_linear_velocity())


            if np.linalg.norm(accel_vectors_filtered[item]) > max_acc:
                max_acc = np.linalg.norm(accel_vectors_filtered[item])


            if len(force_dir_list[item]) is not 0:
                item, len(force_dir_list[item])
                # find the sum of forces and the moment about the center of mass of each capsule
                #COM = skel.bodynodes[item].C + [0.96986 / DART_TO_FLEX_CONV, 2.4 / DART_TO_FLEX_CONV, self.STARTING_HEIGHT / DART_TO_FLEX_CONV]
                COM = skel.bodynodes[item].C + [1.185 / DART_TO_FLEX_CONV, 2.55 / DART_TO_FLEX_CONV, self.STARTING_HEIGHT / DART_TO_FLEX_CONV]

                #print item
                #print self.pmat_idx_list_prev[item], pmat_idx_list[item]
                #print self.force_dir_list_prev[item]
                #print "dir:", len(force_dir_list[item]), force_dir_list[item]

                #Calculate the spring force
                ##PARTICLE BASIS##
                f_spring = K*np.asarray(force_dir_list[item]) + np.asarray([0.00001, 0.00001, 0.00001])
                force_spring_COM = np.sum(f_spring, axis=0)


                #Calculate the damping force
                ##PARTICLE BASIS##
                f_damping = LibDartSkel().get_particle_based_damping_force(pmat_idx_list, self.pmat_idx_list_prev, force_dir_list, self.force_dir_list_prev, force_vel_list, item, B)
                force_damping_COM = np.sum(f_damping, axis=0)
                ##CAPSULE BASIS##
                #force_damping_COM = - B*skel.bodynodes[item].com_linear_velocity()


                #Calculate the friction force
                ##PARTICLE BASIS##
                f_normal = f_spring + f_damping
                V_capsule = skel.bodynodes[item].com_linear_velocity() + np.asarray([0.00001, 0.00001, 0.00001])
                f_friction = LibDartSkel().get_particle_based_friction_force(f_normal, V_capsule, FRICTION_COEFF)
                force_friction_COM = np.sum(f_friction, axis = 0)
                ##CAPSULE BASIS##
                #force_friction_COM = LibDartSkel().get_capsule_based_friction_force(skel.bodynodes[item], force_spring_COM, force_damping_COM, FRICTION_COEFF)


                #
                force_resultant_COM = force_spring_COM + force_damping_COM + force_friction_COM


                #Calculate the moment arm
                d_forces = force_loc_list[item] - COM

                #Calculate the moment
                ##PARTICLE BASIS##
                moments = np.cross(d_forces, f_normal)#+f_friction
                ##CAPSULE BASIS##
                #moments = np.cross(d_forces, f_spring)



                moment_at_COM = np.sum(moments, axis=0)

                active_bn_list.append(item)
                active_force_resultant_COM_list.append(force_resultant_COM)
                active_moment_at_COM_list.append(moment_at_COM)
                LibDartSkel().impose_force(skel=skel, body_node=item, force=force_resultant_COM,
                                           offset_from_centroid=np.asarray([0.0, 0.0, 0.0]),
                                           cap_offsets=self.cap_offsets,
                                           render=False, init=False)

                LibDartSkel().impose_torque(skel=skel, body_node=item, torque=moment_at_COM, init=False)


        #time4 = time() - time3 - time2 - time1 - time0

        #now apply the forces and step through dart for some repeated number of times. not particularly fast. expect 20 ms for
        for step in range(self.num_dart_steps-1):
            self.world.step()

            for active_bn in range(len(active_bn_list)):
                LibDartSkel().impose_force(skel=skel, body_node=active_bn_list[active_bn], force=active_force_resultant_COM_list[active_bn],
                                           offset_from_centroid=np.asarray([0.0, 0.0, 0.0]),
                                           cap_offsets=self.cap_offsets,
                                           render=False, init=False)

                LibDartSkel().impose_torque(skel=skel, body_node=active_bn_list[active_bn], torque=active_moment_at_COM_list[active_bn], init=False)




        #print "dart timing", time1, time2, time3, time4, time() - time4-time3-time2-time1-time0

        #this root joint position will tell us how to shift the root when we remesh the capsule model



        root_joint_pos = [skel.bodynodes[0].C[0] - self.cap_offsets[0][0]*np.cos(skel.q[2]) + self.cap_offsets[0][1]*np.sin(skel.q[2]),
                          skel.bodynodes[0].C[1] - self.cap_offsets[0][0]*np.sin(skel.q[2]) - self.cap_offsets[0][1]*np.cos(skel.q[2])]

        #print skel.bodynodes[0].C
        #print root_joint_pos
        #print self.cap_offsets[0]

        #print "appending time", time() - time_orig

        #LibDartSkel().impose_force(skel=skel, body_node=self.body_node, force=self.force, offset_from_centroid=self.offset_from_centroid, cap_offsets=self.cap_offsets, render=False, init=False)

        self.force_dir_list_prev = force_dir_list
        self.pmat_idx_list_prev = pmat_idx_list
        self.force_loc_list_prev = force_loc_list


        return skel.q, skel.bodynodes, root_joint_pos, max_vel, max_acc




    def run_simulation(self, num_steps = 100):
        self.num_steps = num_steps
        #pydart.gui.viewer.launch(world)

        #run without visualizing
        if self.render_dart == False:
            for i in range(0, num_steps):
                self.world.step()
                print "did a step"
                skel = self.world.skeletons[0]
                print skel.q


                LibDartSkel().impose_force(skel=skel, body_node=self.body_node, force=self.force, offset_from_centroid = self.offset_from_centroid, cap_offsets = self.cap_offsets, render=False, init=False)





        #run with OpenGL GLUT
        elif self.render_dart == True:
            default_camera = None
            if default_camera is not None:
                self.scene.set_camera(default_camera)
            self.run_sim_with_window()


if __name__ == '__main__':
    dss = DartSkelSim(render=True)
    dss.run_simulation()
