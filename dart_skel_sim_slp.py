

import numpy as np
import pydart2 as pydart
from pydart2 import skeleton_builder
from dart_opengl_window import GLUTWindow

from lib_dart_skel_slp import LibDartSkel
from capsule_body import get_capsules, joint2name, rots0

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
from pydart2.gui.opengl.scene import OpenGLScene
from time import time
import scipy.signal as signal
from time import sleep
import lib_kinematics

#import pymrt.geometry

GRAVITY = -9.81
STARTING_HEIGHT = 1.3

#K = 1269.0
#K = 1521.1
K = 1196.5
#K = 704.5
#K = 1042.05
B = K*4
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
    def __init__(self, render, m, gender, posture, stiffness, shiftSIDE = 0.0, shiftUD = 0.0, check_only_distal = True, filepath_prefix = '/home/henry', add_floor = True, volume = None):
        #print('dss m', m.pose)
        if gender == "n":
            regs = np.load(filepath_prefix+'/git/smplify_public_hc/code/models/regressors_locked_normalized_hybrid.npz')
        elif gender == "f":
            regs = np.load(filepath_prefix+'/git/smplify_public_hc/code/models/regressors_locked_normalized_female.npz')
        else:
            regs = np.load(filepath_prefix+'/git/smplify_public_hc/code/models/regressors_locked_normalized_male.npz')
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
        self.glute_height = 1.0

        self.has_reset_velocity1 = False
        self.has_reset_velocity2 = False

        joint_ref = list(m.kintree_table[1]) #joints
        parent_ref = list(m.kintree_table[0]) #parent of each joint
        parent_ref[0] = -1

        self.capsules = capsules
        self.step_num = 0

        pydart.init(verbose=True)
        print('pydart initialization OK')
        #print m.J_transformed
        #print shiftSIDE
        #print shiftUD

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
        #print np.array(m.J_transformed) - np.array(m.J[0, :])
        #print np.array(m.J_transformed) - np.array(m.J[0, :]) + np.array(shift)
        #print (np.array(m.J_transformed) - np.array(m.J[0, :]) + np.array(shift))*2.58872
        #print (np.array(m.J_transformed) - np.array(m.J[0, :]) + np.array(shift))*2.58872 + np.array([1.185, 2.55, 0.0]), 'final mesh tree'

        #print np.array(m.J_transformed) - np.array(m.J[0, :]) + np.array(shift) + np.array([0.0, -0.04, 0.0])
        #print m.pose


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
        self.root_capsule_rad = float(capsules[0].rad[0])
        if self.root_capsule_rad < 0.0001: self.root_capsule_rad = 0.0001

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
            cap_rad = float(capsule.rad[0])
            if cap_rad < 0.0001: cap_rad = 0.0001
            cap_len = float(capsule.length[0])
            if cap_len < 0.0001: cap_len = 0.0001
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
            #elif count == 4 or count == 5:
            #    self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
            #                           cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
            #                           joint_type="REVOLUTE_X", joint_name=joint_names[count])
            #elif count == 16 or count == 17:
            #    self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
            #                           cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
            #                           joint_type="REVOLUTE_Y", joint_name=joint_names[count])
            else:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type="BALL", joint_name=joint_names[count])

            lowest_pt_cap_rad = float(capsule.rad[0])
            if lowest_pt_cap_rad < 0.0001: lowest_pt_cap_rad = 0.0001
            lowest_points.append(np.asarray(joint_locs_trans_abs)[count, 2] - lowest_pt_cap_rad)


            count += 1


        #print "pelvis cap",
        #print np.asarray(joint_locs_trans_abs)[:, 2]
        self.STARTING_HEIGHT = STARTING_HEIGHT - np.min(np.array(lowest_points))*DART_TO_FLEX_CONV


        if add_floor == True:
            #add a floor-STARTING_HEIGHT / DART_TO_FLEX_CONV
            self.world.add_weld_box(width = 10.0, length = 10.0, height = 0.2, joint_loc = [0.0, 0.0, -self.STARTING_HEIGHT/DART_TO_FLEX_CONV/2 - 0.05], box_rot=[0.0, 0.0, 0.0], joint_name = "floor") #-0.05

            print "added floor!!!!"

            if posture == "sit": #need to hack the 0.5 to the right spot
                self.world.add_weld_box(width = 10.0, length = 10.0, height = 0.2, joint_loc = [0.0, 0.43, 0.0], box_rot=[np.pi/3, 0.0, 0.0], joint_name = "headrest") #-0.05

        skel = self.world.add_built_skeleton(_skel_id=0, _skel_name="human")


        print skel.bodynodes[0].C, 'SKEL BODYNODES 0 POS'
        print skel.bodynodes[0].C[0]*2.58872 + 1.185, skel.bodynodes[0].C[1]*2.58872 + 2.55

        if check_only_distal == True:
            skel.set_self_collision_check(True)
            skel.set_collision_filter(True)
        else:
            skel.set_self_collision_check(True)
            skel.set_adjacent_body_check(True)



        if gender == "f":
            #volume_median = [0.015277643666666658, 0.007676252166666666, 0.007705970166666665, 0.007150662166666678, 0.004637961,
            #                 0.0046260565, 0.006919925999999999, 0.0009656045, 0.000978959, 0.010161063749999996,
            #                 0.0014945347499999992, 0.002064434, 0.002040836916666667, 0.003933601916666668, 0.00184907,
            #                 0.0018632635, 0.0009073805, 0.0009390935, 0.000400365, 0.0004192055]
            volume_median = [0.020918456, 0.010185206, 0.010185206, 0.011688111, 0.004861564,
                             0.004851375, 0.012607848, 0.001170955, 0.001184272, 0.015511426,
                             0.003288661, 0.002799336, 0.002799336, 0.005684597, 0.002302096,
                             0.002304728, 0.001159883, 0.001192307, 0.000451576, 0.000483304]
        else:
            #volume_median = [0.014475041666666666, 0.007873465666666668, 0.007963612000000004, 0.009262944666666675, 0.0052877485,
            #                 0.0052843175, 0.006637316166666669, 0.001300277, 0.001316104, 0.011297247333333335,
            #                 0.002356231416666665, 0.003092792416666667, 0.0030801008333333343, 0.004614448, 0.0022653085,
            #                 0.002294299, 0.0014234765, 0.001413545, 0.000390882, 0.000409517]
            volume_median = [0.019681484, 0.010447408, 0.01054436, 0.013953241, 0.005571068,
                             0.005571068, 0.012321504, 0.00156056, 0.001583692, 0.016830335,
                             0.004977728, 0.003976808, 0.003944192, 0.006793848, 0.00284046,
                             0.0028836, 0.001737588, 0.001753752, 0.000454112, 0.000483304]


        if volume is None:
            #weight the capsules appropriately
            print "volume is none!!!"
            volume = []
            #volume_median = []
            for body_ct in range(NUM_CAPSULES):
                #give the capsules a weight propertional to their volume
                cap_rad = float(capsules[body_ct].rad[0])
                if cap_rad < 0.0001: cap_rad = 0.0001
                cap_len = float(capsules[body_ct].length[0])
                if cap_len < 0.0001: cap_len = 0.0001

                #cap_rad_median = float(capsules_median[body_ct].rad[0])
                #if cap_rad_median < 0.0001: cap_rad_median = 0.0001
                #cap_len_median = float(capsules_median[body_ct].length[0])
                #if cap_len_median < 0.0001: cap_len_median = 0.0001

                volume.append(np.pi*np.square(cap_rad)*(cap_rad*4/3 + cap_len))
                #volume_median.append(np.pi*np.square(cap_rad_median)*(cap_rad_median*4/3 + cap_len_median))

        #print volume
        #sleep(1)


        self.volume = volume
        self.volume_median = volume_median

        volume_torso = volume[0] + volume[3] + volume[6] + volume[9] + volume[11] + volume[12]
        volume_head = volume[10] + volume[13]
        volume_torso_median = volume_median[0] + volume_median[3] + volume_median[6] + volume_median[9] + volume_median[11] + volume_median[12]
        volume_head_median = volume_median[10] + volume_median[13]


        #Human Body Dynamics: Classical Mechanics and Human Movement by Aydin Tozeren, the average percentage of weight for each body part is as follows:
        #Trunk(Chest, back and abdomen)- 50.80,  Head - 7.30, Thigh - 9.88 x 2, Lower leg - 4.65 x 2, Foot - 1.45 x 2, Upper arm - 2.7 x 2, Forearm - 1.60 x 2, Hand - 0.66 x 2,
        #Trunk(Chest, back and abdomen) Women- 50.80,  Head - 9.40, Thigh - 8.30 x 2, Lower leg - 5.50 x 2, Foot - 1.20 x 2, Upper arm - 2.7 x 2, Forearm - 1.60 x 2, Hand - 0.50 x 2,
        #Trunk(Chest, back and abdomen) Men - 48.30,  Head - 7.10, Thigh - 10.50 x 2, Lower leg - 4.50 x 2, Foot - 1.50 x 2, Upper arm - 3.3 x 2, Forearm - 1.90 x 2, Hand - 0.60 x 2,


        #print "DART SKEL SIM! "
        #print volume_torso, volume_head
        #print volume_median[0] + volume_median[3] + volume_median[6] + volume_median[9] + volume_median[11] + volume_median[12], volume_median[10] + volume_median[13]

        if gender == "f":
            BODY_MASS = 62.5 #kg median height: -1.658 or about 5 foot 5.3 in
            skel.bodynodes[0].set_mass(BODY_MASS * 0.5080 * (volume_median[0]/volume_torso_median) * (volume[0]/volume_median[0]) + 0.00001)
            skel.bodynodes[1].set_mass(BODY_MASS * 0.0830 * (volume[1]/volume_median[1]) + 0.00001)
            skel.bodynodes[2].set_mass(BODY_MASS * 0.0830 * (volume[2]/volume_median[2]) + 0.00001)
            skel.bodynodes[3].set_mass(BODY_MASS * 0.5080 * (volume_median[3]/volume_torso_median) * (volume[3]/volume_median[3]) + 0.00001)
            skel.bodynodes[4].set_mass(BODY_MASS * 0.0550 * (volume[4]/volume_median[4]) + 0.00001)
            skel.bodynodes[5].set_mass(BODY_MASS * 0.0550 * (volume[5]/volume_median[5]) + 0.00001)
            skel.bodynodes[6].set_mass(BODY_MASS * 0.5080 * (volume_median[6]/volume_torso_median) * (volume[6]/volume_median[6])+ 0.00001)
            skel.bodynodes[7].set_mass(BODY_MASS * 0.0120 * (volume[7]/volume_median[7]) + 0.00001)
            skel.bodynodes[8].set_mass(BODY_MASS * 0.0120 * (volume[8]/volume_median[8]) + 0.00001)
            skel.bodynodes[9].set_mass(BODY_MASS * 0.5080 * (volume_median[9]/volume_torso_median) * (volume[9]/volume_median[9]) + 0.00001)
            skel.bodynodes[10].set_mass(BODY_MASS * 0.0940 * (volume_median[10]/volume_head_median) * (volume[10]/volume_median[10]) + 0.00001)
            skel.bodynodes[11].set_mass(BODY_MASS * 0.5080 * (volume_median[11]/volume_torso_median) * (volume[11]/volume_median[11]) + 0.00001)
            skel.bodynodes[12].set_mass(BODY_MASS * 0.5080 * (volume_median[12]/volume_torso_median) * (volume[12]/volume_median[12]) + 0.00001)
            skel.bodynodes[13].set_mass(BODY_MASS * 0.0940 * (volume_median[13]/volume_head_median) * (volume[13]/volume_median[13]) + 0.00001)
            skel.bodynodes[14].set_mass(BODY_MASS * 0.0270 * (volume[14]/volume_median[14]) + 0.00001)
            skel.bodynodes[15].set_mass(BODY_MASS * 0.0270 * (volume[15]/volume_median[15]) + 0.00001)
            skel.bodynodes[16].set_mass(BODY_MASS * 0.0160 * (volume[16]/volume_median[16]) + 0.00001)
            skel.bodynodes[17].set_mass(BODY_MASS * 0.0160 * (volume[17]/volume_median[17]) + 0.00001)
            skel.bodynodes[18].set_mass(BODY_MASS * 0.0050 * (volume[18]/volume_median[18]) + 0.00001)
            skel.bodynodes[19].set_mass(BODY_MASS * 0.0050 * (volume[19]/volume_median[19]) + 0.00001)
        else:
            BODY_MASS = 78.4 #kg . median height: 1.791 or about 5 foot 10.5 in
            skel.bodynodes[0].set_mass(BODY_MASS * 0.4830 * (volume_median[0]/volume_torso_median) * (volume[0]/volume_median[0]) + 0.00001)
            skel.bodynodes[1].set_mass(BODY_MASS * 0.1050 * (volume[1]/volume_median[1]) + 0.00001)
            skel.bodynodes[2].set_mass(BODY_MASS * 0.1050 * (volume[2]/volume_median[2]) + 0.00001)
            skel.bodynodes[3].set_mass(BODY_MASS * 0.4830 * (volume_median[3]/volume_torso_median) * (volume[3]/volume_median[3]) + 0.00001)
            skel.bodynodes[4].set_mass(BODY_MASS * 0.0450 * (volume[4]/volume_median[4]) + 0.00001)
            skel.bodynodes[5].set_mass(BODY_MASS * 0.0450 * (volume[5]/volume_median[5]) + 0.00001)
            skel.bodynodes[6].set_mass(BODY_MASS * 0.4830 * (volume_median[6]/volume_torso_median) * (volume[6]/volume_median[6]) + 0.00001)
            skel.bodynodes[7].set_mass(BODY_MASS * 0.0150 * (volume[7]/volume_median[7]) + 0.00001)
            skel.bodynodes[8].set_mass(BODY_MASS * 0.0150 * (volume[8]/volume_median[8]) + 0.00001)
            skel.bodynodes[9].set_mass(BODY_MASS * 0.4830 * (volume_median[9]/volume_torso_median) * (volume[9]/volume_median[9]) + 0.00001)
            skel.bodynodes[10].set_mass(BODY_MASS * 0.0710 * (volume_median[10]/volume_head_median) * (volume[10]/volume_median[10]) + 0.00001)
            skel.bodynodes[11].set_mass(BODY_MASS * 0.4830 * (volume_median[11]/volume_torso_median) * (volume[11]/volume_median[11]) + 0.00001)
            skel.bodynodes[12].set_mass(BODY_MASS * 0.4830 * (volume_median[12]/volume_torso_median) * (volume[12]/volume_median[12]) + 0.00001)
            skel.bodynodes[13].set_mass(BODY_MASS * 0.0710 * (volume_median[13]/volume_head_median) * (volume[13]/volume_median[13]) + 0.00001)
            skel.bodynodes[14].set_mass(BODY_MASS * 0.0330 * (volume[14]/volume_median[14]) + 0.00001)
            skel.bodynodes[15].set_mass(BODY_MASS * 0.0330 * (volume[15]/volume_median[15]) + 0.00001)
            skel.bodynodes[16].set_mass(BODY_MASS * 0.0190 * (volume[16]/volume_median[16]) + 0.00001)
            skel.bodynodes[17].set_mass(BODY_MASS * 0.0190 * (volume[17]/volume_median[17]) + 0.00001)
            skel.bodynodes[18].set_mass(BODY_MASS * 0.0060 * (volume[18]/volume_median[18]) + 0.00001)
            skel.bodynodes[19].set_mass(BODY_MASS * 0.0060 * (volume[19]/volume_median[19]) + 0.00001)

        body_mass = 0.0
        #set the mass moment of inertia matrices
        for body_ct in range(NUM_CAPSULES):
            radius = float(capsules[body_ct].rad[0])
            if radius < 0.0001: radius = 0.0001
            length = float(capsules[body_ct].length[0])
            if length < 0.0001: length = 0.0001
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


        print skel.bodynodes[0].C, 'SKEL BODYNODES 0 POS'
        print skel.bodynodes[0].C[0]*2.58872 + 1.185, skel.bodynodes[0].C[1]*2.58872 + 2.55


        skel = LibDartSkel().assign_init_joint_angles(skel, m, root_joint_type)

        print skel.bodynodes[0].C, 'SKEL BODYNODES 0 POS'
        print skel.bodynodes[0].C[0]*2.58872 + 1.185, skel.bodynodes[0].C[1]*2.58872 + 2.55

        skel = LibDartSkel().assign_joint_rest_and_stiffness(skel, m, STIFFNESS = stiffness, posture = posture, body_mass = body_mass)

        #skel = LibDartSkel().assign_joint_limits_and_damping(skel)

        #skel = LibDartSkel().assign_capsule_friction(skel, friction = 1000.0)


        #print "Body mass is: ", body_mass, "kg"
        self.body_mass = body_mass

        self.body_node = 9 #need to solve for the body node that corresponds to a force using flex.
        self.force = np.asarray([0.0, 100.0, 100.0])
        self.offset_from_centroid = np.asarray([-0.15, 0.0, 0.0])


        self.pmat_red_all = np.load(filepath_prefix+'/git/volumetric_pose_gen/data/pmat_red.npy', allow_pickle=True)
        self.force_dir_red_dart_all = np.load(filepath_prefix+'/git/volumetric_pose_gen/data/force_dir_red.npy', allow_pickle=True)
        for element in range(len(self.force_dir_red_dart_all)):
            self.force_dir_red_dart_all[element] = (np.multiply(np.asarray(self.force_dir_red_dart_all[element]),np.expand_dims(np.asarray(self.pmat_red_all[element]), axis = 1)))
        self.force_loc_red_dart_all = np.load(filepath_prefix+'/git/volumetric_pose_gen/data/force_loc_red.npy', allow_pickle=True).tolist()
        self.nearest_capsule_list_all = np.load(filepath_prefix+'/git/volumetric_pose_gen/data/nearest_capsule.npy', allow_pickle=True).tolist()


        #print('init pose = %s' % skel.q)
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



    def getCapsuleVolumes(self, mm_resolution, dataset_num = 1):
        print "calculating volumes", dataset_num

        res_multiplier = 1000./mm_resolution

        self.run_sim_step()
        contacts = self.world.collision_result.contact_sets

        contacts_with_all_nodes = []
        voxel_space_list = []
        capsule_centers_list = []

        for body_ct in range(NUM_CAPSULES):
            #give the capsules a weight propertional to their volume
            cap_rad = float(self.capsules[body_ct].rad[0])
            if cap_rad < 0.0001: cap_rad = 0.0001
            cap_len = float(self.capsules[body_ct].length[0])
            if cap_len < 0.0001: cap_len = 0.0001


            contacts_with_node = []
            for contact in contacts: #check all possible contacts
                if body_ct in contact: #check if the contact set includes the node we're looping through
                    for bn_contact in contact: #get the body node that the node we're looping through is in contact with
                        if body_ct != bn_contact:
                            contacts_with_node.append(bn_contact)

            #print contacts_with_node
            contacts_with_all_nodes.append(contacts_with_node)

            len_range = int((self.capsule_locs_abs[body_ct][0] - cap_len/2 - cap_rad)*res_multiplier), int((self.capsule_locs_abs[body_ct][0] + cap_len/2 + cap_rad)*res_multiplier)
            rad_range = int((self.capsule_locs_abs[body_ct][1] - cap_rad)*res_multiplier), int((self.capsule_locs_abs[body_ct][1] + cap_rad)*res_multiplier)

            if body_ct in [0, 3, 6, 9, 11, 12, 14, 15, 16, 17, 18, 19]:
                voxel_space = np.zeros((len_range[1]-len_range[0] + 1, rad_range[1] - rad_range[0], rad_range[1] - rad_range[0]))
                voxel_space_shape = np.shape(voxel_space)

                r = voxel_space_shape[1]/2.
                l = voxel_space_shape[0] - 2*r + 1

                #get sphere
                sphere = self.get_sphere_mask(radius = r)
                voxel_space[0:int(r), :, :] = sphere[0:int(r), :, :]
                voxel_space[int(r)+int(l)-1:, :, :] = sphere[int(r):, :, :]

                #get x-dir cylinder
                cylinder = self.get_cylinder_mask(radius = r, length = l, axis = "x")

                try:
                    voxel_space[int(r):int(r)+int(l), :, :] = cylinder
                except:
                    voxel_space[int(r):int(r)+int(l), :, :] = cylinder[1:, :, :]


                if body_ct in [0, 3, 6, 9]:
                    capsule_centers_list.append((np.array(self.capsule_locs_abs[body_ct])*res_multiplier).astype(int))
                elif body_ct in [11, 14, 16, 18]:
                    capsule_centers_list.append(((np.array(self.capsule_locs_abs[body_ct])+np.array([cap_len/2, 0, 0]))*res_multiplier).astype(int))
                elif body_ct in [12, 15, 17, 19]:
                    capsule_centers_list.append(((np.array(self.capsule_locs_abs[body_ct])-np.array([cap_len/2, 0, 0]))*res_multiplier).astype(int))


            elif body_ct in [1, 2, 4, 5, 10, 13]:
                voxel_space = np.zeros((rad_range[1] - rad_range[0], len_range[1]-len_range[0] + 1, rad_range[1] - rad_range[0]))
                voxel_space_shape = np.shape(voxel_space)

                r = voxel_space_shape[0]/2.
                l = voxel_space_shape[1] - 2*r + 1

                #get sphere
                sphere = self.get_sphere_mask(radius = r)
                voxel_space[:, 0:int(r), :] = sphere[:, 0:int(r), :]
                voxel_space[:, int(r)+int(l)-1:, :] = sphere[:, int(r):, :]

                #get x-dir cylinder
                cylinder = self.get_cylinder_mask(radius = r, length = l, axis = "y")
                try:
                    voxel_space[:, int(r):int(r)+int(l), :] = cylinder
                except:
                    voxel_space[:, int(r):int(r)+int(l), :] = cylinder[:, 1:, :]

                capsule_centers_list.append(((np.array(self.capsule_locs_abs[body_ct])-np.array([0, cap_len/2, 0]))*res_multiplier).astype(int))

            elif body_ct in [7, 8]:
                voxel_space = np.zeros((rad_range[1] - rad_range[0], rad_range[1] - rad_range[0], len_range[1]-len_range[0] + 1))
                voxel_space_shape = np.shape(voxel_space)

                r = voxel_space_shape[0]/2.
                l = voxel_space_shape[2] - 2*r + 1

                #get sphere
                sphere = self.get_sphere_mask(radius = r)
                voxel_space[:, :, 0:int(r)] = sphere[:, :, 0:int(r)]
                voxel_space[:, :, int(r)+int(l)-1:] = sphere[:, :, int(r):]

                #get x-dir cylinder
                cylinder = self.get_cylinder_mask(radius = r, length = l, axis = "z")
                try:
                    voxel_space[:, :, int(r):int(r)+int(l)] = cylinder
                except:
                    voxel_space[:, :, int(r):int(r)+int(l)] = cylinder[:, :, 1:]

                capsule_centers_list.append(((np.array(self.capsule_locs_abs[body_ct])+np.array([0, 0, cap_len/2]))*res_multiplier).astype(int))




            voxel_space_list.append(voxel_space)


            #print body_ct, cap_len, cap_rad,  self.capsule_locs_abs[body_ct]
            #print voxel_space_shape, cap_rad, cap_len+2*cap_rad, np.sum(voxel_space)/(res_multiplier*res_multiplier*res_multiplier), self.volume[body_ct]

            #break
        print "got initial volume"

        volume_analytic = []
        volume_discret = []
        volume_discret_mod = []
        for body_ct in range(NUM_CAPSULES):
            orig_shape_center = capsule_centers_list[body_ct]
            orig_shape_dim = np.shape(voxel_space_list[body_ct])
            #print capsule_centers_list[body_ct], np.shape(voxel_space_list[body_ct])
            voxel_space_mod = np.copy(voxel_space_list[body_ct])
            #print body_ct, "vox sum", np.sum(voxel_space_mod)

            duplicate_check = []
            print contacts_with_all_nodes[body_ct]
            for other_bn in contacts_with_all_nodes[body_ct]:
                if other_bn not in duplicate_check:
                    duplicate_check.append(other_bn)
                    new_shape_center = capsule_centers_list[other_bn]
                    new_shape_dim = np.shape(voxel_space_list[other_bn])

                    x_d_resp_orig = - orig_shape_center[0] + orig_shape_dim[0]/2 + new_shape_center[0] - new_shape_dim[0]/2
                    if x_d_resp_orig < 0: x_d_resp_orig = 0
                    if x_d_resp_orig > orig_shape_dim[0]: x_d_resp_orig = orig_shape_dim[0]

                    x_u_resp_orig = - orig_shape_center[0] + orig_shape_dim[0]/2 + new_shape_center[0] + new_shape_dim[0]/2
                    if x_u_resp_orig < 0: x_u_resp_orig = 0
                    if x_u_resp_orig > orig_shape_dim[0]: x_u_resp_orig = orig_shape_dim[0]

                    x_d_resp_new = orig_shape_center[0] + new_shape_dim[0]/2 - new_shape_center[0] - orig_shape_dim[0]/2
                    if x_d_resp_new < 0: x_d_resp_new = 0
                    if x_d_resp_new > new_shape_dim[0]: x_d_resp_new = new_shape_dim[0]

                    x_u_resp_new = orig_shape_center[0] + new_shape_dim[0]/2 - new_shape_center[0] + orig_shape_dim[0]/2
                    if x_u_resp_new < 0: x_u_resp_new = 0
                    if x_u_resp_new > new_shape_dim[0]: x_u_resp_new = new_shape_dim[0]

                    x_discrepancy = (x_u_resp_new - x_d_resp_new) - (x_u_resp_orig - x_d_resp_orig)
                    if x_discrepancy != 0:
                        #print x_discrepancy
                        if x_discrepancy > 0 and (x_d_resp_orig >= x_discrepancy):
                            x_d_resp_orig -= x_discrepancy
                        elif x_discrepancy > 0 and (x_u_resp_orig <= (orig_shape_dim[0] - x_discrepancy)):
                            x_u_resp_orig += x_discrepancy
                        elif x_discrepancy < 0 and (x_d_resp_new >= -x_discrepancy):
                            x_d_resp_new += x_discrepancy
                        elif x_discrepancy < 0 and (x_u_resp_new <= (new_shape_dim[0] + x_discrepancy)):
                            x_u_resp_new -= x_discrepancy


                    y_d_resp_orig = - orig_shape_center[1] + orig_shape_dim[1]/2 + new_shape_center[1] - new_shape_dim[1]/2
                    if y_d_resp_orig < 0: y_d_resp_orig = 0
                    if y_d_resp_orig > orig_shape_dim[1]: y_d_resp_orig = orig_shape_dim[1]

                    y_u_resp_orig = - orig_shape_center[1] + orig_shape_dim[1]/2 + new_shape_center[1] + new_shape_dim[1]/2
                    if y_u_resp_orig < 0: y_u_resp_orig = 0
                    if y_u_resp_orig > orig_shape_dim[1]: y_u_resp_orig = orig_shape_dim[1]

                    y_d_resp_new = orig_shape_center[1] + new_shape_dim[1]/2 - new_shape_center[1] - orig_shape_dim[1]/2
                    if y_d_resp_new < 0: y_d_resp_new = 0
                    if y_d_resp_new > new_shape_dim[1]: y_d_resp_new = new_shape_dim[1]

                    y_u_resp_new = orig_shape_center[1] + new_shape_dim[1]/2 - new_shape_center[1] + orig_shape_dim[1]/2
                    if y_u_resp_new < 0: y_u_resp_new = 0
                    if y_u_resp_new > new_shape_dim[1]: y_u_resp_new = new_shape_dim[1]


                    y_discrepancy = (y_u_resp_new - y_d_resp_new) - (y_u_resp_orig - y_d_resp_orig)
                    if y_discrepancy != 0:
                        #print y_discrepancy
                        if y_discrepancy > 0 and (y_d_resp_orig >= y_discrepancy):
                            y_d_resp_orig -= y_discrepancy
                        elif y_discrepancy > 0 and (y_u_resp_orig <= (orig_shape_dim[1] - y_discrepancy)):
                            y_u_resp_orig += y_discrepancy
                        elif y_discrepancy < 0 and (y_d_resp_new >= -y_discrepancy):
                            y_d_resp_new += y_discrepancy
                        elif y_discrepancy < 0 and (y_u_resp_new <= (new_shape_dim[1] + y_discrepancy)):
                            y_u_resp_new -= y_discrepancy

                    z_d_resp_orig = - orig_shape_center[2] + orig_shape_dim[2]/2 + new_shape_center[2] - new_shape_dim[2]/2
                    if z_d_resp_orig < 0: z_d_resp_orig = 0
                    if z_d_resp_orig > orig_shape_dim[2]: z_d_resp_orig = orig_shape_dim[2]

                    z_u_resp_orig = - orig_shape_center[2] + orig_shape_dim[2]/2 + new_shape_center[2] + new_shape_dim[2]/2
                    if z_u_resp_orig < 0: z_u_resp_orig = 0
                    if z_u_resp_orig > orig_shape_dim[2]: z_u_resp_orig = orig_shape_dim[2]

                    z_d_resp_new = orig_shape_center[2] + new_shape_dim[2]/2 - new_shape_center[2] - orig_shape_dim[2]/2
                    if z_d_resp_new < 0: z_d_resp_new = 0
                    if z_d_resp_new > new_shape_dim[2]: z_d_resp_new = new_shape_dim[2]

                    z_u_resp_new = orig_shape_center[2] + new_shape_dim[2]/2 - new_shape_center[2] + orig_shape_dim[2]/2
                    if z_u_resp_new < 0: z_u_resp_new = 0
                    if z_u_resp_new > new_shape_dim[2]: z_u_resp_new = new_shape_dim[2]

                    z_discrepancy = (z_u_resp_new - z_d_resp_new) - (z_u_resp_orig - z_d_resp_orig)
                    if z_discrepancy != 0:
                        #print z_discrepancy
                        if z_discrepancy > 0 and (z_d_resp_orig >= z_discrepancy):
                            z_d_resp_orig -= z_discrepancy
                        elif z_discrepancy > 0 and (z_u_resp_orig <= (orig_shape_dim[2] - z_discrepancy)):
                            z_u_resp_orig += z_discrepancy
                        elif z_discrepancy < 0 and (z_d_resp_new >= -z_discrepancy):
                            z_d_resp_new += z_discrepancy
                        elif z_discrepancy < 0 and (z_u_resp_new <= (new_shape_dim[2] + z_discrepancy)):
                            z_u_resp_new -= z_discrepancy

                    #print other_bn, x_d_resp_orig, x_u_resp_orig, "out of", orig_shape_dim[0], "  ", x_d_resp_new, x_u_resp_new, "out of", new_shape_dim[0]
                    #print ' ', y_d_resp_orig, y_u_resp_orig, "out of", orig_shape_dim[1], "  ", y_d_resp_new, y_u_resp_new, "out of", new_shape_dim[1]
                    #print ' ', z_d_resp_orig, z_u_resp_orig, "out of", orig_shape_dim[2], "  ", z_d_resp_new, z_u_resp_new, "out of", new_shape_dim[2]

                    #print capsule_centers_list[other_bn], np.shape(voxel_space_list[other_bn])

                    voxel_space_mod[x_d_resp_orig:x_u_resp_orig, y_d_resp_orig:y_u_resp_orig, z_d_resp_orig:z_u_resp_orig] += \
                        voxel_space_list[other_bn][x_d_resp_new:x_u_resp_new, y_d_resp_new:y_u_resp_new, z_d_resp_new:z_u_resp_new]

                    #print "new vox sum", np.sum(voxel_space_mod)

            voxel_space_mod = np.array(voxel_space_mod.astype(float).flatten())
            voxel_space_mod[voxel_space_mod == 0] += 1.
            voxel_space = np.array(np.copy(voxel_space_list[body_ct]).astype(float).flatten())
            voxel_space_mod = np.divide(voxel_space, voxel_space_mod)



            volume_analytic.append(self.volume[body_ct])

            volume_discret.append(np.sum(voxel_space) / (res_multiplier * res_multiplier * res_multiplier))

            volume_discret_mod.append(np.sum(voxel_space_mod)/(res_multiplier*res_multiplier*res_multiplier))



            print body_ct,  volume_analytic[-1], volume_discret[-1], volume_discret_mod[-1]

            #break



        #print contacts

        return volume_analytic, volume_discret, volume_discret_mod

    def get_sphere_mask(self, radius):
        radius -= 0.5

        xx, yy, zz = np.mgrid[0:radius*2+1, 0:radius*2+1, 0:radius*2+1]
        sphere = (xx - radius) ** 2 + (yy - radius) ** 2 + (zz - radius) ** 2
        sphere[sphere <= radius**2] = 1
        sphere[sphere > radius**2] = 0
        #print sphere.shape
        return sphere

    def get_cylinder_mask(self, radius, length, axis):
        radius -= 0.5

        if axis == "x":
            xx, yy, zz = np.mgrid[0:length, 0:radius*2+1, 0:radius*2+1]
            cylinder = (yy - radius) ** 2 + (zz - radius) ** 2
        elif axis == "y":
            xx, yy, zz = np.mgrid[0:radius*2+1, 0:length, 0:radius*2+1]
            cylinder = (xx - radius) ** 2 + (zz - radius) ** 2
        elif axis == "z":
            xx, yy, zz = np.mgrid[0:radius*2+1, 0:radius*2+1, 0:length]
            cylinder = (xx - radius) ** 2 + (yy - radius) ** 2

        cylinder[cylinder <= radius**2] = 1
        cylinder[cylinder > radius**2] = 0
        return cylinder


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



        #root_joint_pos = [skel.bodynodes[0].C[0] - self.cap_offsets[0][0]*np.cos(skel.q[2]) + self.cap_offsets[0][1]*np.sin(skel.q[2]),
        #                  skel.bodynodes[0].C[1] - self.cap_offsets[0][0]*np.sin(skel.q[2]) - self.cap_offsets[0][1]*np.cos(skel.q[2])]


        #print skel.bodynodes[0].C
        #print skel.q[0:3], "FIRST 3 ANGLES"
        #print root_joint_pos, 'old root pos'


        Trans1 = lib_kinematics.matrix_from_dir_cos_angles(skel.q[0:3])
        dist = np.matmul(Trans1, np.array([0.0, -0.04, 0.0]))


        #print skel.bodynodes[0].C - dist, 'new root pos'
        root_joint_pos = skel.bodynodes[0].C - dist
        root_joint_pos[2] += self.STARTING_HEIGHT / DART_TO_FLEX_CONV
        print root_joint_pos, 'radius: ', self.root_capsule_rad

        #here lets transform the position of skel.bodybodes[0].C by a length of length/2
        midglute_to_left = np.matmul(Trans1, np.array([np.abs(float(self.capsules[0].length[0]))/2, 0.0, 0.0]))
        dist_down = -np.abs(midglute_to_left[2])
        self.glute_height = 2.58872 * skel.bodynodes[0].C[2] + self.STARTING_HEIGHT - 2.58872 * np.abs(float(self.capsules[0].rad[0]))
        self.glute_height += 2.58872 * dist_down #add a bit down for when the person is tilted to the side



        #print "appending time", time() - time_orig

        #LibDartSkel().impose_force(skel=skel, body_node=self.body_node, force=self.force, offset_from_centroid=self.offset_from_centroid, cap_offsets=self.cap_offsets, render=False, init=False)

        self.force_dir_list_prev = force_dir_list
        self.pmat_idx_list_prev = pmat_idx_list
        self.force_loc_list_prev = force_loc_list

        if self.step_num == 0:
            self.world.check_collision()
            contact_check_bns = [4, 5, 7, 8, 16, 17, 18, 19]
            #for contact_set in self.world.collision_result.contact_sets:
            #    if contact_set[0] in contact_check_bns or contact_set[1] in contact_check_bns:  # consider removing spine 3 and upper legs
            print self.world.collision_result.contact_sets
                    #sleep(1)

        self.step_num += 1



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
