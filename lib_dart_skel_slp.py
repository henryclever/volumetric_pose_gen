# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group
import numpy as np
import math
import pydart2 as pydart
from pydart2 import skeleton_builder
from dart_opengl_window import GLUTWindow

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
from pydart2.gui.opengl.scene import OpenGLScene
from time import time

HANDS_BEHIND_HEAD = False

class LibDartSkel():

    def eulerAnglesToRotationMatrix(self, theta):
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R


    def assign_init_joint_angles(self, skel, m, root_joint_type = "FREE"):
        ####################################### ASSIGN INITIAL JOINT ANGLES ############################################
        #skel_q_init = np.random.rand(skel.ndofs) - 0.5
        skel_q_init = skel.ndofs * [0]
        print "NUMBER OF SKEL DOFS", skel.ndofs

        if root_joint_type == "FREE":
            index_offset = 3
            skel_q_init[0:3] = np.asarray(m.pose[0:3])
            #root node index:
            #0 to 3 is orientation
            #3 to 6 is position

        else:
            index_offset = 0



        skel_q_init[index_offset+3:6+index_offset] = np.asarray(m.pose[3:6]) #left glute
        skel_q_init[index_offset+6:9+index_offset] = np.asarray(m.pose[6:9]) #right glute
        skel_q_init[index_offset+9:12+index_offset] = np.asarray(m.pose[9:12]) #low spine

        skel_q_init[index_offset+12:15+index_offset] = np.asarray(m.pose[12:15]) #left knee. should be 0.0 to 3.14
        skel_q_init[index_offset+15:18+index_offset] = np.asarray(m.pose[15:18]) #right knee. should be 0.0 to 3.14

        skel_q_init[index_offset+18:21+index_offset] = np.asarray(m.pose[18:21]) #mid spine

        skel_q_init[index_offset+21:24+index_offset] = np.asarray(m.pose[21:24]) #left foot
        skel_q_init[index_offset+24:27+index_offset] = np.asarray(m.pose[24:27]) #right foot

        skel_q_init[index_offset+27:30+index_offset] = np.asarray(m.pose[27:30]) #upper spine

        skel_q_init[index_offset+30:33+index_offset] = np.asarray(m.pose[36:39]) #neck

        skel_q_init[index_offset+33:36+index_offset] = np.asarray(m.pose[39:42]) #left inner shoulder
        skel_q_init[index_offset+36:39+index_offset] = np.asarray(m.pose[42:45]) #right inner shoulder

        skel_q_init[index_offset+39:42+index_offset] = np.asarray(m.pose[45:48]) #head

        skel_q_init[index_offset+42:45+index_offset] = np.asarray(m.pose[48:51]) #left outer shoulder
        skel_q_init[index_offset+45:48+index_offset] = np.asarray(m.pose[51:54]) #right outer shoulder

        skel_q_init[index_offset+48:51+index_offset] = np.asarray(m.pose[54:57]) #left elbow. should be -3.14 to 0.0
        skel_q_init[index_offset+51:54+index_offset] = np.asarray(m.pose[57:60]) #right elbow. should be  0.0 to 3.14

        skel_q_init[index_offset+54:57+index_offset] = np.asarray(m.pose[60:63]) #left hand
        skel_q_init[index_offset+57:60+index_offset] = np.asarray(m.pose[63:66]) #right hand


        #this is where you set the angles according to m, the angle axis representation.
        skel.set_positions(skel_q_init)

        #print skel.root_bodynode()
        #print skel.name
        #from pydart2 import bodynode
        #bn = bodynode.BodyNode(skel, 8)
        return skel

    def assign_joint_rest_and_stiffness(self, skel, m, STIFFNESS, posture, body_mass):

        ################################# ASSIGN JOINT REST POSITION AND SPRING COEFF ##################################


        #if STIFFNESS == "LOW":
        #    arm_stiffness = 1.0
        #    head_stiffness = 10.0
        #    leg_stiffness = 20.0
        #    knee_stiffness = 10.0
        #    torso_stiffness = 200.0

        #elif STIFFNESS == "MED":
        #    arm_stiffness = 10.0
        #    head_stiffness = 10.0
        #    leg_stiffness = 50.0
        #    knee_stiffness = 50.0
        #    torso_stiffness = 500.0

        #elif STIFFNESS == "HIGH":
        #    arm_stiffness = 50.0
        #    head_stiffness = 50.0
        #    leg_stiffness = 200.0
        #    knee_stiffness = 200.0
        #    torso_stiffness = 1000.0

        bm_fraction = body_mass/70.45


        if posture == "sit":
            torso_stiffness = 200.0*bm_fraction
        else:
            torso_stiffness = 200.0*bm_fraction

        if STIFFNESS == "upperbody":
            r_arm_stiffness = 100.0*bm_fraction
            l_arm_stiffness = 100.0*bm_fraction
            head_stiffness = 200.0*bm_fraction
            r_leg_stiffness = 10.0*bm_fraction
            l_leg_stiffness = 10.0*bm_fraction
            r_knee_stiffness = 10.0*bm_fraction
            l_knee_stiffness = 10.0*bm_fraction
        elif STIFFNESS == "lowerbody":
            r_arm_stiffness = 2.0*bm_fraction
            l_arm_stiffness = 2.0*bm_fraction
            head_stiffness = 200.0*bm_fraction
            r_leg_stiffness = 200.0*bm_fraction
            l_leg_stiffness = 200.0*bm_fraction
            r_knee_stiffness = 200.0*bm_fraction
            l_knee_stiffness = 200.0*bm_fraction
        elif STIFFNESS == "rightside":
            r_arm_stiffness = 100.0*bm_fraction
            l_arm_stiffness = 2.0*bm_fraction
            head_stiffness = 200.0*bm_fraction
            r_leg_stiffness = 200.0*bm_fraction
            l_leg_stiffness = 10.0*bm_fraction
            r_knee_stiffness = 200.0*bm_fraction
            l_knee_stiffness = 10.0*bm_fraction
        elif STIFFNESS == "leftside":
            r_arm_stiffness = 2.0*bm_fraction
            l_arm_stiffness = 100.0*bm_fraction
            head_stiffness = 200.0*bm_fraction
            r_leg_stiffness = 10.0*bm_fraction
            l_leg_stiffness = 200.0*bm_fraction
            r_knee_stiffness = 10.0*bm_fraction
            l_knee_stiffness = 200.0*bm_fraction
        else: #not stiff
            r_arm_stiffness = 2.0*bm_fraction*1.
            l_arm_stiffness = 2.0*bm_fraction*1.
            r_elbow_stiffness = 2.0*bm_fraction*1.
            l_elbow_stiffness = 2.0*bm_fraction*1.
            head_stiffness = 200.0*bm_fraction
            r_leg_stiffness = 6.0*bm_fraction*1.
            l_leg_stiffness = 6.0*bm_fraction*1.
            r_knee_stiffness = 3.0*bm_fraction*1.
            l_knee_stiffness = 3.0*bm_fraction*1.

        #print(m.pose, 'pose to set rest')

        for joint in skel.joints:
            #print joint.spring_stiffness(0)
            if joint.name == "leftThigh":
                joint.set_rest_position(0, float(m.pose[3]))
                joint.set_rest_position(1, float(m.pose[4]))
                joint.set_rest_position(2, float(m.pose[5]))
                joint.set_spring_stiffness(0, l_leg_stiffness)
                joint.set_spring_stiffness(1, l_leg_stiffness)
                joint.set_spring_stiffness(2, l_leg_stiffness)
            elif joint.name == "rightThigh":
                joint.set_rest_position(0, float(m.pose[6]))
                joint.set_rest_position(1, float(m.pose[7]))
                joint.set_rest_position(2, float(m.pose[8]))
                joint.set_spring_stiffness(0, r_leg_stiffness)
                joint.set_spring_stiffness(1, r_leg_stiffness)
                joint.set_spring_stiffness(2, r_leg_stiffness)
            elif joint.name == "spine":
                joint.set_rest_position(0, float(m.pose[9]))
                joint.set_rest_position(1, float(m.pose[10]))
                joint.set_rest_position(2, float(m.pose[11]))
                joint.set_spring_stiffness(0, torso_stiffness)
                joint.set_spring_stiffness(1, torso_stiffness)
                joint.set_spring_stiffness(2, torso_stiffness)
            elif joint.name == "leftCalf":
                joint.set_rest_position(0, float(m.pose[12]))
                joint.set_rest_position(1, float(m.pose[13]))
                joint.set_rest_position(2, float(m.pose[14]))
                joint.set_spring_stiffness(0, l_knee_stiffness)
                joint.set_spring_stiffness(1, head_stiffness)
                joint.set_spring_stiffness(2, head_stiffness)
            elif joint.name == "rightCalf":
                joint.set_rest_position(0, float(m.pose[15]))
                joint.set_rest_position(1, float(m.pose[16]))
                joint.set_rest_position(2, float(m.pose[17]))
                joint.set_spring_stiffness(0, r_knee_stiffness)
                joint.set_spring_stiffness(1, head_stiffness)
                joint.set_spring_stiffness(2, head_stiffness)
            elif joint.name == "spine1":
                joint.set_rest_position(0, float(m.pose[18]))
                joint.set_rest_position(1, float(m.pose[19]))
                joint.set_rest_position(2, float(m.pose[20]))
                joint.set_spring_stiffness(0, torso_stiffness)
                joint.set_spring_stiffness(1, torso_stiffness)
                joint.set_spring_stiffness(2, torso_stiffness)
            elif joint.name == "leftFoot":
                joint.set_rest_position(0, float(m.pose[21]))
                joint.set_rest_position(1, float(m.pose[22]))
                joint.set_rest_position(2, float(m.pose[23]))
                joint.set_spring_stiffness(0, l_leg_stiffness)
                joint.set_spring_stiffness(1, l_leg_stiffness)
                joint.set_spring_stiffness(2, l_leg_stiffness)
            elif joint.name == "rightFoot":
                joint.set_rest_position(0, float(m.pose[24]))
                joint.set_rest_position(1, float(m.pose[25]))
                joint.set_rest_position(2, float(m.pose[26]))
                joint.set_spring_stiffness(0, r_leg_stiffness)
                joint.set_spring_stiffness(1, r_leg_stiffness)
                joint.set_spring_stiffness(2, r_leg_stiffness)
            elif joint.name == "spine2":
                joint.set_rest_position(0, float(m.pose[27]))
                joint.set_rest_position(1, float(m.pose[28]))
                joint.set_rest_position(2, float(m.pose[29]))
                joint.set_spring_stiffness(0, torso_stiffness)
                joint.set_spring_stiffness(1, torso_stiffness)
                joint.set_spring_stiffness(2, torso_stiffness)
            elif joint.name == "neck":
                joint.set_rest_position(0, float(m.pose[36]))
                joint.set_rest_position(1, float(m.pose[37]))
                joint.set_rest_position(2, float(m.pose[38]))
                joint.set_spring_stiffness(0, head_stiffness)
                joint.set_spring_stiffness(1, head_stiffness)
                joint.set_spring_stiffness(2, head_stiffness)
            elif joint.name == "leftShoulder":
                joint.set_rest_position(0, float(m.pose[39]))
                joint.set_rest_position(1, float(m.pose[40]))
                joint.set_rest_position(2, float(m.pose[41]))
                joint.set_spring_stiffness(0, l_arm_stiffness)
                joint.set_spring_stiffness(1, l_arm_stiffness)
                joint.set_spring_stiffness(2, l_arm_stiffness)
            elif joint.name == "rightShoulder":
                joint.set_rest_position(0, float(m.pose[42]))
                joint.set_rest_position(1, float(m.pose[43]))
                joint.set_rest_position(2, float(m.pose[44]))
                joint.set_spring_stiffness(0, r_arm_stiffness)
                joint.set_spring_stiffness(1, r_arm_stiffness)
                joint.set_spring_stiffness(2, r_arm_stiffness)
            elif joint.name == "head":
                joint.set_rest_position(0, float(m.pose[45]))
                joint.set_rest_position(1, float(m.pose[46]))
                joint.set_rest_position(2, float(m.pose[47]))
                joint.set_spring_stiffness(0, head_stiffness)
                joint.set_spring_stiffness(1, head_stiffness)
                joint.set_spring_stiffness(2, head_stiffness)
            elif joint.name == "leftUpperArm":
                joint.set_rest_position(0, float(m.pose[48]))
                joint.set_rest_position(1, float(m.pose[49]))
                joint.set_rest_position(2, float(m.pose[50]))
                joint.set_spring_stiffness(0, l_arm_stiffness)
                joint.set_spring_stiffness(1, l_arm_stiffness)
                joint.set_spring_stiffness(2, l_arm_stiffness)
            elif joint.name == "rightUpperArm":
                joint.set_rest_position(0, float(m.pose[51]))
                joint.set_rest_position(1, float(m.pose[52]))
                joint.set_rest_position(2, float(m.pose[53]))
                joint.set_spring_stiffness(0, r_arm_stiffness)
                joint.set_spring_stiffness(1, r_arm_stiffness)
                joint.set_spring_stiffness(2, r_arm_stiffness)
            elif joint.name == "leftForeArm":
                joint.set_rest_position(0, float(m.pose[54]))
                joint.set_rest_position(1, float(m.pose[55]))
                joint.set_rest_position(2, float(m.pose[56]))
                joint.set_spring_stiffness(0, head_stiffness)
                joint.set_spring_stiffness(1, l_elbow_stiffness)
                joint.set_spring_stiffness(2, head_stiffness)
            elif joint.name == "rightForeArm":
                joint.set_rest_position(0, float(m.pose[57]))
                joint.set_rest_position(1, float(m.pose[58]))
                joint.set_rest_position(2, float(m.pose[59]))
                joint.set_spring_stiffness(0, head_stiffness)
                joint.set_spring_stiffness(1, r_elbow_stiffness)
                joint.set_spring_stiffness(2, head_stiffness)
            elif joint.name == "leftHand":
                joint.set_rest_position(0, float(m.pose[60]))
                joint.set_rest_position(1, float(m.pose[61]))
                joint.set_rest_position(2, float(m.pose[62]))
                joint.set_spring_stiffness(0, l_arm_stiffness)
                joint.set_spring_stiffness(1, l_arm_stiffness)
                joint.set_spring_stiffness(2, l_arm_stiffness)
            elif joint.name == "rightHand":
                joint.set_rest_position(0, float(m.pose[63]))
                joint.set_rest_position(1, float(m.pose[64]))
                joint.set_rest_position(2, float(m.pose[65]))
                joint.set_spring_stiffness(0, r_arm_stiffness)
                joint.set_spring_stiffness(1, r_arm_stiffness)
                joint.set_spring_stiffness(2, r_arm_stiffness)


        r_arm_damping = r_arm_stiffness*5
        l_arm_damping = l_arm_stiffness*5
        r_elbow_damping = r_elbow_stiffness*5
        l_elbow_damping = l_elbow_stiffness*5
        head_damping = head_stiffness*5
        r_leg_damping = r_leg_stiffness*5
        l_leg_damping = l_leg_stiffness*5
        r_knee_damping = r_knee_stiffness*5
        l_knee_damping = l_knee_stiffness*5
        torso_damping = torso_stiffness*5
        #print("DAMPINGGGG")


        for joint in skel.joints:

            if joint.name == "leftThigh":
                joint.set_position_lower_limit(0, -140.*np.pi/180)#-2.7443260550003967)  # ext
                joint.set_position_upper_limit(0, 10.*np.pi/180.)#-0.14634814003149707)
                joint.set_position_lower_limit(1, -60.*np.pi/180.)#-1.0403111466710133)  # yaw
                joint.set_position_upper_limit(1, 120.*np.pi/180.)#1.1185343875601006)
                joint.set_position_lower_limit(2, -60.*np.pi/180.)#-0.421484532214729)  # abd
                joint.set_position_upper_limit(2, 60.*np.pi/180.)#0.810063927501682)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_leg_damping)
                joint.set_damping_coefficient(1, l_leg_damping)
                joint.set_damping_coefficient(2, l_leg_damping)
            elif joint.name == "rightThigh":
                joint.set_position_lower_limit(0, -140.*np.pi/180)#-2.7443260550003967)  # ext
                joint.set_position_upper_limit(0, 10.*np.pi/180.)#-0.14634814003149707)
                joint.set_position_lower_limit(1, -60.*np.pi/180.)#-1.1185343875601006)  # yaw
                joint.set_position_upper_limit(1, 120.*np.pi/180.)#1.0403111466710133)
                joint.set_position_lower_limit(2, -60.*np.pi/180.)#-0.810063927501682)  # abd
                joint.set_position_upper_limit(2, 60.*np.pi/180.)#0.421484532214729)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_leg_damping)
                joint.set_damping_coefficient(1, r_leg_damping)
                joint.set_damping_coefficient(2, r_leg_damping)
            elif joint.name == "spine" or joint.name == "spine1" or joint.name == "spine2":
                joint.set_position_lower_limit(0, -np.pi/6)
                joint.set_position_upper_limit(0, np.pi/6)
                joint.set_position_lower_limit(1, -np.pi/12)
                joint.set_position_upper_limit(1, np.pi/12)
                joint.set_position_lower_limit(2, -np.pi/12)
                joint.set_position_upper_limit(2, np.pi/12)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, torso_damping)
                joint.set_damping_coefficient(1, torso_damping)
                joint.set_damping_coefficient(2, torso_damping)
            elif joint.name == "leftCalf":
                joint.set_position_lower_limit(0, -1.*np.pi/180.)#0.0)
                joint.set_position_upper_limit(0, 170.*np.pi/180.)#2.7020409229712863)
                joint.set_position_lower_limit(1, -np.pi)#0.0)
                joint.set_position_upper_limit(1, np.pi)#2.7020409229712863)
                joint.set_position_lower_limit(2, -np.pi)#0.0)
                joint.set_position_upper_limit(2, np.pi)#2.7020409229712863)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_knee_damping)
                joint.set_damping_coefficient(1, head_damping)
                joint.set_damping_coefficient(2, head_damping)
            elif joint.name == "rightCalf":
                joint.set_position_lower_limit(0, -1.*np.pi/180.)#0.0)
                joint.set_position_upper_limit(0, 170.*np.pi/180.)#2.7020409229712863)
                joint.set_position_lower_limit(1, -np.pi)#0.0)
                joint.set_position_upper_limit(1, np.pi)#2.7020409229712863)
                joint.set_position_lower_limit(2, -np.pi)#0.0)
                joint.set_position_upper_limit(2, np.pi)#2.7020409229712863)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_knee_damping)
                joint.set_damping_coefficient(1, head_damping)
                joint.set_damping_coefficient(2, head_damping)
            elif joint.name == "leftFoot":
                joint.set_position_lower_limit(0, -np.pi/6)
                joint.set_position_upper_limit(0, np.pi/3)
                joint.set_position_lower_limit(1, -np.pi/6)
                joint.set_position_upper_limit(1, np.pi/6)
                joint.set_position_lower_limit(2, -np.pi/6)
                joint.set_position_upper_limit(2, np.pi/6)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_leg_damping)
                joint.set_damping_coefficient(1, l_leg_damping)
                joint.set_damping_coefficient(2, l_leg_damping)
            elif joint.name == "rightFoot":
                joint.set_position_lower_limit(0, -np.pi/6)
                joint.set_position_upper_limit(0, np.pi/3)
                joint.set_position_lower_limit(1, -np.pi/6)
                joint.set_position_upper_limit(1, np.pi/6)
                joint.set_position_lower_limit(2, -np.pi/6)
                joint.set_position_upper_limit(2, np.pi/6)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_leg_damping)
                joint.set_damping_coefficient(1, r_leg_damping)
                joint.set_damping_coefficient(2, r_leg_damping)
            elif joint.name == "leftShoulder":
                joint.set_position_lower_limit(0, -40.*np.pi/180.)  # -1.8674195346872975 * 1 / 3)  # roll
                joint.set_position_upper_limit(0, 40.*np.pi/180.)  #1.410545172086535 * 1 / 3)
                joint.set_position_lower_limit(1, -50.*np.pi/180.)  #-1.530112726921327 * 1 / 3)  # yaw
                joint.set_position_upper_limit(1, 20.*np.pi/180.)  #1.2074724617209949 * 1 / 3)
                joint.set_position_lower_limit(2, -40.*np.pi/180.)  #-1.9550515937478927 * 1 / 3)  # pitch
                joint.set_position_upper_limit(2, 40.*np.pi/180.)  #1.7587935205169856 * 1 / 3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_arm_damping)
                joint.set_damping_coefficient(1, l_arm_damping)
                joint.set_damping_coefficient(2, l_arm_damping)
            elif joint.name == "rightShoulder":
                joint.set_position_lower_limit(0, -40.*np.pi/180.)  #-1.8674195346872975 * 1 / 3)  # roll
                joint.set_position_upper_limit(0, 40.*np.pi/180.)  #1.410545172086535 * 1 / 3)
                joint.set_position_lower_limit(1, -20.*np.pi/180.)  #-1.2074724617209949 * 1 / 3)  # yaw
                joint.set_position_upper_limit(1, 50.*np.pi/180.)  #1.530112726921327  * 1 / 3)
                joint.set_position_lower_limit(2, -40.*np.pi/180.)  #-1.7587935205169856 * 1 / 3)  # pitch
                joint.set_position_upper_limit(2, 40.*np.pi/180.)  #1.9550515937478927 * 1 / 3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_arm_damping)
                joint.set_damping_coefficient(1, r_arm_damping)
                joint.set_damping_coefficient(2, r_arm_damping)
            elif joint.name == "leftUpperArm":
                joint.set_position_lower_limit(0, -70.*np.pi/180.)  #-1.8674195346872975 * 2 / 3)  # roll
                joint.set_position_upper_limit(0, 70.*np.pi/180.)  #1.410545172086535 * 2 / 3)
                joint.set_position_lower_limit(1, -90.*np.pi/180.)  #-1.530112726921327 * 2 / 3)  # yaw
                joint.set_position_upper_limit(1, 35.*np.pi/180.)  #1.2074724617209949 * 2 / 3)
                joint.set_position_lower_limit(2, -90.*np.pi/180.)  #-1.9550515937478927 * 2 / 3)  # pitch
                joint.set_position_upper_limit(2, 60.*np.pi/180.)  #1.7587935205169856 * 2 / 3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_arm_damping)
                joint.set_damping_coefficient(1, l_arm_damping)
                joint.set_damping_coefficient(2, l_arm_damping)
            elif joint.name == "rightUpperArm":
                joint.set_position_lower_limit(0, -70.*np.pi/180.)  #-1.8674195346872975 * 2 / 3)  # roll
                joint.set_position_upper_limit(0, 70.*np.pi/180.)  #1.410545172086535 * 2 / 3)
                joint.set_position_lower_limit(1, -35.*np.pi/180.)  #-1.2074724617209949 * 2 / 3)  # yaw
                joint.set_position_upper_limit(1, 90.*np.pi/180.)  #1.530112726921327 * 2 / 3)
                joint.set_position_lower_limit(2, -60.*np.pi/180.)  #-1.7587935205169856 * 2 / 3)  # pitch
                joint.set_position_upper_limit(2, 90.*np.pi/180.)  #1.9550515937478927 * 2 / 3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_arm_damping)
                joint.set_damping_coefficient(1, r_arm_damping)
                joint.set_damping_coefficient(2, r_arm_damping)
            elif joint.name == "leftForeArm":
                joint.set_position_lower_limit(0, -np.pi)#0.0)
                joint.set_position_upper_limit(0, np.pi)#0.0)
                joint.set_position_lower_limit(1, -170.*np.pi/180.)  #-2.463868908637374)
                joint.set_position_upper_limit(1, 10.*np.pi/180.)  #0.0)
                joint.set_position_lower_limit(2, -np.pi)#0.0)
                joint.set_position_upper_limit(2, np.pi)#0.0)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, head_damping)
                joint.set_damping_coefficient(1, l_elbow_damping)
                joint.set_damping_coefficient(2, head_damping)
            elif joint.name == "rightForeArm":
                joint.set_position_lower_limit(0, -np.pi)#0.0)
                joint.set_position_upper_limit(0, np.pi)#0.0)
                joint.set_position_lower_limit(1, -10.*np.pi/180.)  #0.0)
                joint.set_position_upper_limit(1, 170.*np.pi/180.)  #2.463868908637374)
                joint.set_position_lower_limit(2, -np.pi)#0.0)
                joint.set_position_upper_limit(2, np.pi)#0.0)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, head_damping)
                joint.set_damping_coefficient(1, r_elbow_damping)
                joint.set_damping_coefficient(2, head_damping)
            elif joint.name == "leftHand":
                joint.set_position_lower_limit(0, -np.pi/6)
                joint.set_position_upper_limit(0, np.pi/6)
                joint.set_position_lower_limit(1, -np.pi/6)
                joint.set_position_upper_limit(1, np.pi/6)
                joint.set_position_lower_limit(2, -np.pi/6)
                joint.set_position_upper_limit(2, np.pi/6)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_arm_damping)
                joint.set_damping_coefficient(1, l_arm_damping)
                joint.set_damping_coefficient(2, l_arm_damping)
            elif joint.name == "rightHand":
                joint.set_position_lower_limit(0, -np.pi/6)
                joint.set_position_upper_limit(0, np.pi/6)
                joint.set_position_lower_limit(1, -np.pi/6)
                joint.set_position_upper_limit(1, np.pi/6)
                joint.set_position_lower_limit(2, -np.pi/6)
                joint.set_position_upper_limit(2, np.pi/6)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_arm_damping)
                joint.set_damping_coefficient(1, r_arm_damping)
                joint.set_damping_coefficient(2, r_arm_damping)
            elif joint.name == "neck" or joint.name == "head":
                joint.set_position_lower_limit(0, -np.pi/6)
                joint.set_position_upper_limit(0, np.pi/6)
                joint.set_position_lower_limit(1, -np.pi/12)
                joint.set_position_upper_limit(1, np.pi/12)
                joint.set_position_lower_limit(2, -np.pi/12)
                joint.set_position_upper_limit(2, np.pi/12)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, head_damping)
                joint.set_damping_coefficient(1, head_damping)
                joint.set_damping_coefficient(2, head_damping)
            elif joint.name == "pelvis":
                joint.set_position_lower_limit(0, -np.pi/3)
                joint.set_position_upper_limit(0, np.pi/3)
                joint.set_position_lower_limit(1, -np.pi/3)
                joint.set_position_upper_limit(1, np.pi/3)
                joint.set_position_lower_limit(2, -np.pi/3)
                joint.set_position_upper_limit(2, np.pi/3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, torso_damping)
                joint.set_damping_coefficient(1, torso_damping)
                joint.set_damping_coefficient(2, torso_damping)
            else:
                print joint.name

        return skel


    def fix_joint_rest_and_stiffness(self, world):

        skel = world.skeletons[0]
        init_skel = np.array(skel.q)

        #print(skel.q[51:54], 'skel q init1')
        #print(np.sum(np.abs(init_skel - np.array(skel.q))))

        world.step()

        #print(skel.q[51:54], 'skel q init2')
        #print(np.sum(np.abs(init_skel - np.array(skel.q))))

        for joint_idx in range(np.shape(np.array(skel.joints))[0]-1):
            for i in range(3):
                angle_idx = joint_idx*3 + i + 3
                #print(angle_idx, np.array(skel.q)[angle_idx] - init_skel[angle_idx])
                if np.array(skel.q)[angle_idx] - init_skel[angle_idx] > (3/2)*np.pi:
                    print("increasing angle ", angle_idx, " by 2pi")
                    curr_lower_lim = float(skel.joints[joint_idx].position_lower_limit(i))
                    skel.joints[joint_idx].set_position_lower_limit(i, curr_lower_lim + 2*np.pi)

                    curr_upper_lim = float(skel.joints[joint_idx].position_upper_limit(i))
                    skel.joints[joint_idx].set_position_upper_limit(i, curr_upper_lim + 2*np.pi)

                    curr_rest_pos = float(skel.joints[joint_idx].rest_position(i))
                    skel.joints[joint_idx].set_rest_position(i, curr_rest_pos + 2*np.pi)


                elif np.array(skel.q)[angle_idx] - init_skel[angle_idx] < -(3/2)*np.pi:
                    print("decreasing ", angle_idx, " by 2pi")
                    curr_lower_lim = float(skel.joints[joint_idx].position_lower_limit(i))
                    skel.joints[joint_idx].set_position_lower_limit(i, curr_lower_lim - 2*np.pi)

                    curr_upper_lim = float(skel.joints[joint_idx].position_upper_limit(i))
                    skel.joints[joint_idx].set_position_upper_limit(i, curr_upper_lim - 2*np.pi)

                    curr_rest_pos = float(skel.joints[joint_idx].rest_position(i))
                    skel.joints[joint_idx].set_rest_position(i, curr_rest_pos - 2*np.pi)


            #print(joint_idx, skel.joints[joint_idx].rest_position(0), skel.joints[joint_idx].rest_position(1), skel.joints[joint_idx].rest_position(2))
            #print(joint_idx, skel.joints[joint_idx].position_lower_limit(0), skel.joints[joint_idx].position_lower_limit(1), skel.joints[joint_idx].position_lower_limit(2))
            #print(joint_idx, skel.joints[joint_idx].position_upper_limit(0), skel.joints[joint_idx].position_upper_limit(1), skel.joints[joint_idx].position_upper_limit(2))

        return world


    def fix_jump_in_axis_angle(self, world, last_joint_angles):

        skel = world.skeletons[0]
        init_skel = np.array(skel.q)


        for joint_idx in range(np.shape(np.array(skel.joints))[0]-1):
            flag_joint = False

            for i in range(3):
                angle_idx = joint_idx*3 + i + 3
                #print(joint_idx, i, last_joint_angles[angle_idx], init_skel[angle_idx])
                if np.abs(last_joint_angles[angle_idx] - init_skel[angle_idx]) > np.pi:
                    flag_joint = True

            if flag_joint == True:
                print(joint_idx, "FLAGGED JOINT!")

                for i in range(3):
                    angle_idx = joint_idx*3 + i + 3
                    print(last_joint_angles[angle_idx], init_skel[angle_idx], float(init_skel[angle_idx] - last_joint_angles[angle_idx]))
                    print("prev: ", float(skel.joints[joint_idx].rest_position(i)), float(skel.joints[joint_idx].position_lower_limit(i)), float(skel.joints[joint_idx].position_upper_limit(i)))

                    amount_to_add = float(init_skel[angle_idx] - last_joint_angles[angle_idx])

                    #print("increasing angle ", angle_idx, " by 2pi")
                    curr_lower_lim = float(skel.joints[joint_idx].position_lower_limit(i))
                    skel.joints[joint_idx].set_position_lower_limit(i, curr_lower_lim + amount_to_add)

                    curr_upper_lim = float(skel.joints[joint_idx].position_upper_limit(i))
                    skel.joints[joint_idx].set_position_upper_limit(i, curr_upper_lim + amount_to_add)

                    curr_rest_pos = float(skel.joints[joint_idx].rest_position(i))
                    skel.joints[joint_idx].set_rest_position(i, curr_rest_pos + amount_to_add)
                    print("curr: ", float(skel.joints[joint_idx].rest_position(i)), float(skel.joints[joint_idx].position_lower_limit(i)), float(skel.joints[joint_idx].position_upper_limit(i)))
        return world



    def assign_capsule_friction(self, skel, friction):
        for bn in skel.bodynodes:
            bn.set_friction_coeff(friction)
            #bn.set_restitution_coeff(1.0)
        return skel


    def get_particle_based_damping_force(self, pmat_idx_list, pmat_idx_list_prev, force_dir_list, force_dir_list_prev, force_vel_list, item, B):

        # get the velocities of the array to make the damping force
        prev_offset = 0
        for idx in range(len(pmat_idx_list[item])):
            idx_match = False
            while idx_match == False:
                try:
                    if pmat_idx_list_prev[item][idx + prev_offset] == pmat_idx_list[item][idx]:
                        force_vel_list[item].append(force_dir_list_prev[item][idx + prev_offset] - force_dir_list[item][idx])
                        idx_match = True
                    elif pmat_idx_list_prev[item][idx + prev_offset] < pmat_idx_list[item][idx]:
                        prev_offset += 1
                        if idx + prev_offset < 0 or pmat_idx_list_prev[item][idx + prev_offset] > pmat_idx_list[item][idx]:
                            force_vel_list[item].append(-force_dir_list[item][idx])
                            idx_match = True

                    elif pmat_idx_list_prev[item][idx + prev_offset] > pmat_idx_list[item][idx]:
                        prev_offset -= 1
                        if idx + prev_offset < 0 or pmat_idx_list_prev[item][idx + prev_offset] < pmat_idx_list[item][idx]:
                            force_vel_list[item].append(-force_dir_list[item][idx])
                            idx_match = True
                except:
                    force_vel_list[item].append(-force_dir_list[item][idx])
                    idx_match = True

        f_damping = -B * np.asarray(force_vel_list[item])

        # print "vel:", len(force_vel_list[item]), force_vel_list[item]
        return f_damping



    def get_particle_based_friction_force(self, f_normal, V, uk):

        # print np.dot(f_normal, V)

        # print np.linalg.norm(f_normal, axis = 1)

        # print np.square(np.linalg.norm(f_normal, axis = 1))

        # print np.dot(f_normal, V)/np.square(np.linalg.norm(f_normal, axis = 1))

        #print "projection: ", np.expand_dims(np.dot(f_normal, V) / np.square(np.linalg.norm(f_normal, axis=1)), 1) * f_normal

        if np.sum(np.linalg.norm(f_normal, axis=1)) > 0:
            f_friction_dir = (V - np.expand_dims(np.dot(f_normal, V) / np.square(np.linalg.norm(f_normal, axis=1)), 1) * f_normal)
            f_friction = -uk * (f_friction_dir / np.expand_dims(np.linalg.norm(f_friction_dir, axis=1), 1)) * np.expand_dims(np.linalg.norm(f_normal, axis=1), 1)
        else:
            f_friction = np.asarray([0., 0., 0.])

        return f_friction

    def get_capsule_based_friction_force(self, skel_bn_item, force_spring_COM, force_damping_COM, uk):
        # F_uk
        if skel_bn_item.com_linear_velocity()[0] < 0.0:
            force_friction_X = uk * np.linalg.norm(force_spring_COM + force_damping_COM)
        else:
            force_friction_X = -uk * np.linalg.norm(force_spring_COM + force_damping_COM)

        if skel_bn_item.com_linear_velocity()[1] < 0.0:
            force_friction_Y = uk * np.linalg.norm(force_spring_COM + force_damping_COM)
        else:
            force_friction_Y = -uk * np.linalg.norm(force_spring_COM + force_damping_COM)

        force_friction_COM = np.asarray([force_friction_X, force_friction_Y, 0.0])
        return force_friction_COM




    def impose_force(self, skel, body_node, force, offset_from_centroid, cap_offsets, render = True, init=True):
        ######################################### ADD INITIAL FORCE ARROWS #############################################

        rot_force = np.matrix(skel.bodynodes[body_node].T)[0:3, 0:3].I
        force_dir = np.matmul(rot_force, np.expand_dims(force / np.linalg.norm(force) * 0.5, 1))
        force_dir = np.asarray([force_dir[0, 0], force_dir[1, 0], force_dir[2, 0]])
        location = list(np.asarray(skel.bodynodes[body_node].C) + offset_from_centroid)

        arrow_tail = list(cap_offsets[body_node] + offset_from_centroid - force_dir.T)
        arrow_head = list(cap_offsets[body_node] + offset_from_centroid)  # origin is the center of the joint sphere
        #print arrow_head, 'arrow head'

        if init == True:
            if render == True:
                skel.bodynodes[body_node].add_ext_force_with_arrow(force, location, arrow_tail[0], arrow_tail[1], arrow_tail[2], arrow_head[0], arrow_head[1], arrow_head[2], body_node, False, False)
            else:
                skel.bodynodes[body_node].add_ext_force(force, location, False, False)
        else:
            if render == True:
                skel.bodynodes[body_node].set_ext_force_with_arrow(force, location, arrow_tail[0], arrow_tail[1], arrow_tail[2], arrow_head[0], arrow_head[1], arrow_head[2], body_node, False, False)
            else:
                skel.bodynodes[body_node].set_ext_force(force, location, False, False)


    def impose_torque(self, skel, body_node, torque, init=True):


        if init == True:
            skel.bodynodes[body_node].add_ext_torque(torque, False)
        else:
            skel.bodynodes[body_node].set_ext_torque(torque, False)


