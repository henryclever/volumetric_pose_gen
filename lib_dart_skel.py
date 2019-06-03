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

        skel_q_init[index_offset+12] = float(m.pose[12]) #left knee. should be 0.0 to 3.14
        skel_q_init[index_offset+13] = float(m.pose[15]) #right knee. should be 0.0 to 3.14

        skel_q_init[index_offset+14:17+index_offset] = np.asarray(m.pose[18:21]) #mid spine

        skel_q_init[index_offset+17:20+index_offset] = np.asarray(m.pose[21:24]) #left foot
        skel_q_init[index_offset+20:23+index_offset] = np.asarray(m.pose[24:27]) #right foot

        skel_q_init[index_offset+23:26+index_offset] = np.asarray(m.pose[27:30]) #upper spine

        skel_q_init[index_offset+26:29+index_offset] = np.asarray(m.pose[36:39]) #neck

        skel_q_init[index_offset+29:32+index_offset] = np.asarray(m.pose[39:42]) #left inner shoulder
        skel_q_init[index_offset+32:35+index_offset] = np.asarray(m.pose[42:45]) #right inner shoulder

        skel_q_init[index_offset+35:38+index_offset] = np.asarray(m.pose[45:48]) #head

        skel_q_init[index_offset+38:41+index_offset] = np.asarray(m.pose[48:51]) #left outer shoulder
        skel_q_init[index_offset+41:44+index_offset] = np.asarray(m.pose[51:54]) #right outer shoulder

        skel_q_init[index_offset+44] = float(m.pose[55]) #left elbow. should be -3.14 to 0.0
        skel_q_init[index_offset+45] = float(m.pose[58]) #right elbow. should be  0.0 to 3.14

        skel_q_init[index_offset+46:49+index_offset] = np.asarray(m.pose[60:63]) #left hand
        skel_q_init[index_offset+49:52+index_offset] = np.asarray(m.pose[63:66]) #right hand

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
            torso_stiffness = 300.0*bm_fraction
        else:
            torso_stiffness = 300.0*bm_fraction

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
            r_arm_stiffness = 2.0*bm_fraction
            l_arm_stiffness = 2.0*bm_fraction
            head_stiffness = 200.0*bm_fraction
            r_leg_stiffness = 10.0*bm_fraction
            l_leg_stiffness = 10.0*bm_fraction
            r_knee_stiffness = 10.0*bm_fraction
            l_knee_stiffness = 10.0*bm_fraction


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
                joint.set_spring_stiffness(0, l_knee_stiffness)
            elif joint.name == "rightCalf":
                joint.set_rest_position(0, float(m.pose[15]))
                joint.set_spring_stiffness(0, r_knee_stiffness)
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
                joint.set_rest_position(0, float(m.pose[55]))
                joint.set_spring_stiffness(0, l_arm_stiffness)
            elif joint.name == "rightForeArm":
                joint.set_rest_position(0, float(m.pose[58]))
                joint.set_spring_stiffness(0, r_arm_stiffness)
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
        head_damping = head_stiffness*5
        r_leg_damping = r_leg_stiffness*5
        l_leg_damping = l_leg_stiffness*5
        r_knee_damping = r_knee_stiffness*5
        l_knee_damping = l_knee_stiffness*5
        torso_damping = torso_stiffness*5

        for joint in skel.joints:
            print joint.spring_stiffness(0)

            if joint.name == "leftThigh":
                joint.set_position_lower_limit(0, -2.6630969584625968)  # ext
                joint.set_position_upper_limit(0, -0.14634814003149707)
                joint.set_position_lower_limit(1, -1.0403111466710133)  # yaw
                joint.set_position_upper_limit(1, 0.7797238444581264)
                joint.set_position_lower_limit(2, -0.3678530987804197)  # abd
                joint.set_position_upper_limit(2, 0.8654752836217577)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_leg_damping)
                joint.set_damping_coefficient(1, l_leg_damping)
                joint.set_damping_coefficient(2, l_leg_damping)
            elif joint.name == "rightThigh":
                joint.set_position_lower_limit(0, -2.6630969584625968)  # ext
                joint.set_position_upper_limit(0, -0.14634814003149707)
                joint.set_position_lower_limit(1, -0.7797238444581264)  # yaw
                joint.set_position_upper_limit(1, 1.0403111466710133)
                joint.set_position_lower_limit(2, -0.8654752836217577)  # abd
                joint.set_position_upper_limit(2, 0.3678530987804197)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_leg_damping)
                joint.set_damping_coefficient(1, r_leg_damping)
                joint.set_damping_coefficient(2, r_leg_damping)
            elif joint.name == "spine" or joint.name == "spine1" or joint.name == "spine2":
                joint.set_position_lower_limit(0, -np.pi/6)
                joint.set_position_upper_limit(0, np.pi/6)
                joint.set_position_lower_limit(1, -np.pi/36)
                joint.set_position_upper_limit(1, np.pi/36)
                joint.set_position_lower_limit(2, -np.pi/36)
                joint.set_position_upper_limit(2, np.pi/36)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, torso_damping)
                joint.set_damping_coefficient(1, torso_damping)
                joint.set_damping_coefficient(2, torso_damping)
            elif joint.name == "leftCalf":
                joint.set_position_lower_limit(0, 0.0)
                joint.set_position_upper_limit(0, 2.7115279828308503)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_knee_damping)
            elif joint.name == "rightCalf":
                joint.set_position_lower_limit(0, 0.0)
                joint.set_position_upper_limit(0, 2.7115279828308503)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_knee_damping)
            elif joint.name == "leftFoot":
                joint.set_position_lower_limit(0, -np.pi/6)
                joint.set_position_upper_limit(0, np.pi/6)
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
                joint.set_position_upper_limit(0, np.pi/6)
                joint.set_position_lower_limit(1, -np.pi/6)
                joint.set_position_upper_limit(1, np.pi/6)
                joint.set_position_lower_limit(2, -np.pi/6)
                joint.set_position_upper_limit(2, np.pi/6)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_leg_damping)
                joint.set_damping_coefficient(1, r_leg_damping)
                joint.set_damping_coefficient(2, r_leg_damping)
            elif joint.name == "leftShoulder":
                joint.set_position_lower_limit(0,  -1.8674195346872975 * 1 / 3)  # roll
                joint.set_position_upper_limit(0, 1.410545172086535 * 1 / 3)
                joint.set_position_lower_limit(1, -1.530112726921327 * 1 / 3)  # yaw
                joint.set_position_upper_limit(1, 1.2074724617209949 * 1 / 3)
                joint.set_position_lower_limit(2, -1.9550515937478927 * 1 / 3)  # pitch
                joint.set_position_upper_limit(2, 1.7587935205169856 * 1 / 3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_arm_damping)
                joint.set_damping_coefficient(1, l_arm_damping)
                joint.set_damping_coefficient(2, l_arm_damping)
            elif joint.name == "leftUpperArm":
                joint.set_position_lower_limit(0, -1.8674195346872975 * 2 / 3)  # roll
                joint.set_position_upper_limit(0, 1.410545172086535 * 2 / 3)
                joint.set_position_lower_limit(1, -1.530112726921327 * 2 / 3)  # yaw
                joint.set_position_upper_limit(1, 1.2074724617209949 * 2 / 3)
                joint.set_position_lower_limit(2, -1.9550515937478927 * 2 / 3)  # pitch
                joint.set_position_upper_limit(2, 1.7587935205169856 * 2 / 3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_arm_damping)
                joint.set_damping_coefficient(1, l_arm_damping)
                joint.set_damping_coefficient(2, l_arm_damping)
            elif joint.name == "rightShoulder":
                joint.set_position_lower_limit(0, -1.8674195346872975 * 1 / 3)  # roll
                joint.set_position_upper_limit(0, 1.410545172086535 * 1 / 3)
                joint.set_position_lower_limit(1, -1.2074724617209949 * 1 / 3)  # yaw
                joint.set_position_upper_limit(1, 1.530112726921327  * 1 / 3)
                joint.set_position_lower_limit(2, -1.7587935205169856 * 1 / 3)  # pitch
                joint.set_position_upper_limit(2, 1.9550515937478927 * 1 / 3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_arm_damping)
                joint.set_damping_coefficient(1, r_arm_damping)
                joint.set_damping_coefficient(2, r_arm_damping)
            elif joint.name == "rightUpperArm":
                joint.set_position_lower_limit(0, -1.8674195346872975 * 2 / 3)  # roll
                joint.set_position_upper_limit(0, 1.410545172086535 * 2 / 3)
                joint.set_position_lower_limit(1, -1.2074724617209949 * 2 / 3)  # yaw
                joint.set_position_upper_limit(1, 1.530112726921327 * 2 / 3)
                joint.set_position_lower_limit(2, -1.7587935205169856 * 2 / 3)  # pitch
                joint.set_position_upper_limit(2, 1.9550515937478927 * 2 / 3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_arm_damping)
                joint.set_damping_coefficient(1, r_arm_damping)
                joint.set_damping_coefficient(2, r_arm_damping)
            elif joint.name == "leftForeArm":
                joint.set_position_lower_limit(0, -2.463868908637374)
                joint.set_position_upper_limit(0, 0.0)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, l_arm_damping)
            elif joint.name == "rightForeArm":
                joint.set_position_lower_limit(0, 0.0)
                joint.set_position_upper_limit(0, 2.463868908637374)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, r_arm_damping)
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
                joint.set_position_lower_limit(1, -np.pi/36)
                joint.set_position_upper_limit(1, np.pi/36)
                joint.set_position_lower_limit(2, -np.pi/36)
                joint.set_position_upper_limit(2, np.pi/36)
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


