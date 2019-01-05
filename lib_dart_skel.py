# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group
import numpy as np
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
    def assign_init_joint_angles(self, skel, m):
        ####################################### ASSIGN INITIAL JOINT ANGLES ############################################
        #skel_q_init = np.random.rand(skel.ndofs) - 0.5
        skel_q_init = skel.ndofs * [0]
        skel_q_init[3:6] = np.asarray(m.pose[3:6]) #left glute
        skel_q_init[6:9] = np.asarray(m.pose[6:9]) #right glute
        skel_q_init[9:12] = np.asarray(m.pose[9:12]) #low spine

        skel_q_init[12] = float(m.pose[12]) #left knee. should be 0.0 to 3.14
        skel_q_init[13] = float(m.pose[15]) #right knee. should be 0.0 to 3.14

        skel_q_init[14:17] = np.asarray(m.pose[18:21]) #mid spine

        skel_q_init[17:20] = np.asarray(m.pose[21:24]) #left foot
        skel_q_init[20:23] = np.asarray(m.pose[24:27]) #right foot

        skel_q_init[23:26] = np.asarray(m.pose[27:30]) #upper spine

        skel_q_init[26:29] = np.asarray(m.pose[36:39]) #neck

        skel_q_init[29:32] = np.asarray(m.pose[39:42]) #left inner shoulder
        skel_q_init[32:35] = np.asarray(m.pose[42:45]) #right inner shoulder

        skel_q_init[35:38] = np.asarray(m.pose[45:48]) #head

        skel_q_init[38:41] = np.asarray(m.pose[48:51]) #left outer shoulder
        skel_q_init[41:44] = np.asarray(m.pose[51:54]) #right outer shoulder

        skel_q_init[44] = float(m.pose[55]) #left elbow. should be -3.14 to 0.0
        skel_q_init[45] = float(m.pose[58]) #right elbow. should be  0.0 to 3.14

        skel_q_init[46:49] = np.asarray(m.pose[60:63]) #left hand
        skel_q_init[49:52] = np.asarray(m.pose[63:66]) #right hand



        #this is where you set the angles according to m, the angle axis representation.
        skel.set_positions(skel_q_init)
        #print skel.root_bodynode()
        #print skel.name
        #from pydart2 import bodynode
        #bn = bodynode.BodyNode(skel, 8)
        return skel

    def assign_joint_limits_and_damping(self, skel):
        ######################################## ASSIGN JOINT LIMITS AND DAMPING #######################################

        arm_damping = 1.0
        leg_damping = 2.0
        head_damping = 2.0
        torso_damping = 5.0
        for joint in skel.joints:
            print joint

            if joint.name == "leftThigh":
                joint.set_position_lower_limit(0, -0.9999455988016526) #ext
                joint.set_position_upper_limit(0, -0.1716378496297634)
                joint.set_position_lower_limit(1, -0.9763807967155833) #yaw
                joint.set_position_upper_limit(1,  0.9792771003511667)
                joint.set_position_lower_limit(2, -0.35342183756490175) #abd
                joint.set_position_upper_limit(2,  0.9029919511354418)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, leg_damping)
                joint.set_damping_coefficient(1, leg_damping)
                joint.set_damping_coefficient(2, leg_damping)
            elif joint.name == "rightThigh":
                joint.set_position_lower_limit(0, -0.9999902632535546) #ext
                joint.set_position_upper_limit(0, -0.1026015392807176)
                joint.set_position_lower_limit(1, -0.9701910881430709) #yaw
                joint.set_position_upper_limit(1,  0.9821826206061518)
                joint.set_position_lower_limit(2, -0.8767199629302654) #abd
                joint.set_position_upper_limit(2,  0.35738032396710084)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, leg_damping)
                joint.set_damping_coefficient(1, leg_damping)
                joint.set_damping_coefficient(2, leg_damping)
            elif joint.name == "leftCalf":
                joint.set_position_lower_limit(0, 0.0)
                joint.set_position_upper_limit(0, 2.3720944626178713)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, leg_damping)
            elif joint.name == "rightCalf":
                joint.set_position_lower_limit(0, 0.0)
                joint.set_position_upper_limit(0, 2.320752282574325)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, leg_damping)
            elif joint.name == "leftShoulder":
                joint.set_position_lower_limit(0, -1.9811361489978918*1/3) #roll
                joint.set_position_upper_limit(0,  1.4701759095910327*1/3)
                joint.set_position_lower_limit(1, -1.5656401670211908*1/3) #yaw
                joint.set_position_upper_limit(1,  1.047255481259413*1/3)
                joint.set_position_lower_limit(2, -1.9671878788002621*1/3) #pitch
                joint.set_position_upper_limit(2,  1.3280993848963953*1/3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, arm_damping)
                joint.set_damping_coefficient(1, arm_damping)
                joint.set_damping_coefficient(2, arm_damping)
            elif joint.name == "leftUpperArm":
                joint.set_position_lower_limit(0, -1.9811361489978918*2/3) #roll
                joint.set_position_upper_limit(0,  1.4701759095910327*2/3)
                joint.set_position_lower_limit(1, -1.5656401670211908*2/3) #yaw
                joint.set_position_upper_limit(1,  1.047255481259413*2/3)
                joint.set_position_lower_limit(2, -1.9671878788002621*2/3) #pitch
                joint.set_position_upper_limit(2,  1.3280993848963953*2/3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, arm_damping)
                joint.set_damping_coefficient(1, arm_damping)
                joint.set_damping_coefficient(2, arm_damping)
            elif joint.name == "rightShoulder":
                joint.set_position_lower_limit(0, -1.7735924284100764*1/3) #roll
                joint.set_position_upper_limit(0,  1.7843466954767204*1/3)
                joint.set_position_lower_limit(1, -1.3128987757338355*1/3) #yaw
                joint.set_position_upper_limit(1,  1.5001029778132429*1/3)
                joint.set_position_lower_limit(2, -1.483831592135514*1/3) #pitch
                joint.set_position_upper_limit(2,  2.050392704184662*1/3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, arm_damping)
                joint.set_damping_coefficient(1, arm_damping)
                joint.set_damping_coefficient(2, arm_damping)
            elif joint.name == "rightUpperArm":
                joint.set_position_lower_limit(0, -1.7735924284100764*2/3) #roll
                joint.set_position_upper_limit(0,  1.7843466954767204*2/3)
                joint.set_position_lower_limit(1, -1.3128987757338355*2/3) #yaw
                joint.set_position_upper_limit(1,  1.5001029778132429*2/3)
                joint.set_position_lower_limit(2, -1.483831592135514*2/3) #pitch
                joint.set_position_upper_limit(2,  2.050392704184662*2/3)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, arm_damping)
                joint.set_damping_coefficient(1, arm_damping)
                joint.set_damping_coefficient(2, arm_damping)
            elif joint.name == "leftForeArm":
                joint.set_position_lower_limit(0, -2.3104353421664428)
                joint.set_position_upper_limit(0, 0.0)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, arm_damping)
            elif joint.name == "rightForeArm":
                joint.set_position_lower_limit(0, 0.0)
                joint.set_position_upper_limit(0, 2.206311095551016)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, arm_damping)
            elif joint.name == "rightForeArm":
                joint.set_position_lower_limit(0, 0.0)
                joint.set_position_upper_limit(0, 2.206311095551016)
                joint.set_position_limit_enforced(True)
                joint.set_damping_coefficient(0, arm_damping)
            elif joint.name == "leftHand" or joint.name == "rightHand":
                joint.set_damping_coefficient(0, arm_damping)
                joint.set_damping_coefficient(1, arm_damping)
                joint.set_damping_coefficient(2, arm_damping)
            elif joint.name == "neck" or joint.name == "head":
                joint.set_damping_coefficient(0, head_damping)
                joint.set_damping_coefficient(1, head_damping)
                joint.set_damping_coefficient(2, head_damping)
            else:
                joint.set_damping_coefficient(0, torso_damping)
                joint.set_damping_coefficient(1, torso_damping)
                joint.set_damping_coefficient(2, torso_damping)

        return skel

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
                skel.bodynodes[body_node].add_ext_force_with_arrow(force, location, arrow_tail[0], arrow_tail[1], arrow_tail[2], arrow_head[0], arrow_head[1], arrow_head[2], False, False)
            else:
                skel.bodynodes[body_node].add_ext_force(force, location, False, False)
        else:
            if render == True:
                skel.bodynodes[body_node].set_ext_force_with_arrow(force, location, arrow_tail[0], arrow_tail[1], arrow_tail[2], arrow_head[0], arrow_head[1], arrow_head[2], False, False)
            else:
                skel.bodynodes[body_node].set_ext_force(force, location, False, False)


