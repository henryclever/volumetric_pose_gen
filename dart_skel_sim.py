# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group
import numpy as np
import pydart2 as pydart
from pydart2 import skeleton_builder
from dart_opengl_window import GLUTWindow


class DampingController(object):
    """ Add damping force to the skeleton """
    def __init__(self, skel):
        self.skel = skel

    def compute(self):
        damping = -0.01 * self.skel.dq
        damping[1::3] *= 0.1
        return damping

class DartSkelSim(object):
    def __init__(self, render, m, capsules, joint_names, initial_rots):
        self.render_dart = render

        joint_ref = list(m.kintree_table[1]) #joints
        parent_ref = list(m.kintree_table[0]) #parent of each joint
        parent_ref[0] = -1



        pydart.init(verbose=True)
        print('pydart initialization OK')

        self.world = pydart.World(0.0002, "EMPTY")
        self.world.set_gravity([0, 0,  9.81])
        self.world.set_collision_detector(detector_type=2)
        self.world.create_empty_skeleton(_skel_name="human")




        joint_root_loc = np.asarray(np.transpose(capsules[0].t)[0])
        joint_locs = []
        capsule_locs = []
        joint_locs_abs = []
        capsule_locs_abs = []

        mJ = np.asarray(m.J)


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

        #print red_joint_ref
        #print red_parent_ref

        #make lists of the locations of the joint locations and the smplify capsule initial ends
        for i in range(np.shape(mJ)[0]):
            if i == 0:
                joint_locs.append(list(mJ[0, :] - mJ[0, :]))
                joint_locs_abs.append(list(mJ[0, :] - mJ[0, :]))
                if i < 20:
                    capsule_locs.append(list(np.asarray(np.transpose(capsules[i].t)[0]) - joint_root_loc))
                    capsule_locs_abs.append(list(np.asarray(np.transpose(capsules[i].t)[0]) - joint_root_loc))
                    print(capsule_locs_abs, "caps locs abs")
            else:
                joint_locs.append(list(mJ[i, :] - mJ[parent_ref[i], :]))
                joint_locs_abs.append(list(mJ[i, :] - mJ[0, :]))
                if i < 20:
                    capsule_locs.append(list(np.asarray(np.transpose(capsules[i].t)[0]) - np.asarray(np.transpose(capsules[red_parent_ref[i]].t)[0])))
                    capsule_locs_abs.append(list(np.asarray(np.transpose(capsules[i].t)[0]) - joint_root_loc))
                    capsule_locs_abs[i][0] += float(capsules[0].length[0]) / 2
                    if i in [1, 2]: #shift over the legs relative to where the pelvis mid capsule is
                        capsule_locs[i][0] += float(capsules[0].length[0]) / 2
                    if i in [3, 6, 9]: #shift over the torso segments relative to their length and their parents length to match the mid capsule
                        capsule_locs[i][0] -= (float(capsules[i].length[0])-float(capsules[red_parent_ref[i]].length[0]))/2
                    if i in [10, 11, 12]: #shift over the inner shoulders and neck to match the middle of the top spine capsule
                        capsule_locs[i][0] += float(capsules[red_parent_ref[i]].length[0]) / 2
                    if i in [3, 6, 9]: #shift over everything to match the root
                        capsule_locs_abs[i][0] -= float(capsules[i].length[0]) / 2

        del(joint_locs[10])
        del(joint_locs[10])
        del(joint_locs_abs[10])
        del(joint_locs_abs[10])

        #print len(joint_locs)

        #for i in range(20):
        #    print joint_names[i]
        #    print "joint locs m: ", joint_locs[i]
        #    print "joint locs s: ", capsule_locs[i]

        count = 0

        #joint_locs = capsule_locs

        for capsule in capsules:
            print "************* Capsule No.",count, joint_names[count], " joint ref: ", red_joint_ref[count]," parent_ref: ", red_parent_ref[count]," ****************"
            cap_rad = float(capsule.rad[0])
            cap_len = float(capsule.length[0])
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
            if count in head_ref: cap_offset[1] = cap_len/2

            cap_offset[0] += capsule_loc_abs[0] - joint_loc_abs[0]
            cap_offset[1] += capsule_loc_abs[1] - joint_loc_abs[1] - .04
            cap_offset[2] += capsule_loc_abs[2] - joint_loc_abs[2]


            print "radius: ", cap_rad, "   length: ", cap_len
            print "Rot0: ", cap_init_rot
            print "T joint: ", joint_loc
            print "T capsu: ", capsule_loc

            #print list((np.asarray(np.transpose(capsule.t)[0])))

            #print joint_root_loc
            #print "Rot: ", list(np.asarray(capsule.rod))
            #print capsule.centers
            self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len, cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc, joint_damping = 5.0, joint_type="BALL", joint_name=joint_names[count])
            #self.world.add_capsule(parent=int(-1), radius=cap_rad, length=0.001, cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc_abs, joint_damping = 5.0, joint_type="BALL", joint_name=joint_names[count])

            #capsule_loc_abs[0] += 1.0
            #self.world.add_capsule(parent=int(-1), radius=cap_rad, length=0.001, cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=capsule_loc_abs, joint_damping = 5.0, joint_type="BALL", joint_name=joint_names[count])


            #if count == 10:
            #    break
            count += 1

        #self.world.add_capsule(parent=-1, radius=0.1, length=1.0, joint_loc=[0, 0, 0], joint_damping = 5.0, joint_type="BALL", joint_name="joint 1")
        #self.world.add_capsule(parent=0, radius=0.1, length=0.5, joint_loc=[0, 0, 1.0], joint_damping = 5.0, joint_type="BALL", joint_name="joint 2")
        #self.world.add_capsule(parent=0, radius=0.1, length=1.0, joint_loc=[0, 0, 1.0], joint_damping = 5.0, joint_type="REVOLUTE", joint_name="joint 3")
        #self.world.add_capsule(parent=2, radius=0.1, length=0.5, joint_loc=[0, 0, 1.0], joint_damping = 5.0, joint_type="REVOLUTE", joint_name="joint 4")
        #self.world.add_capsule(parent=3, radius=0.1, length=0.5, joint_loc=[0, 0, 1.0], joint_damping = 5.0, joint_type="REVOLUTE", joint_name="joint 5")

        built_skel = self.world.add_built_skeleton(_skel_id=0, _skel_name="human")
        built_skel.set_self_collision_check(True)


        skel = self.world.skeletons[0]
        #skel_q_init = np.random.rand(skel.ndofs) - 0.5
        skel_q_init = skel.ndofs * [0]
        #skel_q_init[3] = np.pi/2
        print skel_q_init
        skel.set_positions(skel_q_init)
        #print skel.root_bodynode()
        #print skel.name
        #from pydart2 import bodynode
        #bn = bodynode.BodyNode(skel, 8)
        for body in skel.bodynodes:
            print body.C

        for joint in skel.joints:
            print joint

        for marker in skel.markers:
            print marker.world_position()


        print('init pose = %s' % skel.q)
        skel.controller = DampingController(skel)


    def run(self):
        #pydart.gui.viewer.launch(world)

        if self.render_dart == False:
            for i in range(0, 100):
                self.world.step()
                print "did a step"
                skel = self.world.skeletons[0]
                print skel.q

        elif self.render_dart == True:
            win = GLUTWindow(self.world, title=None)
            default_camera = None
            if default_camera is not None:
                win.scene.set_camera(default_camera)
            win.run()


if __name__ == '__main__':
    dss = DartSkelSim(render=True)
    dss.run()
