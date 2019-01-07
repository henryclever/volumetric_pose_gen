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

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
from pydart2.gui.opengl.scene import OpenGLScene
from time import time


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

        self.capsules = capsules


        pydart.init(verbose=True)
        print('pydart initialization OK')

        self.world = pydart.World(0.0002, "EMPTY")
        self.world.set_gravity([0, 0,  -9.81])
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

        self.red_joint_ref = red_joint_ref
        self.red_parent_ref = red_parent_ref


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
                    if i in [3, 6, 9]: #shift over everything in the abs list to match the root
                        capsule_locs_abs[i][0] -= float(capsules[i].length[0]) / 2

        del(joint_locs[10])
        del(joint_locs[10])
        del(joint_locs_abs[10])
        del(joint_locs_abs[10])

        self.joint_locs = joint_locs


        count = 0

        self.cap_offsets = []
        self.cap_init_rots = []
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
            self.cap_offsets.append(np.asarray(cap_offset))
            self.cap_init_rots.append(np.asarray(cap_init_rot))


            print "radius: ", cap_rad, "   length: ", cap_len
            print "Rot0: ", cap_init_rot
            print "T joint: ", joint_loc_abs
            print "T capsu: ", capsule_loc_abs

            #print list((np.asarray(np.transpose(capsule.t)[0])))

            #print joint_root_loc
            #print "Rot: ", list(np.asarray(capsule.rod))
            #print capsule.centers
            if count == 4 or count == 5:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type="REVOLUTE_X", joint_name=joint_names[count])
            elif count == 16 or count == 17:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type="REVOLUTE_Y", joint_name=joint_names[count])
            elif count == 0:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type="BALL", joint_name=joint_names[count])
            else:
                self.world.add_capsule(parent=int(red_parent_ref[count]), radius=cap_rad, length=cap_len,
                                       cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc,
                                       joint_type="BALL", joint_name=joint_names[count])

            #self.world.add_capsule(parent=int(-1), radius=cap_rad, length=0.001, cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=joint_loc_abs, joint_type="BALL", joint_name=joint_names[count])
            #capsule_loc_abs[0] += 1.0
            #self.world.add_capsule(parent=int(-1), radius=cap_rad, length=0.001, cap_rot=cap_init_rot, cap_offset=cap_offset, joint_loc=capsule_loc_abs, joint_type="BALL", joint_name=joint_names[count])


            #if count == 10:
            #    break
            count += 1

        built_skel = self.world.add_built_skeleton(_skel_id=0, _skel_name="human")
        built_skel.set_self_collision_check(True)


        skel = self.world.skeletons[0]

        skel = LibDartSkel().assign_init_joint_angles(skel, m)

        skel = LibDartSkel().assign_joint_limits_and_damping(skel)

        #weight the capsules appropriately
        for body_ct in range(len(skel.bodynodes)):
            #give the capsules a weight propertional to their volume
            cap_rad = float(capsules[body_ct].rad[0])
            cap_len = float(capsules[body_ct].length[0])
            volume = np.pi*np.square(cap_rad)*(cap_rad*4/3 + cap_len)

            skel.bodynodes[body_ct].set_mass(volume*1000)
            #print "COM: ", skel.bodynodes[body_ct].C
            #print "MASS: ", skel.bodynodes[body_ct].m




        self.body_node = 9 #need to solve for the body node that corresponds to a force using flex.
        self.force = np.asarray([0.0, 0.0, 100.0])
        self.offset_from_centroid = np.asarray([-0.15, 0.0, 0.0])

        #LibDartSkel().impose_force(skel=skel, body_node=self.body_node, force=self.force,
        #                           offset_from_centroid = self.offset_from_centroid, cap_offsets = self.cap_offsets,
        #                           render=self.render_dart, init=True)



        for marker in skel.markers:
            print marker.world_position()


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

    def idle(self, ):
        if self.world is None:
            return
        if self.is_simulating:
            self.world.step()
            print "did a step"
            self.world.check_collision()
            print self.world.collision_result.contacted_bodies
            skel = self.world.skeletons[0]
            print skel.q



            #LibDartSkel().impose_force(skel=skel, body_node=self.body_node, force=self.force,
            #                           offset_from_centroid = self.offset_from_centroid, cap_offsets = self.cap_offsets,
            #                           render=True, init=False)


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

    def run_sim_with_window(self, ):
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



    def run_sim_step(self):
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        self.world.step()
        #print "did a step"
        skel = self.world.skeletons[0]

        #LibDartSkel().impose_force(skel=skel, body_node=self.body_node, force=self.force, offset_from_centroid=self.offset_from_centroid, cap_offsets=self.cap_offsets, render=False, init=False)

        return skel.q, skel.bodynodes


    def run_simulation(self):
        #pydart.gui.viewer.launch(world)

        #run without visualizing
        if self.render_dart == False:
            for i in range(0, 100):
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
