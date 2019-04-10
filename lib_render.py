import numpy as np
import random
import copy
from opendr.renderer import ColoredRenderer
from opendr.renderer import DepthRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model
from util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin, reflect_pose

#volumetric pose gen libraries
import lib_visualization as libVisualization
import lib_kinematics as libKinematics
from process_yash_data import ProcessYashData
import dart_skel_sim
from time import sleep

#ROS
#import rospy
#import tf
DATASET_CREATE_TYPE = 1

import cv2

import math
from random import shuffle
import torch
import torch.nn as nn

import tensorflow as tensorflow
import cPickle as pickle


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL

import trimesh
import pyrender
import pyglet




def standard_render(m):

    ## Create OpenDR renderer
    rn = ColoredRenderer()

    ## Assign attributes to renderer
    w, h = (640, 480)

    rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

    ## Construct point light source
    rn.vc = LambertianPointLight(
        f=m.f,
        v=rn.v,
        num_verts=len(m),
        light_pos=np.array([-1000,-1000,-2000]),
        vc=np.ones_like(m)*.9,
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

class pyRenderMesh():
    def __init__(self):
        ## Create OpenDR renderer
        self.rn = ColoredRenderer()


        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.scene = pyrender.Scene()


    def mesh_render(self,m):

        #print m.r
        #print np.max(m.f)

        tm = trimesh.base.Trimesh(vertices=m.r, faces=m.f)

        #tm = trimesh.load('/home/henry/Downloads/fuze.obj')

        human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 1.0 ,0.5])
        pymesh = pyrender.Mesh.from_trimesh(tm, material=human_mat, wireframe = True)




        #print tm.vertices
        #print np.max(tm.faces)


        #from pyglet.window import Window
        #from pyglet.gl import Config;
        #w = Window(config=Config(major_version=4, minor_version=1))
        #print w.context
        #print('{}.{}'.format(w.context.config.major_version, w.context.config.minor_version))

        #print self.scene
        #print self.pymesh



        #print "Viewing"
        if self.first_pass == True:

            ## Assign attributes to renderer
            w, h = (1000, 1000)
            self.rn.camera = ProjectPoints(
                v=m,
                rt=np.array([np.pi, 0.0, 0.0]),
                t=np.array([0, -0.3, 2.0]),
                f=np.array([w, w]) / 2.,
                c=np.array([w, h]) / 2.,
                k=np.zeros(5))
            self.rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
            self.rn.set(v=m, f=m.f, bgcolor=np.array([1.0, 1.0, 1.0]))

            ## Construct point light source
            self.rn.vc = LambertianPointLight(
                f=m.f,
                v=self.rn.v,
                num_verts=len(m),
                light_pos=np.array([1000, 1000, 2000]),
                vc=np.ones_like(m) * .9,
                light_color=np.array([1.0, 0.7, 0.65]))

            self.scene.add(pymesh)
            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
            self.first_pass = False
            for node in self.scene.get_nodes(obj=pymesh):
                # print node
                self.human_obj_node = node


        else:
            self.viewer.render_lock.acquire()
            self.scene.remove_node(self.human_obj_node)

            self.scene.add(pymesh)


            for node in self.scene.get_nodes(obj=pymesh):
                # print node
                self.human_obj_node = node
            self.viewer.render_lock.release()

        sleep(0.01)
        print "BLAH"

        #print np.shape(rn.r)
        #sleep(5)

        #clipped_render = rn.r[h/8:7*h/8, w/4:3*w/4, :]


        ## Show it using OpenCV
        #cv2.imshow('render_SMPL', clipped_render)
        #print ('..Print any key while on the display window')
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


        ## Could also use matplotlib to display
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.imshow(rn.r)
        # plt.show()
        # import pdb; pdb.set_trace()