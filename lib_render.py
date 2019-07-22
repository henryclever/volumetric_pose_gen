
import trimesh
import pyrender
import pyglet

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
#import dart_skel_sim
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
import matplotlib.cm as cm #use cm.jet(list)
#from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL

import cPickle as pkl

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import os





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


def render_rviz(self, mJ_transformed):

    print mJ_transformed
    rospy.init_node('smpl_model')

    shift_sideways = np.zeros((24,3))
    shift_sideways[:, 0] = 1.0

    for i in range(0, 10):
        #libVisualization.rviz_publish_output(None, np.array(self.m.J_transformed))
        time.sleep(0.5)

        concatted = np.concatenate((np.array(mJ_transformed), np.array(mJ_transformed) + shift_sideways), axis = 0)
        #print concatted
        #libVisualization.rviz_publish_output(None, np.array(self.m.J_transformed) + shift_sideways)
        libVisualization.rviz_publish_output(None, concatted)
        time.sleep(0.5)



class pyRenderMesh():
    def __init__(self):
        ## Create OpenDR renderer
        self.rn = ColoredRenderer()


        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.scene = pyrender.Scene()
        self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 1.0 ,0.0])
        self.mesh_parts_mat_list = [
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[166. / 255., 206. / 255., 227. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[31. / 255., 120. / 255., 180. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[251. / 255., 154. / 255., 153. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[227. / 255., 26. / 255., 28. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[178. / 255., 223. / 255., 138. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[51. / 255., 160. / 255., 44. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[253. / 255., 191. / 255., 111. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[255. / 255., 127. / 255., 0. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[202. / 255., 178. / 255., 214. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[106. / 255., 61. / 255., 154. / 255., 0.0])]

        self.artag_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 1.0, 0.3, 0.5])
        self.artag_mat_other = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 0.0])
        self.artag_r = np.array([[-0.055, -0.055, 0.0], [-0.055, 0.055, 0.0], [0.055, -0.055, 0.0], [0.055, 0.055, 0.0]])
        self.artag_f = np.array([[0, 1, 3], [3, 1, 0], [0, 2, 3], [3, 2, 0], [1, 3, 2]])
        self.artag_facecolors_root = np.array([[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0]])
        self.artag_facecolors = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],])

        self.segmented_dict = load_pickle('segmented_mesh_idx_faces.p')

    def mesh_render(self,m):

        #print m.r
        #print np.max(m.f)

        tm = trimesh.base.Trimesh(vertices=m.r, faces=m.f)

        #tm = trimesh.load('/home/henry/Downloads/fuze.obj')

        smpl_mesh = pyrender.Mesh.from_trimesh(tm, material=self.human_mat, wireframe = True)#, smooth = True) smoothing doesn't do anything to wireframe



        #print tm.vertices
        #print np.max(tm.faces)


        from pyglet.window import Window
        from pyglet.gl import Config;
        w = Window(config=Config(major_version=4, minor_version=1))
        print w.context
        print('{}.{}'.format(w.context.config.major_version, w.context.config.minor_version))

        #print self.scene
        #print self.smpl_mesh



        print "Viewing"
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

            self.scene.add(smpl_mesh)
            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
            self.first_pass = False
            for node in self.scene.get_nodes(obj=smpl_mesh):
                # print node
                self.human_obj_node = node


        else:
            #self.viewer.render_lock.acquire()
            #self.scene.remove_node(self.human_obj_node)
            #self.scene.add(smpl_mesh)

            #for node in self.scene.get_nodes(obj=smpl_mesh):
                # print node
            #    self.human_obj_node = node
            #self.viewer.render_lock.release()

            pass
        sleep(0.01)
        print "BLAH"




    def get_3D_pmat_markers(self, pmat, angle = 60.0):

        pmat_reshaped = pmat.reshape(64, 27)

        pmat_colors = cm.jet(pmat_reshaped/100)
        print pmat_colors.shape
        pmat_colors[:, :, 3] = 0.5

        pmat_xyz = np.zeros((64, 27, 3))
        pmat_faces = []
        pmat_facecolors = []

        for j in range(64):
            for i in range(27):

                pmat_xyz[j, i, 1] = i * 0.0286 * 1.06 - 0.04#1.0926 - 0.02
                if j > 23:
                    pmat_xyz[j, i, 0] = ((64 - j) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(angle)))*1.04 + 0.11#1.1406 + 0.05
                    pmat_xyz[j, i, 2] = 0.12
                    # print marker.pose.position.x, 'x'
                else:

                    pmat_xyz[j, i, 0] = ((41) * 0.0286 + (23 - j) * 0.0286 * np.cos(np.deg2rad(angle)) \
                                        - (0.0286 * 3 * np.sin(np.deg2rad(angle))) * 0.85)*1.04 + 0.12#1.1406 + 0.05
                    pmat_xyz[j, i, 2] = -((23 - j) * 0.0286 * np.sin(np.deg2rad(angle))) * 0.85 + 0.12
                    # print j, marker.pose.position.z, marker.pose.position.y, 'head'

                if j < 63 and i < 26:
                    coord1 = j * 27 + i
                    coord2 = j * 27 + i + 1
                    coord3 = (j + 1) * 27 + i
                    coord4 = (j + 1) * 27 + i + 1

                    pmat_faces.append([coord1, coord2, coord3]) #bottom surface
                    pmat_faces.append([coord1, coord3, coord2]) #top surface
                    pmat_faces.append([coord4, coord3, coord2]) #bottom surface
                    pmat_faces.append([coord2, coord3, coord4]) #top surface
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])



        pmat_verts = list((pmat_xyz).reshape(1728, 3))

        print "len faces: ", len(pmat_faces)
        print "len verts: ", len(pmat_verts)
        print len(pmat_faces), len(pmat_facecolors)

        return pmat_verts, pmat_faces, pmat_facecolors




    def mesh_render_pose_bed(self, m, root_pos, pc, pc_isnew, pmat, markers, bedangle, segment_limbs = False):

        #get SMPL mesh
        smpl_verts = (m.r - m.J_transformed[0, :])+[root_pos[1]-0.286+0.15, root_pos[0]-0.286, 0.12-root_pos[2]]#*228./214.
        smpl_faces = np.array(m.f)

        print smpl_verts

        if np.sum(pmat) < 5000:
            smpl_verts = smpl_verts * 0.001


        human_mesh_vtx_parts = [smpl_verts[self.segmented_dict['l_lowerleg_idx_list'], :],
                                smpl_verts[self.segmented_dict['r_lowerleg_idx_list'], :],
                                smpl_verts[self.segmented_dict['l_upperleg_idx_list'], :],
                                smpl_verts[self.segmented_dict['r_upperleg_idx_list'], :],
                                smpl_verts[self.segmented_dict['l_forearm_idx_list'], :],
                                smpl_verts[self.segmented_dict['r_forearm_idx_list'], :],
                                smpl_verts[self.segmented_dict['l_upperarm_idx_list'], :],
                                smpl_verts[self.segmented_dict['r_upperarm_idx_list'], :],
                                smpl_verts[self.segmented_dict['head_idx_list'], :],
                                smpl_verts[self.segmented_dict['torso_idx_list'], :]]
        human_mesh_face_parts = [self.segmented_dict['l_lowerleg_face_list'],
                                 self.segmented_dict['r_lowerleg_face_list'],
                                 self.segmented_dict['l_upperleg_face_list'],
                                 self.segmented_dict['r_upperleg_face_list'],
                                 self.segmented_dict['l_forearm_face_list'],
                                 self.segmented_dict['r_forearm_face_list'],
                                 self.segmented_dict['l_upperarm_face_list'],
                                 self.segmented_dict['r_upperarm_face_list'],
                                 self.segmented_dict['head_face_list'],
                                 self.segmented_dict['torso_face_list']]

        #human_mesh_vtx_parts = [smpl_verts]
        #human_mesh_face_parts = [smpl_faces]


        tm_list = []
        for idx in range(len(human_mesh_vtx_parts)):
            tm_list.append(trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts[idx]), faces = np.array(human_mesh_face_parts[idx])))

        mesh_list = []
        for idx in range(len(tm_list)):
            if len(tm_list) == 1:
                mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[idx], material = self.human_mat, wireframe = True))
            else:
                mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[idx], material = self.mesh_parts_mat_list[idx], wireframe = True))


        #smpl_tm = trimesh.base.Trimesh(vertices=smpl_verts, faces=smpl_faces)
        #smpl_mesh = pyrender.Mesh.from_trimesh(smpl_tm, material=self.human_mat, wireframe = True)


        #get Point cloud mesh
        if pc is not None:
            pc_mesh = pyrender.Mesh.from_points(pc)
        else:
            pc_mesh = None


        #print m.r
        #print artag_r
        #create mini meshes for AR tags
        artag_meshes = []
        for marker in markers:
            if markers[2] is None:
                artag_meshes.append(None)
            elif marker is None:
                artag_meshes.append(None)
            else:
                #print marker - markers[2]
                if marker is markers[2]:
                    artag_tm = trimesh.base.Trimesh(vertices=self.artag_r+marker-markers[2], faces=self.artag_f, face_colors = self.artag_facecolors_root)
                    artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))
                else:
                    artag_tm = trimesh.base.Trimesh(vertices=self.artag_r+marker-markers[2], faces=self.artag_f, face_colors = self.artag_facecolors)
                    artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))




        pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat, bedangle)
        pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors = pmat_facecolors)
        pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth = False)


        #print "Viewing"
        if self.first_pass == True:

            for mesh_part in mesh_list:
                self.scene.add(mesh_part)

            #self.scene.add(smpl_mesh)
            self.scene.add(pc_mesh)
            self.scene.add(pmat_mesh)

            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    self.scene.add(artag_mesh)


            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
            self.first_pass = False

            #for node in self.scene.get_nodes(obj=smpl_mesh):
            #    self.human_obj_node = node

            self.node_list = []
            for mesh_part in mesh_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_list.append(node)

            for node in self.scene.get_nodes(obj=pc_mesh):
                self.point_cloud_node = node

            self.artag_nodes = []
            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    for node in self.scene.get_nodes(obj=artag_mesh):
                        self.artag_nodes.append(node)

            for node in self.scene.get_nodes(obj=pmat_mesh):
                self.pmat_node = node

        else:
            self.viewer.render_lock.acquire()

            #reset the human mesh
            for idx in range(len(mesh_list)):

                self.scene.remove_node(self.node_list[idx])
                self.scene.add(mesh_list[idx])
                for node in self.scene.get_nodes(obj=mesh_list[idx]):
                    self.node_list[idx] = node

            #self.scene.remove_node(self.human_obj_node)
            #self.scene.add(smpl_mesh)
            #for node in self.scene.get_nodes(obj=smpl_mesh):
            #    self.human_obj_node = node


            #reset the point cloud mesh
            if pc_mesh is not None:
                self.scene.remove_node(self.point_cloud_node)
                self.scene.add(pc_mesh)
                for node in self.scene.get_nodes(obj=pc_mesh):
                    self.point_cloud_node = node

            #reset the artag meshes
            for artag_node in self.artag_nodes:
                self.scene.remove_node(artag_node)
            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    self.scene.add(artag_mesh)
            self.artag_nodes = []
            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    for node in self.scene.get_nodes(obj=artag_mesh):
                        self.artag_nodes.append(node)


            #reset the pmat mesh
            self.scene.remove_node(self.pmat_node)
            self.scene.add(pmat_mesh)
            for node in self.scene.get_nodes(obj=pmat_mesh):
                self.pmat_node = node



            #print self.scene.get_nodes()
            self.viewer.render_lock.release()


        sleep(0.01)
        print "BLAH"
