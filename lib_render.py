

#import open3d as o3d
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
        pmat_colors[:, :, 3] = 0.7 #translucency

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


    def reduce_by_cam_dir(self, vertices, faces, camera_point):

        vertices = np.array(vertices)
        faces = np.array(faces)

        tri_norm = np.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :],
                            vertices[faces[:, 2], :] - vertices[faces[:, 0], :])

        tri_norm = tri_norm/np.linalg.norm(tri_norm, axis = 1)[:, None]

        tri_to_cam = camera_point - vertices[faces[:, 0], :]

        tri_to_cam = tri_to_cam/np.linalg.norm(tri_to_cam, axis = 1)[:, None]

        angle_list = tri_norm[:, 0]*tri_to_cam[:, 0] + tri_norm[:, 1]*tri_to_cam[:, 1] + tri_norm[:, 2]*tri_to_cam[:, 2]
        angle_list = np.arccos(angle_list) * 180 / np.pi


        angle_list = np.array(angle_list)
        faces = np.array(faces)
        faces_red = faces[angle_list < 90, :]

        return list(faces_red)


    def get_triangle_area_vert_weight(self, verts, faces, verts_idx_red):

        #first we need all the triangle areas
        tri_verts = verts[faces, :]
        a = np.linalg.norm(tri_verts[:,0]-tri_verts[:,1], axis = 1)
        b = np.linalg.norm(tri_verts[:,1]-tri_verts[:,2], axis = 1)
        c = np.linalg.norm(tri_verts[:,2]-tri_verts[:,0], axis = 1)
        s = (a+b+c)/2
        A = np.sqrt(s*(s-a)*(s-b)*(s-c))
        A = np.swapaxes(np.stack((A, A, A)), 0, 1) #repeat the area for each vert in the triangle
        A = A.flatten()
        faces = np.array(faces).flatten()
        i = np.argsort(faces) #sort the faces and the areas by the face idx
        faces_sorted = faces[i]
        A_sorted = A[i]
        last_face = 0
        area_minilist = []
        area_avg_list = []
        face_sort_list = [] #take the average area for all the trianges surrounding each vert
        for vtx_connect_idx in range(np.shape(faces_sorted)[0]):
            if faces_sorted[vtx_connect_idx] == last_face and vtx_connect_idx != np.shape(faces_sorted)[0]-1:
                area_minilist.append(A_sorted[vtx_connect_idx])
            elif faces_sorted[vtx_connect_idx] > last_face or vtx_connect_idx == np.shape(faces_sorted)[0]-1:
                if len(area_minilist) != 0:
                    area_avg_list.append(np.mean(area_minilist))
                else:
                    area_avg_list.append(0)
                face_sort_list.append(last_face)
                area_minilist = []
                last_face += 1
                if faces_sorted[vtx_connect_idx] == last_face:
                    area_minilist.append(A_sorted[vtx_connect_idx])
                elif faces_sorted[vtx_connect_idx] > last_face:
                    num_tack_on = np.copy(faces_sorted[vtx_connect_idx] - last_face)
                    for i in range(num_tack_on):
                        area_avg_list.append(0)
                        face_sort_list.append(last_face)
                        last_face += 1
                        if faces_sorted[vtx_connect_idx] == last_face:
                            area_minilist.append(A_sorted[vtx_connect_idx])

        area_avg = np.array(area_avg_list)
        area_avg_red = area_avg[area_avg > 0] #find out how many of the areas correspond to verts facing the camera

        #print np.sum(area_avg_red), np.sum(area_avg)

        norm_area_avg = area_avg/np.sum(area_avg_red)
        norm_area_avg = norm_area_avg*np.shape(area_avg_red) #multiply by the REDUCED num of verts
        #print norm_area_avg[0:3], np.min(norm_area_avg), np.max(norm_area_avg), np.mean(norm_area_avg), np.sum(norm_area_avg)
        #print norm_area_avg.shape, np.shape(verts_idx_red)

        norm_area_avg = norm_area_avg[verts_idx_red]
        #print norm_area_avg[0:3], np.min(norm_area_avg), np.max(norm_area_avg), np.mean(norm_area_avg), np.sum(norm_area_avg)
        return norm_area_avg


    def get_triangle_norm_to_vert(self, verts, faces, verts_idx_red):

        tri_norm = np.cross(verts[np.array(faces)[:, 1], :] - verts[np.array(faces)[:, 0], :],
                            verts[np.array(faces)[:, 2], :] - verts[np.array(faces)[:, 0], :])

        tri_norm = tri_norm/np.linalg.norm(tri_norm, axis = 1)[:, None] #but this is for every TRIANGLE. need it per vert
        tri_norm = np.stack((tri_norm, tri_norm, tri_norm))
        tri_norm = np.swapaxes(tri_norm, 0, 1)

        tri_norm = tri_norm.reshape(tri_norm.shape[0]*tri_norm.shape[1], tri_norm.shape[2])

        faces = np.array(faces).flatten()

        i = np.argsort(faces) #sort the faces and the areas by the face idx
        faces_sorted = faces[i]

        tri_norm_sorted = tri_norm[i]

        last_face = 0
        face_sort_list = [] #take the average area for all the trianges surrounding each vert
        vertnorm_minilist = []
        vertnorm_avg_list = []

        for vtx_connect_idx in range(np.shape(faces_sorted)[0]):
            if faces_sorted[vtx_connect_idx] == last_face and vtx_connect_idx != np.shape(faces_sorted)[0]-1:
                vertnorm_minilist.append(tri_norm_sorted[vtx_connect_idx])
            elif faces_sorted[vtx_connect_idx] > last_face or vtx_connect_idx == np.shape(faces_sorted)[0]-1:
                if len(vertnorm_minilist) != 0:
                    mean_vertnorm = np.mean(vertnorm_minilist, axis = 0)
                    mean_vertnorm = mean_vertnorm/np.linalg.norm(mean_vertnorm)
                    vertnorm_avg_list.append(mean_vertnorm)
                else:
                    vertnorm_avg_list.append(np.array([0.0, 0.0, 0.0]))
                face_sort_list.append(last_face)
                vertnorm_minilist = []
                last_face += 1
                if faces_sorted[vtx_connect_idx] == last_face:
                    vertnorm_minilist.append(tri_norm_sorted[vtx_connect_idx])
                elif faces_sorted[vtx_connect_idx] > last_face:
                    num_tack_on = np.copy(faces_sorted[vtx_connect_idx] - last_face)
                    for i in range(num_tack_on):
                        vertnorm_avg_list.append([0.0, 0.0, 0.0])
                        face_sort_list.append(last_face)
                        last_face += 1
                        if faces_sorted[vtx_connect_idx] == last_face:
                            vertnorm_minilist.append(tri_norm_sorted[vtx_connect_idx])


        vertnorm_avg = np.array(vertnorm_avg_list)
        vertnorm_avg_red = np.swapaxes(np.stack((vertnorm_avg[vertnorm_avg[:, 0] != 0, 0],
                                                vertnorm_avg[vertnorm_avg[:, 1] != 0, 1],
                                                vertnorm_avg[vertnorm_avg[:, 2] != 0, 2])), 0, 1)
        return vertnorm_avg_red


    def downspl_pc_get_normals(self, pc, camera_point):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        #print("Downsample the point cloud with a voxel of 0.01")
        downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=0.01)


        #o3d.visualization.draw_geometries([downpcd])

        #print("Recompute the normal of the downsampled point cloud")
        o3d.geometry.estimate_normals(
            downpcd,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,
                                                              max_nn=30))

        o3d.geometry.orient_normals_towards_camera_location(downpcd, camera_location=np.array(camera_point))

        #o3d.visualization.draw_geometries([downpcd])

        points = np.array(downpcd.points)
        normals = np.array(downpcd.normals)

        return points, normals

    def plot_mesh_norms(self, verts, verts_norm):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.normals = o3d.utility.Vector3dVector(verts_norm)

        o3d.visualization.draw_geometries([pcd])



    def eval_dist(self, pc, verts, faces, faces_red, camera_point):
        print pc.shape, verts.shape, np.shape(faces_red)

        verts_idx_red = np.unique(faces_red)

        verts_red = verts[verts_idx_red, :]

        print verts_red.shape


        print verts_idx_red.shape
        print verts_idx_red

        #downsample the point cloud and get the normals
        pc_red, pc_red_norm = self.downspl_pc_get_normals(pc, camera_point)

        #get the vert norms
        verts_norm = self.get_triangle_norm_to_vert(verts, faces, verts_idx_red)
        verts_red_norm = verts_norm[verts_idx_red, :]

        print np.shape(pc_red), np.shape(verts_red)
        print np.shape(pc_red_norm), np.shape(verts_red_norm)

        #plot the mesh with normals on the verts
        #self.plot_mesh_norms(verts_red, verts_red_norm)

        allverts = o3d.geometry.PointCloud()
        allverts.points = o3d.utility.Vector3dVector(verts_red)
        allverts.normals = o3d.utility.Vector3dVector(verts_red_norm)
        allverts.paint_uniform_color([55./256,126./256,184./256])


        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(pc_red)
        points.normals = o3d.utility.Vector3dVector(pc_red_norm)
        points.paint_uniform_color([228./256,26./256,28./256])

        o3d.visualization.draw_geometries([points, allverts])


        #get the nearest point from each pc point to some vert, regardless of the normal
        pc_to_nearest_vert_error_list = []
        for point_idx in range(pc_red.shape[0]):
            curr_point = pc_red[point_idx, :]

            all_dist = verts_red - curr_point
            all_eucl = np.linalg.norm(all_dist, axis = 1)
            curr_error = np.min(all_eucl)
            pc_to_nearest_vert_error_list.append(curr_error)
            #break
        print "average pc point to nearest vert error, regardless of normal:", np.mean(pc_to_nearest_vert_error_list)


        #get the nearest point from each pc point to some vert, considering the normal
        #angle_cutoff = 45.0
        for angle_cutoff in [90., 80., 70., 60., 50., 40., 30., 20., 10.]:
        #for angle_cutoff in [90.]:
            pc_to_nearest_vert_error_list = []
            for point_idx in range(pc_red.shape[0]):

                curr_point = pc_red[point_idx, :]
                curr_normal = pc_red_norm[point_idx, :]
                udotv = curr_normal[0]*verts_red_norm[:, 0] + \
                        curr_normal[1]*verts_red_norm[:, 1] + \
                        curr_normal[2]*verts_red_norm[:, 2]

                theta = np.arccos(udotv)*180/np.pi
                cutoff_mult = np.copy(theta)
                cutoff_add = np.copy(theta)
                cutoff_mult[cutoff_mult < angle_cutoff] = 1
                cutoff_mult[cutoff_mult > angle_cutoff] = 0
                cutoff_add[cutoff_add < angle_cutoff] = 0
                cutoff_add[cutoff_add > angle_cutoff] = 5
                verts_red_facing = verts_red*(np.expand_dims(cutoff_mult, axis = 1))
                verts_red_facing = verts_red_facing+(np.expand_dims(cutoff_add, axis = 1))

                all_dist = verts_red_facing - curr_point
                #all_dist = verts_red - curr_point
                all_eucl = np.linalg.norm(all_dist, axis = 1)

                curr_error = np.min(all_eucl)
                pc_to_nearest_vert_error_list.append(curr_error)
                #break

            #print "average pc point to nearest vert error, considering the normal:", np.mean(pc_to_nearest_vert_error_list)," cutoff: ",angle_cutoff
            print np.mean(pc_to_nearest_vert_error_list)


        #get the nearest point from each vert to some pc point, regardless of the normal
        vert_to_nearest_point_error_list = []
        for vert_idx in range(verts_red.shape[0]):
            curr_vtx = verts_red[vert_idx, :]
            all_dist = pc_red - curr_vtx
            all_eucl = np.linalg.norm(all_dist, axis = 1)
            curr_error = np.min(all_eucl)
            vert_to_nearest_point_error_list.append(curr_error)
            #break
        #print np.mean(vert_to_nearest_point_error_list)

        #normalize by the average area of triangles around each point. the verts are much less spatially distributed well
        #than the points in the point cloud.
        norm_area_avg = self.get_triangle_area_vert_weight(verts, faces_red, verts_idx_red)
        norm_vert_to_nearest_point_error = np.array(vert_to_nearest_point_error_list)*norm_area_avg
        print "average vert to nearest pc point error, regardless of normal:", np.mean(norm_vert_to_nearest_point_error)


        #get the nearest point from each vert to some pc point, considering the normal
        for angle_cutoff in [90., 80., 70., 60., 50., 40., 30., 20., 10.]:
        #for angle_cutoff in [90.]:
            vert_to_nearest_point_error_list = []
            for vert_idx in range(verts_red.shape[0]):
                curr_vtx = verts_red[vert_idx, :]
                curr_normal = verts_red_norm[vert_idx, :]
                udotv = curr_normal[0]*pc_red_norm[:, 0] + \
                        curr_normal[1]*pc_red_norm[:, 1] + \
                        curr_normal[2]*pc_red_norm[:, 2]
                theta = np.arccos(udotv)*180/np.pi
                cutoff_mult = np.copy(theta)
                cutoff_add = np.copy(theta)

                cutoff_mult[cutoff_mult < angle_cutoff] = 1
                cutoff_mult[cutoff_mult > angle_cutoff] = 0
                cutoff_add[cutoff_add < angle_cutoff] = 0
                cutoff_add[cutoff_add > angle_cutoff] = 5.0
                pc_red_facing = pc_red*(np.expand_dims(cutoff_mult, axis = 1))
                pc_red_facing = pc_red_facing+(np.expand_dims(cutoff_add, axis = 1))

                all_dist = pc_red_facing - curr_vtx
                all_eucl = np.linalg.norm(all_dist, axis = 1)

                curr_error = np.min(all_eucl)
                #if curr_error < 0.5:
                vert_to_nearest_point_error_list.append(curr_error)

                '''
                #else:
                #print curr_error
                print curr_vtx
                print curr_normal
                print norm_area_avg[vert_idx]
    
                allverts = o3d.geometry.PointCloud()
                allverts.points = o3d.utility.Vector3dVector(verts_red)
                #allverts.normals = o3d.utility.Vector3dVector(verts_red_norm)
                allverts.paint_uniform_color([0.1, 0.9, 0.1])
    
                vert = o3d.geometry.PointCloud()
                vert.points = o3d.utility.Vector3dVector(np.expand_dims(np.array(curr_vtx), axis = 0))
                vert.normals = o3d.utility.Vector3dVector(np.expand_dims(curr_normal, axis = 0))
                vert.paint_uniform_color([0.1, 0.1, 0.9])
    
                points = o3d.geometry.PointCloud()
                points.points = o3d.utility.Vector3dVector(pc_red_facing)
                #points.normals = o3d.utility.Vector3dVector(pc_red_norm)
                points.paint_uniform_color([0.9, 0.1, 0.1])
    
                o3d.visualization.draw_geometries([points, vert, allverts])
                '''

                #break
            #normalize by the average area of triangles around each point. the verts are much less spatially distributed well
            #than the points in the point cloud.

            #print np.mean(vert_to_nearest_point_error_list), 'mean'
            norm_area_avg = self.get_triangle_area_vert_weight(verts, faces_red, verts_idx_red)

            #print np.shape(vert_to_nearest_point_error_list), np.shape(norm_area_avg)

            norm_vert_to_nearest_point_error = np.array(vert_to_nearest_point_error_list)*norm_area_avg
            #print "average vert to nearest pc point error, considering the normal:", np.mean(norm_vert_to_nearest_point_error)," cutoff: ",angle_cutoff
            print np.mean(norm_vert_to_nearest_point_error)


        print "DONE CALC"

    def mesh_render_pose_bed_orig(self, m, root_pos, pc, pc_isnew, pmat, markers, bedangle):

        # get SMPL mesh
        smpl_verts = (m.r - m.J_transformed[0, :]) + [root_pos[1] - 0.286 + 0.15, root_pos[0] - 0.286,
                                                      0.12 - root_pos[2]]  # *228./214.

        print
        smpl_verts

        if np.sum(pmat) < 5000:
            smpl_verts = smpl_verts * 0.001

        smpl_tm = trimesh.base.Trimesh(vertices=smpl_verts, faces=m.f)
        smpl_mesh = pyrender.Mesh.from_trimesh(smpl_tm, material=self.human_mat, wireframe=True)

        # get Point cloud mesh
        if pc is not None:
            pc_mesh = pyrender.Mesh.from_points(pc)
        else:
            pc_mesh = None

        # print m.r
        # print artag_r
        # create mini meshes for AR tags
        artag_meshes = []
        for marker in markers:
            if markers[2] is None:
                artag_meshes.append(None)
            elif marker is None:
                artag_meshes.append(None)
            else:
                print
                marker - markers[2]
                if marker is markers[2]:
                    artag_tm = trimesh.base.Trimesh(vertices=self.artag_r + marker - markers[2], faces=self.artag_f,
                                                    face_colors=self.artag_facecolors_root)
                    artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth=False))
                else:
                    artag_tm = trimesh.base.Trimesh(vertices=self.artag_r + marker - markers[2], faces=self.artag_f,
                                                    face_colors=self.artag_facecolors)
                    artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth=False))

        pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat, bedangle)
        pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors=pmat_facecolors)
        pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth=False)

        # print "Viewing"
        if self.first_pass == True:

            self.scene.add(smpl_mesh)
            self.scene.add(pc_mesh)
            self.scene.add(pmat_mesh)

            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    self.scene.add(artag_mesh)

            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
            self.first_pass = False

            for node in self.scene.get_nodes(obj=smpl_mesh):
                self.human_obj_node = node

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

            # reset the human mesh
            self.scene.remove_node(self.human_obj_node)
            self.scene.add(smpl_mesh)
            for node in self.scene.get_nodes(obj=smpl_mesh):
                self.human_obj_node = node

            # reset the point cloud mesh
            if pc_mesh is not None:
                self.scene.remove_node(self.point_cloud_node)
                self.scene.add(pc_mesh)
                for node in self.scene.get_nodes(obj=pc_mesh):
                    self.point_cloud_node = node

            # reset the artag meshes
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

            # reset the pmat mesh
            self.scene.remove_node(self.pmat_node)
            self.scene.add(pmat_mesh)
            for node in self.scene.get_nodes(obj=pmat_mesh):
                self.pmat_node = node

            # print self.scene.get_nodes()
            self.viewer.render_lock.release()

        sleep(0.01)
        print
        "BLAH"

    def mesh_render_pose_bed(self, smpl_verts, smpl_faces, pc, pc_isnew, pmat, markers, bedangle, segment_limbs = False):

        camera_point = [1.09898028, 0.46441343, -1.53]


        if np.sum(pmat) < 5000:
            smpl_verts = smpl_verts * 0.001

        if segment_limbs == True:
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
        else:
            human_mesh_vtx_parts = [smpl_verts]
            human_mesh_face_parts = [smpl_faces]

        human_mesh_face_parts_red = []

        #only use the vertices that are facing the camera
        for part_idx in range(len(human_mesh_vtx_parts)):
            human_mesh_face_parts_red.append(self.reduce_by_cam_dir(human_mesh_vtx_parts[part_idx], human_mesh_face_parts[part_idx], camera_point))



        self.eval_dist(pc, human_mesh_vtx_parts[0], human_mesh_face_parts[0], human_mesh_face_parts_red[0], camera_point)


        tm_list = []
        for idx in range(len(human_mesh_vtx_parts)):
            tm_list.append(trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts[idx]), faces = np.array(human_mesh_face_parts_red[idx])))

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

            self.scene.add(pc_mesh)
            self.scene.add(pmat_mesh)

            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    self.scene.add(artag_mesh)


            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
            self.first_pass = False

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


        sleep(100.01)
        print "BLAH"
