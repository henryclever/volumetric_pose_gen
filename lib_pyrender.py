

try:
    import open3d as o3d
except:
    print "COULD NOT IMPORT 03D"
import trimesh
import pyrender
import pyglet

import numpy as np
import random
import copy
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




class pyRenderMesh():
    def __init__(self):

        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.scene = pyrender.Scene()

        #self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 1.0 ,0.0])
        self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 0.0 ,0.0])
        self.human_arm_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.8 ,1.0])
        self.human_mat_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 0.3 ,0.5])
        self.human_bed_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.7, 0.7, 0.2 ,0.5])
        self.human_mat_D = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 1.0], alphaMode="BLEND")

        mesh_color_mult = 0.25

        self.mesh_parts_mat_list = [
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 166. / 255., mesh_color_mult * 206. / 255., mesh_color_mult * 227. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 31. / 255., mesh_color_mult * 120. / 255., mesh_color_mult * 180. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 251. / 255., mesh_color_mult * 154. / 255., mesh_color_mult * 153. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 227. / 255., mesh_color_mult * 26. / 255., mesh_color_mult * 28. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 178. / 255., mesh_color_mult * 223. / 255., mesh_color_mult * 138. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 51. / 255., mesh_color_mult * 160. / 255., mesh_color_mult * 44. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 253. / 255., mesh_color_mult * 191. / 255., mesh_color_mult * 111. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 255. / 255., mesh_color_mult * 127. / 255., mesh_color_mult * 0. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 202. / 255., mesh_color_mult * 178. / 255., mesh_color_mult * 214. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 106. / 255., mesh_color_mult * 61. / 255., mesh_color_mult * 154. / 255., 0.0])]

        self.artag_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 1.0, 0.3, 0.5])
        self.artag_mat_other = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 0.0])
        self.artag_r = np.array([[-0.055, -0.055, 0.0], [-0.055, 0.055, 0.0], [0.055, -0.055, 0.0], [0.055, 0.055, 0.0]])
        self.artag_f = np.array([[0, 1, 3], [3, 1, 0], [0, 2, 3], [3, 2, 0], [1, 3, 2]])
        self.artag_facecolors_root = np.array([[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0]])
        self.artag_facecolors = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],])


        self.pic_num = 0


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

                pmat_xyz[j, i, 1] = i * 0.0286 * 1.02 #1.0926 - 0.02
                pmat_xyz[j, i, 0] = ((63 - j) * 0.0286) * 1.04#1.1406 + 0.05 #only adjusts pmat NOT the SMPL person
                pmat_xyz[j, i, 2] = 0.12 + 0.075
                #if j > 23:
                #    pmat_xyz[j, i, 0] = ((64 - j) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(angle)))*1.04 + 0.15#1.1406 + 0.05
                #    pmat_xyz[j, i, 2] = 0.12 + 0.075
                #    # print marker.pose.position.x, 'x'
                #else:

                #    pmat_xyz[j, i, 0] = ((41) * 0.0286 + (23 - j) * 0.0286 * np.cos(np.deg2rad(angle)) \
                #                        - (0.0286 * 3 * np.sin(np.deg2rad(angle))) * 0.85)*1.04 + 0.15#1.1406 + 0.05
                #    pmat_xyz[j, i, 2] = -((23 - j) * 0.0286 * np.sin(np.deg2rad(angle))) * 0.85 + 0.12 + 0.075
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





    def get_custom_pmat(self, pmat, viz_type, angle = 60.0, coloring_type = 'jet'):

        pmat_reshaped = pmat.reshape(64, 27)


        j_max = 64
        pmat_reshaped = pmat_reshaped[:j_max, :]

        if coloring_type == 'viridis':
            pmat_colors = cm.viridis((pmat_reshaped)/100)
            pmat_colors[:, :, 3] = 0.7 #translucency
        elif coloring_type == 'plasma':
            pmat_colors = cm.plasma((100-pmat_reshaped)/100)
            pmat_colors[:, :, 3] = 0.7 #translucency
        elif coloring_type == 'magma':
            pmat_colors = cm.inferno((100-pmat_reshaped)/100)
            pmat_colors[:, :, 3] = 0.5 #translucency
        elif coloring_type == 'terrain':
            pmat_colors = cm.terrain((100-pmat_reshaped)/100)
            pmat_colors[:, :, 3] = 0.7 #translucency
        else:
            pmat_colors = cm.jet(pmat_reshaped/100)
            pmat_colors[:, :, 3] = 0.6 #translucency


        pmat_colors = pmat_colors[:j_max, :, :]

        pmat_xyz = np.zeros((j_max, 27, 3))
        pmat_faces = []
        pmat_facecolors = []

        if viz_type == "arm_penetration":
            i_lim = [10, 26]
            j_lim = [0, 31]
        elif viz_type == "leg_correction":
            i_lim = [8, 22]
            j_lim = [37, 56]
        else:
            i_lim = [0, 26]
            j_lim = [0, 63]

        for j in range(j_max):
            for i in range(27):

                pmat_xyz[j, i, 1] = i * 0.0286 * 1.06 - 0.06 + self.pmat_shifter[self.pmat_shifter_ct][0]#1.0926 - 0.02
                if j > 23:
                    pmat_xyz[j, i, 0] = ((64 - j) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(angle)))*1.0 + 0.14 \
                                        + self.pmat_shifter[self.pmat_shifter_ct][1]#1.1406 + 0.05
                    pmat_xyz[j, i, 2] = 0.10 + 0.075
                    # print marker.pose.position.x, 'x'
                else:

                    pmat_xyz[j, i, 0] = ((41) * 0.0286 + (23 - j) * 0.0286 * np.cos(np.deg2rad(angle)) \
                                        - (0.0286 * 3 * np.sin(np.deg2rad(angle))) * 0.85)*1.0 + 0.14 \
                                        + self.pmat_shifter[self.pmat_shifter_ct][1]  #1.1406 + 0.05
                    #pmat_xyz[j, i, 2] = -((23 - j) * 0.0286 * np.sin(np.deg2rad(angle))) * 0.85 + 0.12 + 0.075
                    pmat_xyz[j, i, 2] = -((23 - j) * 0.0286 * np.sin(np.deg2rad(angle))) * 0.85 + 0.10 + 0.075
                    # print j, marker.pose.position.z, marker.pose.position.y, 'head'


                #if j < 31 and 10 < i < 26:
                if j_lim[0] <= j < j_lim[1] and i_lim[0] <= i < i_lim[1]:
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


        pmat_verts_orig = (pmat_xyz).reshape(27*j_max, 3)


        #print pmat_verts_orig[0:5]

        pmat_verts_shifted = np.copy((pmat_xyz).reshape(27*j_max, 3))
        pmat_verts_shifted[:, 0] += 0.001
        pmat_verts_shifted[:, 1] += 0.001


        #print pmat_verts_orig[0:5]

        pmat_verts_grid = np.concatenate((pmat_verts_orig, pmat_verts_shifted), axis = 0)

        print np.shape(pmat_verts_grid) #3456

        #print pmat_verts_grid

        print pmat_verts_grid[0:5]
        print pmat_verts_grid[1728:1733]

        pmat_faces_grid = []
        pmat_facecolors_grid = []

        for j in range(j_lim[0], j_lim[1]):
            for i in range(i_lim[0], i_lim[1]):
                coord1 = j * 27 + i
                coord2 = j * 27 + i + 1
                coord3 = j * 27 + i + 1728
                coord4 = j * 27 + i + 1729

                pmat_faces_grid.append([coord1, coord2, coord3]) #bottom surface
                pmat_faces_grid.append([coord1, coord3, coord2]) #top surface
                pmat_faces_grid.append([coord4, coord3, coord2]) #bottom surface
                pmat_faces_grid.append([coord2, coord3, coord4]) #top

                coord1 = j * 27 + i + 1
                coord2 = (j + 1) * 27 + i + 1
                coord3 = j * 27 + i + 1728 + 1
                coord4 = (j + 1) * 27 + i + 1728 + 1

                pmat_faces_grid.append([coord1, coord2, coord3]) #bottom surface
                pmat_faces_grid.append([coord1, coord3, coord2]) #top surface
                pmat_faces_grid.append([coord4, coord3, coord2]) #bottom surface
                pmat_faces_grid.append([coord2, coord3, coord4]) #top

                pmat_colors[0, 0, 0:3] *= 0.0
                for k in range(8):
                    pmat_facecolors_grid.append(pmat_colors[0, 0, :])



        pmat_verts_grid = list(pmat_verts_grid)


        pmat_verts = list((pmat_xyz).reshape(27*j_max, 3))

        print "len faces: ", len(pmat_faces)
        print "len verts: ", len(pmat_verts)
        print len(pmat_faces), len(pmat_facecolors)

        print "len faces grid: ", len(pmat_faces_grid)
        print "len verts grid: ", len(pmat_verts_grid)


        #pmat_faces = pmat_faces[]



        return pmat_verts, pmat_faces, pmat_facecolors, pmat_verts_grid, pmat_faces_grid, pmat_facecolors_grid


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

        print np.shape(verts_idx_red), np.min(verts_idx_red), np.max(verts_idx_red)
        print np.shape(norm_area_avg), np.min(norm_area_avg), np.max(norm_area_avg)

        print verts_idx_red, np.shape(faces_sorted)

        try:
            norm_area_avg = norm_area_avg[verts_idx_red]
        except:
            norm_area_avg = norm_area_avg[verts_idx_red[:-1]]

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


    def get_human_mesh_parts(self, smpl_verts, smpl_faces, viz_type = None, segment_limbs = False):

        if segment_limbs == True:
            if viz_type == 'arm_penetration':
                segmented_dict = load_pickle('segmented_mesh_idx_faces_larm.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['l_arm_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['l_arm_face_list']]
            elif viz_type == 'leg_correction':
                segmented_dict = load_pickle('segmented_mesh_idx_faces_rleg.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['r_leg_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['r_leg_face_list']]
            else:
                segmented_dict = load_pickle('segmented_mesh_idx_faces.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['l_lowerleg_idx_list'], :],
                                        smpl_verts[segmented_dict['r_lowerleg_idx_list'], :],
                                        smpl_verts[segmented_dict['l_upperleg_idx_list'], :],
                                        smpl_verts[segmented_dict['r_upperleg_idx_list'], :],
                                        smpl_verts[segmented_dict['l_forearm_idx_list'], :],
                                        smpl_verts[segmented_dict['r_forearm_idx_list'], :],
                                        smpl_verts[segmented_dict['l_upperarm_idx_list'], :],
                                        smpl_verts[segmented_dict['r_upperarm_idx_list'], :],
                                        smpl_verts[segmented_dict['head_idx_list'], :],
                                        smpl_verts[segmented_dict['torso_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['l_lowerleg_face_list'],
                                         segmented_dict['r_lowerleg_face_list'],
                                         segmented_dict['l_upperleg_face_list'],
                                         segmented_dict['r_upperleg_face_list'],
                                         segmented_dict['l_forearm_face_list'],
                                         segmented_dict['r_forearm_face_list'],
                                         segmented_dict['l_upperarm_face_list'],
                                         segmented_dict['r_upperarm_face_list'],
                                         segmented_dict['head_face_list'],
                                         segmented_dict['torso_face_list']]
        else:
            human_mesh_vtx_parts = [smpl_verts]
            human_mesh_face_parts = [smpl_faces]

        return human_mesh_vtx_parts, human_mesh_face_parts


    def render_only_human_gt(self, m):

        bed1_verts = np.array([[-1.3, -1.35, -3.0], [-0.9, -0.75, -4.0], [1.3, -1.35, -3.0], [0.9, -0.75, -4.0],
                               [-1.2, 0.05, -4.0], [-1.2, 0.0, -4.0], [1.2, 0.05, -4.0], [1.2, 0.0, -4.0],
                               [-1.3, -1.35, -3.0], [-1.3, -1.45, -3.0], [1.3, -1.35, -3.0], [1.3, -1.45, -3.0],
                               [-1.2, 0.05, -4.0], [-1.0, 1.0, -4.0], [1.2, 0.05, -4.0], [1.0, 1.0, -4.0],
                               ])
        bed1_faces = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 3], [1, 3, 2],
                               [4, 5, 6], [4, 6, 5], [5, 6, 7], [5, 7, 6],
                               [8, 9, 10], [8, 10, 9], [9, 10, 11], [9, 11, 10],
                               [12, 13, 14], [12, 14, 13], [13, 14, 15], [13, 15, 14],
                               ])
        bed1_facecolors = []
        for i in range(bed1_faces.shape[0]):
            if 4 <= i < 12:
                bed1_facecolors.append([0.8, 0.8, 0.2])
            else:
                bed1_facecolors.append([1.0, 1.0, 0.8])


        smpl_verts = (m.r - m.J_transformed[0, :])


        #smpl_verts = np.concatenate((smpl_verts[:, 1:2] - 1.5, smpl_verts[:, 0:1], -smpl_verts[:, 2:3]), axis = 1)
        smpl_verts = np.concatenate((smpl_verts[:, 0:1] - 1.5, smpl_verts[:, 1:2], smpl_verts[:, 2:3]), axis = 1)
        #smpl_verts = np.concatenate((smpl_verts[:, 1:2], smpl_verts[:, 0:1], -smpl_verts[:, 2:3]), axis = 1)

        smpl_faces = np.array(m.f)
        #smpl_faces = np.concatenate((smpl_faces[:, 0:1],smpl_faces[:, 2:3],smpl_faces[:, 1:2]), axis = 1)

        smpl_verts2 = np.concatenate((smpl_verts[:, 1:2], smpl_verts[:, 2:3] - 2.0, smpl_verts[:, 0:1]), axis = 1)
        smpl_verts3 = np.concatenate((smpl_verts[:, 1:2], smpl_verts[:, 2:3] -2.4, smpl_verts[:, 0:1]), axis = 1)
        laying_rot_M = np.array([[1., 0., 0.], [0., 0.866025, -0.5], [0., 0.5, 0.866025]])
        laying_rot_M2 = np.array([[1., 0., 0.], [0., 0.34202, -0.93969], [0., 0.93969, 0.34202]])
        for i in range(smpl_verts2.shape[0]):
            smpl_verts2[i, :] = np.matmul(laying_rot_M, smpl_verts2[i, :])
            smpl_verts3[i, :] = np.matmul(laying_rot_M2, smpl_verts3[i, :])

            #break



        #smpl_verts2 = np.concatenate((-smpl_verts[:, 2:3] + 1.5, smpl_verts[:, 1:2], smpl_verts[:, 0:1]), axis = 1)
        #smpl_verts3 = np.concatenate((smpl_verts[:, 2:3] - 1.5, smpl_verts[:, 1:2], -smpl_verts[:, 0:1]), axis = 1)
        #print smpl_verts2.shape


        #tm = trimesh.base.Trimesh(vertices=smpl_verts, faces=smpl_faces)
        #mesh = pyrender.Mesh.from_trimesh(tm, material=self.human_mat_for_study, wireframe=False)

        tm2 = trimesh.base.Trimesh(vertices=smpl_verts2, faces=smpl_faces)
        mesh2 = pyrender.Mesh.from_trimesh(tm2, material=self.human_mat_for_study, wireframe=False)

        tm3 = trimesh.base.Trimesh(vertices=smpl_verts3, faces=smpl_faces)
        mesh3 = pyrender.Mesh.from_trimesh(tm3, material=self.human_mat_for_study, wireframe=False)

        tm_bed1 = trimesh.base.Trimesh(vertices=bed1_verts, faces=bed1_faces, face_colors = np.array(bed1_facecolors))
        mesh_bed1 = pyrender.Mesh.from_trimesh(tm_bed1, material=self.human_bed_for_study, wireframe=False, smooth=False)

        fig = plt.figure()
        #plt.plot(np.arange(0, 400), np.arange(0, 400)*.5 + 800)


        if self.first_pass == True:
            self.scene.add(mesh_bed1)
            #self.scene.add(mesh)
            self.scene.add(mesh2)
            self.scene.add(mesh3)


            self.first_pass = False

            self.node_list_bed1 = []
            for node in self.scene.get_nodes(obj=mesh_bed1):
                self.node_list_bed1.append(node)

            #self.node_list = []
            #for node in self.scene.get_nodes(obj=mesh):
            #    self.node_list.append(node)

            self.node_list_2 = []
            for node in self.scene.get_nodes(obj=mesh2):
                self.node_list_2.append(node)


            self.node_list_3 = []
            for node in self.scene.get_nodes(obj=mesh3):
                self.node_list_3.append(node)

            camera_pose = np.eye(4)
            # camera_pose[0,0] = -1.0
            # camera_pose[1,1] = -1.0
            camera_pose[0, 3] = 0.0  # -1.0
            camera_pose[1, 3] = 0.0  # -1.0
            camera_pose[2, 3] = 4.0

            # self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True,
            #                              lighting_intensity=10.,
            #                              point_size=5, run_in_thread=True, viewport_size=(1000, 1000))
            # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            camera = pyrender.OrthographicCamera(xmag=2.0, ymag=2.0)

            self.scene.add(camera, pose=camera_pose)
            light = pyrender.SpotLight(color=np.ones(3), intensity=200.0, innerConeAngle=np.pi / 10.0,
                                       outerConeAngle=np.pi / 2.0)
            light_pose = np.copy(camera_pose)
            #light_pose[1, 3] = 2.0
            light_pose[0, 3] = -3.0
            light_pose[2, 3] = 4.0

            light_pose2 = np.copy(camera_pose)
            light_pose2[0, 3] = 2.0
            light_pose2[1, 3] = 2.0
            light_pose2[2, 3] = 2.0
            # light_pose[1, ]

            self.scene.add(light, pose=light_pose)
            self.scene.add(light, pose=light_pose2)

        else:
            #self.viewer.render_lock.acquire()

            # reset the human mesh
            self.scene.remove_node(self.node_list_bed1[0])
            self.scene.add(mesh_bed1)
            for node in self.scene.get_nodes(obj=mesh_bed1):
                self.node_list_bed1[0] = node

            # reset the human mesh
            #self.scene.remove_node(self.node_list[0])
            #self.scene.add(mesh)
            #for node in self.scene.get_nodes(obj=mesh):
            #    self.node_list[0] = node


            self.scene.remove_node(self.node_list_2[0])
            self.scene.add(mesh2)
            for node in self.scene.get_nodes(obj=mesh2):
                self.node_list_2[0] = node

            self.scene.remove_node(self.node_list_3[0])
            self.scene.add(mesh3)
            for node in self.scene.get_nodes(obj=mesh3):
                self.node_list_3[0] = node



            #self.viewer.render_lock.release()



        r = pyrender.OffscreenRenderer(2000, 2000)
        # r.render(self.scene)
        color, depth = r.render(self.scene)
        # plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(color)
        # plt.subplot(1, 2, 2)
        # plt.axis('off')
        # plt.imshow(depth, cmap=plt.cm.gray_r) >> > plt.show()

        fig.set_size_inches(15., 15.)
        fig.tight_layout()
        save_name = 'm_hbh_'+'{:04}'.format(self.pic_num)
        fig.savefig('/home/henry/Pictures/CVPR2020_study/'+save_name+'.png', dpi=300)

        #plt.savefig('test2png.png', dpi=100)

        self.pic_num += 1
        #plt.show()
        if self.pic_num == 60:
            print "DONE"
            time.sleep(1000000)
        print "got here"

    def render_mesh_bed_special(self, smpl_verts, smpl_faces, bedangle,
                                    pmats = None, viz_type = None, segment_limbs = False):
        self.pmat_shifter = [[-0.6, 0.0], [0.0, 0.0], [0.0, -0.8], [0.7, 0.3], [0.7, -0.3], [1.3, 0.0], [1.3, -0.8], ]
        self.pmat_shifter_ct = 0

        human_mesh_vtx_parts_gt, human_mesh_face_parts = self.get_human_mesh_parts(smpl_verts[0], smpl_faces, viz_type, segment_limbs)
        human_mesh_vtx_parts_n1, human_mesh_face_parts = self.get_human_mesh_parts(smpl_verts[1], smpl_faces, viz_type, segment_limbs)
        human_mesh_vtx_parts_n2, human_mesh_face_parts = self.get_human_mesh_parts(smpl_verts[2], smpl_faces, viz_type, segment_limbs)

        human_mesh_vtx_parts_gt2 = [np.array(human_mesh_vtx_parts_gt[0])]
        human_mesh_vtx_parts_n2[0] = np.array(human_mesh_vtx_parts_n2[0])
        human_mesh_vtx_parts_n2[0][:, 1] += 1.3
        human_mesh_vtx_parts_gt2[0][:, 1] += 1.3



        tm_list_gt = []
        for idx in range(len(human_mesh_vtx_parts_gt)):
            if viz_type == "leg_correction":
                human_mesh_facecolor_parts = []
                for i in range(np.shape(human_mesh_face_parts[idx])[0]):
                    human_mesh_facecolor_parts.append([0.2, 0.9, 0.2, 0.8])
                tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts_gt[idx]),
                                               faces = np.array(human_mesh_face_parts[idx]),
                                               face_colors=np.array(human_mesh_facecolor_parts))
                tm_curr2 = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts_gt2[idx]),
                                               faces = np.array(human_mesh_face_parts[idx]),
                                               face_colors=np.array(human_mesh_facecolor_parts))
                tm_list_gt.append(tm_curr)
                tm_list_gt.append(tm_curr2)

        tm_list_n1 = []
        for idx in range(len(human_mesh_vtx_parts_n1)):
            if viz_type == "leg_correction":
                human_mesh_facecolor_parts = []
                for i in range(np.shape(human_mesh_face_parts[idx])[0]):
                    human_mesh_facecolor_parts.append([0.0, 0.0, 0.0, 0.9])
                tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts_n1[idx]),
                                               faces = np.array(human_mesh_face_parts[idx]),
                                               face_colors=np.array(human_mesh_facecolor_parts))
            elif viz_type == "arm_penetration":
                tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts_n1[idx]),
                                               faces = np.array(human_mesh_face_parts[idx]))

            tm_list_n1.append(tm_curr)


        tm_list_n2 = []
        for idx in range(len(human_mesh_vtx_parts_n2)):
            if viz_type == "leg_correction":
                human_mesh_facecolor_parts = []
                for i in range(np.shape(human_mesh_face_parts[idx])[0]):
                    human_mesh_facecolor_parts.append([0.0, 0.0, 0.0, 0.9])
                tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts_n2[idx]),
                                               faces = np.array(human_mesh_face_parts[idx]),
                                               face_colors=np.array(human_mesh_facecolor_parts))
                tm_list_n2.append(tm_curr)




        mesh_n1_list = []
        for idx in range(len(tm_list_n1)):
            if len(tm_list_n1) == 1:
                if viz_type == "arm_penetration":
                    mesh_n1_list.append(pyrender.Mesh.from_trimesh(tm_list_n1[idx], material = self.human_arm_mat, wireframe = False))
                elif viz_type == "leg_correction":
                    mesh_n1_list.append(pyrender.Mesh.from_trimesh(tm_list_n1[idx], smooth = False, wireframe = True))
            else:
                mesh_n1_list.append(pyrender.Mesh.from_trimesh(tm_list_n1[idx], material = self.mesh_parts_mat_list[idx], wireframe = True))


        mesh_gt_list = []
        for idx in range(len(tm_list_gt)):
            if viz_type == "leg_correction":
                mesh_gt_list.append(pyrender.Mesh.from_trimesh(tm_list_gt[idx], smooth = False, wireframe = False))
        mesh_n2_list = []
        for idx in range(len(tm_list_n2)):
            if viz_type == "leg_correction":
                mesh_n2_list.append(pyrender.Mesh.from_trimesh(tm_list_n2[idx], smooth = False, wireframe = True))



        pmat_meshes = []
        pmat_mesh_grids = []
        ct = 0
        if pmats is not None:
            for pmat in pmats:
                if ct == 0 or ct == 3:
                    coloring_type = 'jet'
                else:
                    coloring_type = 'viridis' #terrain and magma best so far
                ct += 1

                pmat_verts, pmat_faces, pmat_facecolors, \
                pmat_verts_grid, pmat_faces_grid, pmat_facecolors_grid = self.get_custom_pmat(pmat, viz_type, bedangle, coloring_type)


                pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors = pmat_facecolors)
                pmat_meshes.append(pyrender.Mesh.from_trimesh(pmat_tm, smooth = False, wireframe = False)) #could add wireframe here

                for i in range(len(pmat_facecolors)):
                    pmat_facecolors[i][0:3] *= 0.0

                pmat_tm_grid = trimesh.base.Trimesh(vertices=pmat_verts_grid, faces=pmat_faces_grid, face_colors = pmat_facecolors_grid)
                pmat_mesh_grids.append(pyrender.Mesh.from_trimesh(pmat_tm_grid, smooth = False, wireframe = True)) #could add wireframe here

                self.pmat_shifter_ct += 1
        else:
            pmat_meshes.append(None)
            pmat_mesh_grids.append(None)


        #print "Viewing"
        if self.first_pass == True:

            for mesh_part in mesh_n1_list:
                self.scene.add(mesh_part)
            for mesh_part in mesh_n2_list:
                self.scene.add(mesh_part)
            for mesh_part in mesh_gt_list:
                self.scene.add(mesh_part)
            if pmat_meshes[0] is not None:
                for item in pmat_meshes:
                    self.scene.add(item)
            if pmat_mesh_grids[0] is not None:
                for item in pmat_mesh_grids:
                    self.scene.add(item)

            lighting_intensity = 10.


            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                                          point_size=5, run_in_thread=True, viewport_size=(1400, 1400))



            self.first_pass = False

            self.node_n1_list = []
            for mesh_part in mesh_n1_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_n1_list.append(node)

            self.node_n2_list = []
            for mesh_part in mesh_n2_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_n2_list.append(node)

            self.node_gt_list = []
            for mesh_part in mesh_gt_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_gt_list.append(node)

            self.pmat_nodes = []
            if pmat_meshes[0] is not None:
                for item in pmat_meshes:
                    for node in self.scene.get_nodes(obj=item):
                        self.pmat_nodes.append(node)
            self.pmat_grid_nodes = []
            if pmat_mesh_grids[0] is not None:
                for item in pmat_mesh_grids:
                    for node in self.scene.get_nodes(obj=item):
                        self.pmat_grid_nodes.append(node)


        else:
            self.viewer.render_lock.acquire()

            #reset the human mesh
            for idx in range(len(mesh_n1_list)):
                self.scene.remove_node(self.node_n1_list[idx])
                self.scene.add(mesh_n1_list[idx])
                for node in self.scene.get_nodes(obj=mesh_n1_list[idx]):
                    self.node_n1_list[idx] = node

            #reset the human mesh
            for idx in range(len(mesh_n2_list)):
                self.scene.remove_node(self.node_n2_list[idx])
                self.scene.add(mesh_n2_list[idx])
                for node in self.scene.get_nodes(obj=mesh_n2_list[idx]):
                    self.node_n2_list[idx] = node

            #reset the human mesh
            for idx in range(len(mesh_gt_list)):
                self.scene.remove_node(self.node_gt_list[idx])
                self.scene.add(mesh_gt_list[idx])
                for node in self.scene.get_nodes(obj=mesh_gt_list[idx]):
                    self.node_gt_list[idx] = node



            #reset the pmat mesh
            if pmat_meshes[0] is not None:
                for node in self.pmat_nodes:
                    self.scene.remove_node(node)
                self.pmat_nodes = []
                for item in pmat_meshes:
                    self.scene.add(item)
                    for node in self.scene.get_nodes(obj=item):
                        self.pmat_nodes.append(node)

            if pmat_mesh_grid is not None:
                for node in self.pmat_grid_nodes:
                    self.scene.remove_node(node)
                self.pmat_grid_nodes = []
                for item in pmat_mesh_grids:
                    self.scene.add(item)
                    for node in self.scene.get_nodes(obj=item):
                        self.pmat_grid_nodes.append(node)



            #print self.scene.get_nodes()
            self.viewer.render_lock.release()



    def render_mesh_pc_bed_pyrender_everything(self, smpl_verts, smpl_faces, camera_point, bedangle,
                                    pc = None, pmat = None, smpl_render_points = False, markers = None,
                                    dropout_variance=None):

        if pc is not None:
            print np.shape(pc), 'shape pc'
            pc[:, 0] = pc[:, 0] - 0.17 - 0.036608
            pc[:, 1] = pc[:, 1] + 0.04

        #adjust the point cloud


        #segment_limbs = True

        if pmat is not None:
            if np.sum(pmat) < 5000:
                smpl_verts = smpl_verts * 0.001


        smpl_verts_quad = np.concatenate((smpl_verts, np.ones((smpl_verts.shape[0], 1))), axis = 1)
        smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)

        print smpl_verts_quad.shape

        transform_A = np.identity(4)

        transform_B = np.identity(4)
        transform_B[1, 3] = 4.0 #move things over
        smpl_verts_B = np.swapaxes(np.matmul(transform_B, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_C = np.identity(4)
        transform_C[1, 3] = 2.0 #move things over
        smpl_verts_C = np.swapaxes(np.matmul(transform_C, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_D = np.identity(4)
        transform_D[1, 3] = 3.0 #move things over
        smpl_verts_D = np.swapaxes(np.matmul(transform_D, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_E = np.identity(4)
        transform_E[1, 3] = 5.0 #move things over
        smpl_verts_E = np.swapaxes(np.matmul(transform_E, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_F = np.identity(4)
        transform_F[1, 3] = 1.0 #move things over





        from matplotlib import cm

        #downsample the point cloud and get the normals
        pc_red, pc_red_norm = self.downspl_pc_get_normals(pc, camera_point)

        pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
        pc_red_C = np.swapaxes(np.matmul(transform_C, pc_red_quad), 0, 1)[:, 0:3]
        pc_red_norm_tri = np.swapaxes(pc_red_norm, 0, 1)
        pc_red_norm_C = np.swapaxes(np.matmul(transform_C[0:3, 0:3], pc_red_norm_tri), 0, 1)[:, 0:3]
        camera_point_C = np.matmul(transform_C, np.array([camera_point[0], camera_point[1], camera_point[2], 1.0]))[0:3]


        pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
        pc_red_D = np.swapaxes(np.matmul(transform_D, pc_red_quad), 0, 1)[:, 0:3]
        pc_red_norm_tri = np.swapaxes(pc_red_norm, 0, 1)
        pc_red_norm_D = np.swapaxes(np.matmul(transform_D[0:3, 0:3], pc_red_norm_tri), 0, 1)[:, 0:3]
        camera_point_D = np.matmul(transform_D, np.array([camera_point[0], camera_point[1], camera_point[2], 1.0]))[0:3]

        pc_red_quad = np.swapaxes(np.concatenate((pc_red, np.ones((pc_red.shape[0], 1))), axis = 1), 0, 1)
        pc_red_F = np.swapaxes(np.matmul(transform_F, pc_red_quad), 0, 1)[:, 0:3]
        pc_red_norm_tri = np.swapaxes(pc_red_norm, 0, 1)
        pc_red_norm_F = np.swapaxes(np.matmul(transform_F[0:3, 0:3], pc_red_norm_tri), 0, 1)[:, 0:3]
        camera_point_F = np.matmul(transform_F, np.array([camera_point[0], camera_point[1], camera_point[2], 1.0]))[0:3]



        human_mesh_vtx_all, human_mesh_face_all = self.get_human_mesh_parts(smpl_verts, smpl_faces, segment_limbs=False)
        human_mesh_vtx_parts, human_mesh_face_parts = self.get_human_mesh_parts(smpl_verts_B, smpl_faces, segment_limbs=True)
        human_mesh_vtx_mesherr, human_mesh_face_mesherr = self.get_human_mesh_parts(smpl_verts_C, smpl_faces, segment_limbs=False)
        human_mesh_vtx_pcerr, human_mesh_face_pcerr = self.get_human_mesh_parts(smpl_verts_D, smpl_faces, segment_limbs=False)
        human_mesh_vtx_mcd, human_mesh_face_mcd = self.get_human_mesh_parts(smpl_verts_E, smpl_faces, segment_limbs=False)


        human_mesh_face_mesherr_red = []
        #only use the vertices that are facing the camera
        for part_idx in range(len(human_mesh_vtx_mesherr)):
            human_mesh_face_mesherr_red.append(self.reduce_by_cam_dir(human_mesh_vtx_mesherr[part_idx], human_mesh_face_mesherr[part_idx], camera_point_C))


        human_mesh_face_pcerr_red = []
        #only use the vertices that are facing the camera
        for part_idx in range(len(human_mesh_vtx_mesherr)):
            human_mesh_face_pcerr_red.append(self.reduce_by_cam_dir(human_mesh_vtx_pcerr[part_idx], human_mesh_face_pcerr[part_idx], camera_point_D))


        #GET LIMBS WITH PMAT
        tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all[0]), faces = np.array(human_mesh_face_all[0]))
        tm_list = [tm_curr]


        #GET SEGMENTED LIMBS
        tm_list_seg = []
        for idx in range(len(human_mesh_vtx_parts)):
            tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts[idx]), faces = np.array(human_mesh_face_parts[idx]))
            tm_list_seg.append(tm_curr)



        #GET MESHERROR
        verts_idx_red = np.unique(human_mesh_face_mesherr_red[0])
        verts_red = human_mesh_vtx_mesherr[0][verts_idx_red, :]

        print np.shape(verts_red)

        # get the nearest point from each vert to some pc point, regardless of the normal
        vert_to_nearest_point_error_list = []
        for vert_idx in range(verts_red.shape[0]):
            curr_vtx = verts_red[vert_idx, :]
            mesherr_dist = pc_red_C - curr_vtx
            mesherr_eucl = np.linalg.norm(mesherr_dist, axis=1)
            curr_error = np.min(mesherr_eucl)
            vert_to_nearest_point_error_list.append(curr_error)

        # get the nearest point from ALL verts to some pc point, regardless of the normal - for coloring
        # we need this as a hack because the face indexing only refers to the original set of verts
        all_vert_to_nearest_point_error_list = []
        for all_vert_idx in range(human_mesh_vtx_mesherr[0].shape[0]):
            curr_vtx = human_mesh_vtx_mesherr[0][all_vert_idx, :]
            all_dist = pc_red_C - curr_vtx
            all_eucl = np.linalg.norm(all_dist, axis=1)
            curr_error = np.min(all_eucl)
            all_vert_to_nearest_point_error_list.append(curr_error)

        # normalize by the average area of triangles around each point. the verts are much less spatially distributed well
        # than the points in the point cloud.
        norm_area_avg = self.get_triangle_area_vert_weight(human_mesh_vtx_mesherr[0], human_mesh_face_mesherr_red[0], verts_idx_red)

        print np.shape(norm_area_avg), np.shape(vert_to_nearest_point_error_list)
        vert_to_nearest_point_error_list = vert_to_nearest_point_error_list[0:np.shape(norm_area_avg)[0]]
        norm_vert_to_nearest_point_error = np.array(vert_to_nearest_point_error_list) * norm_area_avg
        print "average vert to nearest pc point error, regardless of normal:", np.mean(norm_vert_to_nearest_point_error)

        verts_color_error = np.array(all_vert_to_nearest_point_error_list) / np.max(vert_to_nearest_point_error_list)
        verts_color_jet = cm.jet(verts_color_error)[:, 0:3]# * 5.

        verts_color_jet_top = np.concatenate((verts_color_jet, np.ones((verts_color_jet.shape[0], 1))*0.9), axis = 1)
        verts_color_jet_bot = np.concatenate((verts_color_jet*0.3, np.ones((verts_color_jet.shape[0], 1))*0.9), axis = 1)


        all_verts = np.array(human_mesh_vtx_mesherr[0])
        faces_red = np.array(human_mesh_face_mesherr_red[0])
        faces_underside = np.concatenate((faces_red[:, 0:1],
                                          faces_red[:, 2:3],
                                          faces_red[:, 1:2]), axis = 1) + 6890

        human_vtx_both_sides = np.concatenate((all_verts, all_verts+0.0001), axis = 0)
        human_mesh_faces_both_sides = np.concatenate((faces_red, faces_underside), axis = 0)
        verts_color_jet_both_sides = np.concatenate((verts_color_jet_top, verts_color_jet_bot), axis = 0)

        tm_curr = trimesh.base.Trimesh(vertices=human_vtx_both_sides,
                                       faces=human_mesh_faces_both_sides,
                                       vertex_colors = verts_color_jet_both_sides)
        tm_list_mesherr =[tm_curr]



        #GET PCERROR
        all_verts = np.array(human_mesh_vtx_pcerr[0])
        faces_red = np.array(human_mesh_face_pcerr_red[0])
        faces_underside = np.concatenate((faces_red[:, 0:1],
                                          faces_red[:, 2:3],
                                          faces_red[:, 1:2]), axis = 1) + 6890


        verts_greysc_color = 1.0 * (all_verts[:, 2:3] - np.max(all_verts[:, 2])) / (np.min(all_verts[:, 2]) - np.max(all_verts[:, 2]))
        #print np.min(verts_greysc_color), np.max(verts_greysc_color), np.shape(verts_greysc_color)

        verts_greysc_color = np.concatenate((verts_greysc_color, verts_greysc_color, verts_greysc_color), axis=1)
        #print np.shape(verts_greysc_color)

        verts_color_grey_top = np.concatenate((verts_greysc_color, np.ones((verts_greysc_color.shape[0], 1))*0.7), axis = 1)
        verts_color_grey_bot = np.concatenate((verts_greysc_color*0.3, np.ones((verts_greysc_color.shape[0], 1))*0.7), axis = 1)

        human_vtx_both_sides = np.concatenate((all_verts, all_verts+0.0001), axis = 0)
        human_mesh_faces_both_sides = np.concatenate((faces_red, faces_underside), axis = 0)
        verts_color_jet_both_sides = np.concatenate((verts_color_grey_top, verts_color_grey_bot), axis = 0)

        tm_curr = trimesh.base.Trimesh(vertices=human_vtx_both_sides,
                                       faces=human_mesh_faces_both_sides,
                                       vertex_colors = verts_color_jet_both_sides)
        tm_list_pcerr = [tm_curr]




        if dropout_variance is not None:
            #GET MONTE CARLO DROPOUT COLORED MESH
            verts_mcd_color = (dropout_variance - np.min(dropout_variance)) / (np.max(dropout_variance) - np.min(dropout_variance))
            verts_mcd_color_jet = cm.Reds(verts_mcd_color)[:, 0:3]
            verts_mcd_color_jet = np.concatenate((verts_mcd_color_jet, np.ones((verts_mcd_color_jet.shape[0], 1))*0.9), axis = 1)
            tm_curr = trimesh.base.Trimesh(vertices=human_mesh_vtx_mcd[0],
                                           faces=human_mesh_face_mcd[0],
                                           vertex_colors = verts_mcd_color_jet)
            tm_list_mcd =[tm_curr]


        mesh_list = []
        mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.human_mat, wireframe = True))


        mesh_list_seg = []
        for idx in range(len(tm_list_seg)):
            mesh_list_seg.append(pyrender.Mesh.from_trimesh(tm_list_seg[idx], material = self.mesh_parts_mat_list[idx], wireframe = True))

        mesh_list_mesherr = []
        mesh_list_mesherr.append(pyrender.Mesh.from_trimesh(tm_list_mesherr[0], smooth=False))

        mesh_list_pcerr = []
        mesh_list_pcerr.append(pyrender.Mesh.from_trimesh(tm_list_pcerr[0], material = self.human_mat_D, smooth=False))

        if dropout_variance is not None:
            mesh_list_mcd = []
            mesh_list_mcd.append(pyrender.Mesh.from_trimesh(tm_list_mcd[0], smooth=False))






        #smpl_tm = trimesh.base.Trimesh(vertices=smpl_verts, faces=smpl_faces)
        #smpl_mesh = pyrender.Mesh.from_trimesh(smpl_tm, material=self.human_mat, wireframe = True)

        pc_greysc_color = 0.3 * (pc_red_C[:, 2:3] - np.max(pc_red_C[:, 2])) / (np.min(pc_red_C[:, 2]) - np.max(pc_red_C[:, 2]))
        pc_mesh_mesherr = pyrender.Mesh.from_points(pc_red_C, colors=np.concatenate((pc_greysc_color, pc_greysc_color, pc_greysc_color), axis=1))

        pc_greysc_color2 = 0.0 * (pc_red_F[:, 2:3] - np.max(pc_red_F[:, 2])) / (np.min(pc_red_F[:, 2]) - np.max(pc_red_F[:, 2]))
        pc_mesh_mesherr2 = pyrender.Mesh.from_points(pc_red_F, colors=np.concatenate((pc_greysc_color2, pc_greysc_color2, pc_greysc_color2), axis=1))






        faces_red = human_mesh_face_pcerr_red[0]
        verts_idx_red = np.unique(faces_red)
        verts_red = human_mesh_vtx_pcerr[0][verts_idx_red, :]

        # get the nearest point from each pc point to some vert, regardless of the normal
        pc_to_nearest_vert_error_list = []
        for point_idx in range(pc_red_D.shape[0]):
            curr_point = pc_red_D[point_idx, :]
            all_dist = verts_red - curr_point
            all_eucl = np.linalg.norm(all_dist, axis=1)
            curr_error = np.min(all_eucl)
            pc_to_nearest_vert_error_list.append(curr_error)
            # break
        print "average pc point to nearest vert error, regardless of normal:", np.mean(pc_to_nearest_vert_error_list)
        pc_color_error = np.array(pc_to_nearest_vert_error_list) / np.max(pc_to_nearest_vert_error_list)
        pc_color_jet = cm.jet(pc_color_error)[:, 0:3]

        pc_mesh_pcerr = pyrender.Mesh.from_points(pc_red_D, colors = pc_color_jet)






        if smpl_render_points == True:
            verts_idx_red = np.unique(human_mesh_face_all_red[0])

            verts_red = smpl_verts[verts_idx_red, :]
            smpl_pc_mesh = pyrender.Mesh.from_points(verts_red, colors = [5.0, 0.0, 0.0])
        else: smpl_pc_mesh = None


        #print m.r
        #print artag_r
        #create mini meshes for AR tags
        artag_meshes = []
        if markers is not None:
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



        if pmat is not None:
            pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat, bedangle)
            pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors = pmat_facecolors)
            pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth = False)

            pmat_verts2, _, _ = self.get_3D_pmat_markers(pmat, bedangle)
            pmat_verts2 = np.array(pmat_verts2)
            pmat_verts2 = np.concatenate((np.swapaxes(pmat_verts2, 0, 1), np.ones((1, pmat_verts2.shape[0]))), axis = 0)
            pmat_verts2 = np.swapaxes(np.matmul(transform_F, pmat_verts2), 0, 1)[:, 0:3]


            pmat_tm2 = trimesh.base.Trimesh(vertices=pmat_verts2, faces=pmat_faces, face_colors = pmat_facecolors)
            pmat_mesh2 = pyrender.Mesh.from_trimesh(pmat_tm2, smooth = False)

        else:
            pmat_mesh = None
            pmat_mesh2 = None


        #print "Viewing"
        if self.first_pass == True:

            for mesh_part in mesh_list:
                self.scene.add(mesh_part)
            for mesh_part_seg in mesh_list_seg:
                self.scene.add(mesh_part_seg)
            for mesh_part_mesherr in mesh_list_mesherr:
                self.scene.add(mesh_part_mesherr)
            for mesh_part_pcerr in mesh_list_pcerr:
                self.scene.add(mesh_part_pcerr)
            if dropout_variance is not None:
                for mesh_part_mcd in mesh_list_mcd:
                    self.scene.add(mesh_part_mcd)


            if pc_mesh_mesherr is not None:
                self.scene.add(pc_mesh_mesherr)
            if pc_mesh_pcerr is not None:
                self.scene.add(pc_mesh_pcerr)


            if pc_mesh_mesherr2 is not None:
                self.scene.add(pc_mesh_mesherr2)

            if pmat_mesh is not None:
                self.scene.add(pmat_mesh)

            if pmat_mesh2 is not None:
                self.scene.add(pmat_mesh2)

            if smpl_pc_mesh is not None:
                self.scene.add(smpl_pc_mesh)

            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    self.scene.add(artag_mesh)


            lighting_intensity = 20.

            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                                          point_size=2, run_in_thread=True, viewport_size=(1000, 1000))



            self.first_pass = False

            self.node_list = []
            for mesh_part in mesh_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_list.append(node)
            self.node_list_seg = []
            for mesh_part_seg in mesh_list_seg:
                for node in self.scene.get_nodes(obj=mesh_part_seg):
                    self.node_list_seg.append(node)
            self.node_list_mesherr = []
            for mesh_part_mesherr in mesh_list_mesherr:
                for node in self.scene.get_nodes(obj=mesh_part_mesherr):
                    self.node_list_mesherr.append(node)
            self.node_list_pcerr = []
            for mesh_part_pcerr in mesh_list_pcerr:
                for node in self.scene.get_nodes(obj=mesh_part_pcerr):
                    self.node_list_pcerr.append(node)
            if dropout_variance is not None:
                self.node_list_mcd = []
                for mesh_part_mcd in mesh_list_mcd:
                    for node in self.scene.get_nodes(obj=mesh_part_mcd):
                        self.node_list_mcd.append(node)




            if pc_mesh_mesherr is not None:
                for node in self.scene.get_nodes(obj=pc_mesh_mesherr):
                    self.point_cloud_node_mesherr = node

            if pc_mesh_pcerr is not None:
                for node in self.scene.get_nodes(obj=pc_mesh_pcerr):
                    self.point_cloud_node_pcerr = node

            if pc_mesh_mesherr2 is not None:
                for node in self.scene.get_nodes(obj=pc_mesh_mesherr2):
                    self.point_cloud_node_mesherr2 = node

            if smpl_pc_mesh is not None:
                for node in self.scene.get_nodes(obj=smpl_pc_mesh):
                    self.smpl_pc_mesh_node = node

            self.artag_nodes = []
            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    for node in self.scene.get_nodes(obj=artag_mesh):
                        self.artag_nodes.append(node)
            if pmat_mesh is not None:
                for node in self.scene.get_nodes(obj=pmat_mesh):
                    self.pmat_node = node
            if pmat_mesh2 is not None:
                for node in self.scene.get_nodes(obj=pmat_mesh2):
                    self.pmat_node2 = node


        else:
            self.viewer.render_lock.acquire()

            #reset the human mesh
            for idx in range(len(mesh_list)):
                self.scene.remove_node(self.node_list[idx])
                self.scene.add(mesh_list[idx])
                for node in self.scene.get_nodes(obj=mesh_list[idx]):
                    self.node_list[idx] = node

            #reset the segmented human mesh
            for idx in range(len(mesh_list_seg)):
                self.scene.remove_node(self.node_list_seg[idx])
                self.scene.add(mesh_list_seg[idx])
                for node in self.scene.get_nodes(obj=mesh_list_seg[idx]):
                    self.node_list_seg[idx] = node

            #reset the mesh error human rendering
            for idx in range(len(mesh_list_mesherr)):
                self.scene.remove_node(self.node_list_mesherr[idx])
                self.scene.add(mesh_list_mesherr[idx])
                for node in self.scene.get_nodes(obj=mesh_list_mesherr[idx]):
                    self.node_list_mesherr[idx] = node

            #reset the pc error human rendering
            for idx in range(len(mesh_list_pcerr)):
                self.scene.remove_node(self.node_list_pcerr[idx])
                self.scene.add(mesh_list_pcerr[idx])
                for node in self.scene.get_nodes(obj=mesh_list_pcerr[idx]):
                    self.node_list_pcerr[idx] = node

            if dropout_variance is not None:
                #reset the mcd human rendering
                for idx in range(len(mesh_list_mcd)):
                    self.scene.remove_node(self.node_list_mcd[idx])
                    self.scene.add(mesh_list_mcd[idx])
                    for node in self.scene.get_nodes(obj=mesh_list_mcd[idx]):
                        self.node_list_mcd[idx] = node





            #reset the point cloud mesh for mesherr
            if pc_mesh_mesherr is not None:
                self.scene.remove_node(self.point_cloud_node_mesherr)
                self.scene.add(pc_mesh_mesherr)
                for node in self.scene.get_nodes(obj=pc_mesh_mesherr):
                    self.point_cloud_node_mesherr = node

            #reset the point cloud mesh for pcerr
            if pc_mesh_pcerr is not None:
                self.scene.remove_node(self.point_cloud_node_pcerr)
                self.scene.add(pc_mesh_pcerr)
                for node in self.scene.get_nodes(obj=pc_mesh_pcerr):
                    self.point_cloud_node_pcerr = node

            #reset the point cloud mesh for mesherr
            if pc_mesh_mesherr2 is not None:
                self.scene.remove_node(self.point_cloud_node_mesherr2)
                self.scene.add(pc_mesh_mesherr2)
                for node in self.scene.get_nodes(obj=pc_mesh_mesherr2):
                    self.point_cloud_node_mesherr2 = node

            #reset the vert pc mesh
            if smpl_pc_mesh is not None:
                self.scene.remove_node(self.smpl_pc_mesh_node)
                self.scene.add(smpl_pc_mesh)
                for node in self.scene.get_nodes(obj=smpl_pc_mesh):
                    self.smpl_pc_mesh_node = node


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
            if pmat_mesh is not None:
                self.scene.remove_node(self.pmat_node)
                self.scene.add(pmat_mesh)
                for node in self.scene.get_nodes(obj=pmat_mesh):
                    self.pmat_node = node


            #reset the pmat mesh
            if pmat_mesh2 is not None:
                self.scene.remove_node(self.pmat_node2)
                self.scene.add(pmat_mesh2)
                for node in self.scene.get_nodes(obj=pmat_mesh2):
                    self.pmat_node2 = node



            #print self.scene.get_nodes()
            self.viewer.render_lock.release()
