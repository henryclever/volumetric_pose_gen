

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
        self.human_mat_D = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1 , 1.0], alphaMode="BLEND")

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
                    pmat_xyz[j, i, 2] = 0.12 + 0.075
                    # print marker.pose.position.x, 'x'
                else:

                    pmat_xyz[j, i, 0] = ((41) * 0.0286 + (23 - j) * 0.0286 * np.cos(np.deg2rad(angle)) \
                                        - (0.0286 * 3 * np.sin(np.deg2rad(angle))) * 0.85)*1.04 + 0.12#1.1406 + 0.05
                    pmat_xyz[j, i, 2] = -((23 - j) * 0.0286 * np.sin(np.deg2rad(angle))) * 0.85 + 0.12 + 0.075
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


    def get_human_mesh_parts(self, smpl_verts, smpl_faces, segment_limbs = False):

        segmented_dict = load_pickle('segmented_mesh_idx_faces.p')

        if segment_limbs == True:
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



    def render_mesh_pc_bed_pyrender(self, smpl_verts, smpl_faces, camera_point, bedangle,
                                    pc = None, pmat = None, smpl_render_points = False, facing_cam_only = False,
                                    viz_type = None, markers = None, segment_limbs = False):

        #downsample the point cloud and get the normals
        if pc is not None:
            pc_red, pc_red_norm = self.downspl_pc_get_normals(pc, camera_point)
        else: pc_red = None

        #segment_limbs = True

        if pmat is not None:
            if np.sum(pmat) < 5000:
                smpl_verts = smpl_verts * 0.001

        human_mesh_vtx_parts, human_mesh_face_parts = self.get_human_mesh_parts(smpl_verts, smpl_faces, segment_limbs)

        if facing_cam_only == True:
            human_mesh_face_parts_red = []
            #only use the vertices that are facing the camera
            for part_idx in range(len(human_mesh_vtx_parts)):
                human_mesh_face_parts_red.append(self.reduce_by_cam_dir(human_mesh_vtx_parts[part_idx], human_mesh_face_parts[part_idx], camera_point))
        else:
            human_mesh_face_parts_red = human_mesh_face_parts


        tm_list = []
        for idx in range(len(human_mesh_vtx_parts)):
            if viz_type == 'mesh_error' and segment_limbs == False:

                verts_idx_red = np.unique(human_mesh_face_parts_red[0])
                verts_red = human_mesh_vtx_parts[0][verts_idx_red, :]

                print np.shape(verts_red), 'verts red'

                # get the nearest point from each vert to some pc point, regardless of the normal
                vert_to_nearest_point_error_list = []
                for vert_idx in range(verts_red.shape[0]):
                    curr_vtx = verts_red[vert_idx, :]
                    all_dist = pc_red - curr_vtx
                    all_eucl = np.linalg.norm(all_dist, axis=1)
                    curr_error = np.min(all_eucl)
                    vert_to_nearest_point_error_list.append(curr_error)

                # get the nearest point from ALL verts to some pc point, regardless of the normal - for coloring
                # we need this as a hack because the face indexing only refers to the original set of verts
                all_vert_to_nearest_point_error_list = []
                for all_vert_idx in range(human_mesh_vtx_parts[0].shape[0]):
                    curr_vtx = human_mesh_vtx_parts[0][all_vert_idx, :]
                    all_dist = pc_red - curr_vtx
                    all_eucl = np.linalg.norm(all_dist, axis=1)
                    curr_error = np.min(all_eucl)
                    all_vert_to_nearest_point_error_list.append(curr_error)

                # normalize by the average area of triangles around each point. the verts are much less spatially distributed well
                # than the points in the point cloud.
                norm_area_avg = self.get_triangle_area_vert_weight(human_mesh_vtx_parts[0], human_mesh_face_parts_red[0], verts_idx_red)
                norm_vert_to_nearest_point_error = np.array(vert_to_nearest_point_error_list) * norm_area_avg
                print "average vert to nearest pc point error, regardless of normal:", np.mean(
                    norm_vert_to_nearest_point_error)

                verts_color_error = np.array(all_vert_to_nearest_point_error_list) / np.max(vert_to_nearest_point_error_list)
                verts_color_jet = cm.jet(verts_color_error)[:, 0:3]# * 5.

                verts_color_jet_top = np.concatenate((verts_color_jet, np.ones((verts_color_jet.shape[0], 1))*0.9), axis = 1)
                verts_color_jet_bot = np.concatenate((verts_color_jet*0.3, np.ones((verts_color_jet.shape[0], 1))*0.9), axis = 1)


                all_verts = np.array(human_mesh_vtx_parts[0])
                faces_red = np.array(human_mesh_face_parts_red[0])
                faces_underside = np.concatenate((faces_red[:, 0:1],
                                                  faces_red[:, 2:3],
                                                  faces_red[:, 1:2]), axis = 1) + 6890

                human_vtx_both_sides = np.concatenate((all_verts, all_verts+0.0001), axis = 0)
                human_mesh_faces_both_sides = np.concatenate((faces_red, faces_underside), axis = 0)
                verts_color_jet_both_sides = np.concatenate((verts_color_jet_top, verts_color_jet_bot), axis = 0)

                tm_curr = trimesh.base.Trimesh(vertices=human_vtx_both_sides,
                                               faces=human_mesh_faces_both_sides,
                                               vertex_colors = verts_color_jet_both_sides)
                tm_list.append(tm_curr)

            elif viz_type == 'pc_error' and segment_limbs == False:

                all_verts = np.array(human_mesh_vtx_parts[0])
                faces_red = np.array(human_mesh_face_parts_red[0])
                faces_underside = np.concatenate((faces_red[:, 0:1],
                                                  faces_red[:, 2:3],
                                                  faces_red[:, 1:2]), axis = 1) + 6890


                verts_greysc_color = 1.0 * (all_verts[:, 2:3] - np.max(all_verts[:, 2])) / (np.min(all_verts[:, 2]) - np.max(all_verts[:, 2]))
                print np.min(verts_greysc_color), np.max(verts_greysc_color), np.shape(verts_greysc_color)

                verts_greysc_color = np.concatenate((verts_greysc_color, verts_greysc_color, verts_greysc_color), axis=1)
                print np.shape(verts_greysc_color)

                verts_color_grey_top = np.concatenate((verts_greysc_color, np.ones((verts_greysc_color.shape[0], 1))*0.7), axis = 1)
                verts_color_grey_bot = np.concatenate((verts_greysc_color*0.3, np.ones((verts_greysc_color.shape[0], 1))*0.7), axis = 1)

                human_vtx_both_sides = np.concatenate((all_verts, all_verts+0.0001), axis = 0)
                human_mesh_faces_both_sides = np.concatenate((faces_red, faces_underside), axis = 0)
                verts_color_jet_both_sides = np.concatenate((verts_color_grey_top, verts_color_grey_bot), axis = 0)

                tm_curr = trimesh.base.Trimesh(vertices=human_vtx_both_sides,
                                               faces=human_mesh_faces_both_sides,
                                               vertex_colors = verts_color_jet_both_sides)
                tm_list.append(tm_curr)

            else:
                tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts[idx]), faces = np.array(human_mesh_face_parts_red[idx]))
                tm_list.append(tm_curr)

        mesh_list = []
        for idx in range(len(tm_list)):
            if len(tm_list) == 1:
                if viz_type is not None:
                    mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[idx], smooth = False))
                else:
                    mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[idx], material = self.human_mat, wireframe = True))
            else:
                mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[idx], material = self.mesh_parts_mat_list[idx], wireframe = True))


        #smpl_tm = trimesh.base.Trimesh(vertices=smpl_verts, faces=smpl_faces)
        #smpl_mesh = pyrender.Mesh.from_trimesh(smpl_tm, material=self.human_mat, wireframe = True)


        #get Point cloud mesh
        if pc_red is not None:
            if viz_type == 'mesh_error':
                pc_greysc_color = 0.3*(pc_red[:, 2:3] - np.max(pc_red[:, 2]))/(np.min(pc_red[:, 2])-np.max(pc_red[:, 2]))

                pc_mesh = pyrender.Mesh.from_points(pc_red, colors = np.concatenate((pc_greysc_color, pc_greysc_color, pc_greysc_color), axis=1))
            elif viz_type == 'pc_error':


                from matplotlib import cm

                faces_red = human_mesh_face_parts_red[0]
                verts_idx_red = np.unique(faces_red)
                verts_red = human_mesh_vtx_parts[0][verts_idx_red, :]

                # get the nearest point from each pc point to some vert, regardless of the normal
                pc_to_nearest_vert_error_list = []
                for point_idx in range(pc_red.shape[0]):
                    curr_point = pc_red[point_idx, :]
                    all_dist = verts_red - curr_point
                    all_eucl = np.linalg.norm(all_dist, axis=1)
                    curr_error = np.min(all_eucl)
                    pc_to_nearest_vert_error_list.append(curr_error)
                    # break
                print "average pc point to nearest vert error, regardless of normal:", np.mean(
                    pc_to_nearest_vert_error_list)
                pc_color_error = np.array(pc_to_nearest_vert_error_list) / np.max(pc_to_nearest_vert_error_list)
                pc_color_jet = cm.jet(pc_color_error)[:, 0:3]

                pc_mesh = pyrender.Mesh.from_points(pc_red, colors = pc_color_jet)

            else:
                pc_mesh = pyrender.Mesh.from_points(pc_red, colors = [1.0, 0.0, 0.0])
        else: pc_mesh = None

        if smpl_render_points == True:
            verts_idx_red = np.unique(human_mesh_face_parts_red[0])

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
        else:
            pmat_mesh = None


        #print "Viewing"
        if self.first_pass == True:

            for mesh_part in mesh_list:
                self.scene.add(mesh_part)
            if pc_mesh is not None:
                self.scene.add(pc_mesh)
            if pmat_mesh is not None:
                self.scene.add(pmat_mesh)
            if smpl_pc_mesh is not None:
                self.scene.add(smpl_pc_mesh)

            for artag_mesh in artag_meshes:
                if artag_mesh is not None:
                    self.scene.add(artag_mesh)

            if segment_limbs == True:
                lighting_intensity = 5.
            else:
                if viz_type == 'pc_error':
                    lighting_intensity = 10.
                else:
                    lighting_intensity = 20.

            #camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
            s = np.sqrt(2) / 2
            camera_pose = np.array([[0.0, -s, s, -0.3],
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, s, s, -0.35],
                                    [0.0, 0.0, 0.0, 1.0]])

            #self.scene.add(camera, pose=camera_pose)

            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                                          point_size=5, run_in_thread=True, viewport_size=(500, 500))



            self.first_pass = False

            self.node_list = []
            for mesh_part in mesh_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_list.append(node)

            if pc_mesh is not None:
                for node in self.scene.get_nodes(obj=pc_mesh):
                    self.point_cloud_node = node
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



            #print self.scene.get_nodes()
            self.viewer.render_lock.release()



    def render_mesh_pc_bed_pyrender_everything(self, smpl_verts, smpl_faces, camera_point, bedangle,
                                    pc = None, pmat = None, smpl_render_points = False, markers = None,
                                    dropout_variance=None):


        #segment_limbs = True

        if pmat is not None:
            if np.sum(pmat) < 5000:
                smpl_verts = smpl_verts * 0.001


        smpl_verts_quad = np.concatenate((smpl_verts, np.ones((smpl_verts.shape[0], 1))), axis = 1)
        smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)

        print smpl_verts_quad.shape

        transform_A = np.identity(4)

        transform_B = np.identity(4)
        transform_B[1, 3] = 1.0 #move things over
        smpl_verts_B = np.swapaxes(np.matmul(transform_B, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_C = np.identity(4)
        transform_C[1, 3] = 2.0 #move things over
        smpl_verts_C = np.swapaxes(np.matmul(transform_C, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_D = np.identity(4)
        transform_D[1, 3] = 3.0 #move things over
        smpl_verts_D = np.swapaxes(np.matmul(transform_D, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_E = np.identity(4)
        transform_E[1, 3] = 4.0 #move things over
        smpl_verts_E = np.swapaxes(np.matmul(transform_E, smpl_verts_quad), 0, 1)[:, 0:3]





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

        mesh_list_mcd = []
        mesh_list_mcd.append(pyrender.Mesh.from_trimesh(tm_list_mcd[0], smooth=False))






        #smpl_tm = trimesh.base.Trimesh(vertices=smpl_verts, faces=smpl_faces)
        #smpl_mesh = pyrender.Mesh.from_trimesh(smpl_tm, material=self.human_mat, wireframe = True)

        pc_greysc_color = 0.3 * (pc_red_C[:, 2:3] - np.max(pc_red_C[:, 2])) / (np.min(pc_red_C[:, 2]) - np.max(pc_red_C[:, 2]))
        pc_mesh_mesherr = pyrender.Mesh.from_points(pc_red_C, colors=np.concatenate((pc_greysc_color, pc_greysc_color, pc_greysc_color), axis=1))






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
        else:
            pmat_mesh = None


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
            for mesh_part_mcd in mesh_list_mcd:
                self.scene.add(mesh_part_mcd)


            if pc_mesh_mesherr is not None:
                self.scene.add(pc_mesh_mesherr)
            if pc_mesh_pcerr is not None:
                self.scene.add(pc_mesh_pcerr)


            if pmat_mesh is not None:
                self.scene.add(pmat_mesh)
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



            #print self.scene.get_nodes()
            self.viewer.render_lock.release()
