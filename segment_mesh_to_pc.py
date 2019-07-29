import numpy as np
import random
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model

import trimesh
import pyrender
import pyglet

#volumetric pose gen libraries
import lib_visualization as libVisualization
import lib_kinematics as libKinematics
import lib_render as libRender
from process_yash_data import ProcessYashData
#ROS
try:
    import rospy
    import tf
except:
    pass

import tensorflow as tensorflow
import cPickle as pickle


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL

import cPickle as pkl

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import os


class GeneratePose():
    def __init__(self, sampling = "NORMAL", sigma = 0, one_side_range = 0, gender="m"):
        ## Load SMPL model (here we load the female model)
        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
        self.m = load_model(model_path)

        ## Assign random pose and shape parameters

        self.m.betas[:] = np.random.rand(self.m.betas.size) * .0
        #self.m.betas[5] = 20.

        self.m.pose[0] = 0 #pitch rotation of the person in space. 0 means the person is upside down facing back. pi is standing up facing forward
        self.m.pose[1] = 0 #roll of the person in space. -pi/2 means they are tilted to their right side
        self.m.pose[2] = 0#-np.pi/4 #yaw of the person in space, like turning around normal to the ground

        self.m.pose[3] = 0#-np.pi/4 #left hip extension (i.e. leg bends back for np.pi/2)
        self.m.pose[4] = 0#np.pi/8 #left leg yaw about hip, where np.pi/2 makes bowed leg
        self.m.pose[5] = 0.#np.pi/4 #left leg abduction (POS) /adduction (NEG)

        self.m.pose[6] = 0#-np.pi/4 #right hip extension (i.e. leg bends back for np.pi/2)
        self.m.pose[7] = 0#-np.pi/8
        self.m.pose[8] = 0.#-np.pi/4 #right leg abduction (NEG) /adduction (POS)

        self.m.pose[9] = 0 #bending of spine at hips. np.pi/2 means person bends down to touch the ground
        self.m.pose[10] = 0 #twisting of spine at hips. body above spine yaws normal to the ground
        self.m.pose[11] = 0 #bending of spine at hips. np.pi/2 means person bends down sideways to touch the ground 3

        self.m.pose[12] = 0#np.pi/4 #left knee extension. (i.e. knee bends back for np.pi/2)
        self.m.pose[13] = 0 #twisting of knee normal to ground. KEEP AT ZERO
        self.m.pose[14] = 0 #bending of knee sideways. KEEP AT ZERO

        self.m.pose[15] = 0#np.pi/4 #right knee extension (i.e. knee bends back for np.pi/2)

        self.m.pose[18] = 0 #bending at mid spine. makes person into a hunchback for positive values
        self.m.pose[19] = 0#twisting of midspine. body above midspine yaws normal to the ground
        self.m.pose[20] = 0 #bending of midspine, np.pi/2 means person bends down sideways to touch ground 6

        self.m.pose[21] = 0 #left ankle flexion/extension
        self.m.pose[22] = 0 #left ankle yaw about leg
        self.m.pose[23] = 0 #left ankle twist KEEP CLOSE TO ZERO

        self.m.pose[24] = 0 #right ankle flexion/extension
        self.m.pose[25] = 0 #right ankle yaw about leg
        self.m.pose[26] = 0#np.pi/4 #right ankle twist KEEP CLOSE TO ZERO

        self.m.pose[27] = 0 #bending at upperspine. makes person into a hunchback for positive values
        self.m.pose[28] = 0#twisting of upperspine. body above upperspine yaws normal to the ground
        self.m.pose[29] = 0 #bending of upperspine, np.pi/2 means person bends down sideways to touch ground 9

        self.m.pose[30] = 0 #flexion/extension of left ankle midpoint

        self.m.pose[33] = 0 #flexion/extension of right ankle midpoint

        self.m.pose[36] = 0#np.pi/2 #flexion/extension of neck. i.e. whiplash 12
        self.m.pose[37] = 0#-np.pi/2 #yaw of neck
        self.m.pose[38] = 0#np.pi/4  #tilt head side to side

        self.m.pose[39] = 0 #left inner shoulder roll
        self.m.pose[40] = 0 #left inner shoulder yaw, negative moves forward
        self.m.pose[41] = 0.#np.pi/4 #left inner shoulder pitch, positive moves up

        self.m.pose[42] = 0
        self.m.pose[43] = 0 #right inner shoulder yaw, positive moves forward
        self.m.pose[44] = 0.#-np.pi/4 #right inner shoulder pitch, positive moves down

        self.m.pose[45] = 0 #flexion/extension of head 15

        self.m.pose[48] = 0#-np.pi/4 #left outer shoulder roll
        self.m.pose[49] = 0#-np.pi/4
        self.m.pose[50] = 0#np.pi/4 #left outer shoulder pitch

        self.m.pose[51] = 0#-np.pi/4 #right outer shoulder roll
        self.m.pose[52] = 0#np.pi/4
        self.m.pose[53] = 0#-np.pi/4

        self.m.pose[54] = 0 #left elbow roll KEEP AT ZERO
        self.m.pose[55] = 0#np.pi/3 #left elbow flexion/extension. KEEP NEGATIVE
        self.m.pose[56] = 0 #left elbow KEEP AT ZERO

        self.m.pose[57] = 0
        self.m.pose[58] = 0#np.pi/4 #right elbow flexsion/extension KEEP POSITIVE

        self.m.pose[60] = 0 #left wrist roll

        self.m.pose[63] = 0 #right wrist roll
        #self.m.pose[65] = np.pi/5

        self.m.pose[66] = 0 #left hand roll

        self.m.pose[69] = 0 #right hand roll
        #self.m.pose[71] = np.pi/5 #right fist


        mu = 0



        for i in range(10):
            if sampling == "NORMAL":
                self.m.betas[i] = random.normalvariate(mu, sigma)
            elif sampling == "UNIFORM":
                self.m.betas[i]  = np.random.uniform(-one_side_range, one_side_range)
        #self.m.betas[0] = random.normalvariate(mu, sigma) #overall body size. more positive number makes smaller, negative makes larger with bigger belly
        #self.m.betas[1] = random.normalvariate(mu, sigma) #positive number makes person very skinny, negative makes fat
        #self.m.betas[2] = random.normalvariate(mu, sigma) #muscle mass. higher makes person less physically fit
        #self.m.betas[3] = random.normalvariate(mu, sigma) #proportion for upper vs lower bone lengths. more negative number makes legs much bigger than arms
        #self.m.betas[4] = random.normalvariate(mu, sigma) #neck. more negative seems to make neck longer and body more skinny
        #self.m.betas[5] = random.normalvariate(mu, sigma) #size of hips. larger means bigger hips
        #self.m.betas[6] = random.normalvariate(mu, sigma) #proportion of belly with respect to rest of the body. higher number is larger belly
        #self.m.betas[7] = random.normalvariate(mu, sigma)
        #self.m.betas[8] = random.normalvariate(-3, 3)
        #self.m.betas[9] = random.normalvariate(-3, 3)

        #print self.m.pose.shape
        #print self.m.pose, 'pose'
        #print self.m.betas, 'betas'

        ## Create OpenDR renderer
        self.rn = ColoredRenderer()

        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.scene = pyrender.Scene()
        self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 1.0, 0.0])
        #166, 206, 227
        #31, 120, 180
        #178, 223, 138
        #51, 160, 44
        #251, 154, 153
        #227, 26, 28
        #253, 191, 111
        #255, 127, 0
        #202, 178, 214
        #106, 61, 154
        self.mesh_parts_mat_list = [pyrender.MetallicRoughnessMaterial(baseColorFactor=[166./255., 206./255., 227./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[31./255., 120./255., 180./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[251./255., 154./255., 153./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[227./255., 26./255., 28./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[178./255., 223./255., 138./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[51./255., 160./255., 44./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[253./255., 191./255., 111./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[255./255., 127./255., 0./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[202./255., 178./255., 214./255., 0.0]),
                                   pyrender.MetallicRoughnessMaterial(baseColorFactor=[106./255., 61./255., 154./255., 0.0])]

        self.artag_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 1.0, 0.3, 0.5])
        self.artag_mat_other = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 0.0])
        self.artag_r = np.array(
            [[-0.055, -0.055, 0.0], [-0.055, 0.055, 0.0], [0.055, -0.055, 0.0], [0.055, 0.055, 0.0]])
        self.artag_f = np.array([[0, 1, 3], [3, 1, 0], [0, 2, 3], [3, 2, 0], [1, 3, 2]])
        self.artag_facecolors_root = np.array(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        self.artag_facecolors = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], ])


    def get_vtx_idx_and_faces(self):

        self.m.betas[:] = np.random.rand(self.m.betas.size) * .0

        self.m.pose[:] = np.random.rand(self.m.pose.size) * .0

        smpl_verts = np.array(self.m.r)
        smpl_faces = np.array(self.m.f)

        l_lowerleg_idx_list = []
        r_lowerleg_idx_list = []
        l_upperleg_idx_list = []
        r_upperleg_idx_list = []
        l_forearm_idx_list = []
        r_forearm_idx_list = []
        l_upperarm_idx_list = []
        r_upperarm_idx_list = []
        head_idx_list = []
        torso_idx_list = []


        print "assigning vertices and indices..."
        for idx in range(6890):
            if smpl_verts[idx, 1] < float(self.m.J_transformed[4, 1]) and smpl_verts[idx, 0] > 0:
                l_lowerleg_idx_list.append(idx)
            elif smpl_verts[idx, 1] < float(self.m.J_transformed[5, 1]) and smpl_verts[idx, 0] < 0:
                r_lowerleg_idx_list.append(idx)
            elif smpl_verts[idx, 1] < float(self.m.J_transformed[1, 1]) and smpl_verts[idx, 1] > float(self.m.J_transformed[4, 1]) and smpl_verts[idx, 0] > 0:
                l_upperleg_idx_list.append(idx)
            elif smpl_verts[idx, 1] < float(self.m.J_transformed[2, 1]) and smpl_verts[idx, 1] > float(self.m.J_transformed[5, 1]) and smpl_verts[idx, 0] < 0:
                r_upperleg_idx_list.append(idx)

            elif smpl_verts[idx, 0] > float(self.m.J_transformed[18, 0]):
                l_forearm_idx_list.append(idx)
            elif smpl_verts[idx, 0] < float(self.m.J_transformed[19, 0]):
                r_forearm_idx_list.append(idx)

            elif smpl_verts[idx, 0] > float(self.m.J_transformed[16, 0]) and smpl_verts[idx, 0] < float(self.m.J_transformed[18, 0]):
                l_upperarm_idx_list.append(idx)
            elif smpl_verts[idx, 0] < float(self.m.J_transformed[17, 0]) and smpl_verts[idx, 0] > float(self.m.J_transformed[19, 0]):
                r_upperarm_idx_list.append(idx)

            elif smpl_verts[idx, 1] > np.mean([float(self.m.J_transformed[12, 1]), float(self.m.J_transformed[15, 1])]):
                head_idx_list.append(idx)

            elif smpl_verts[idx, 1] < np.mean([float(self.m.J_transformed[12, 1]), float(self.m.J_transformed[15, 1])]) and \
                smpl_verts[idx, 0] < float(self.m.J_transformed[16, 0]) and \
                smpl_verts[idx, 0] > float(self.m.J_transformed[17, 0]) and \
                smpl_verts[idx, 1] > float(self.m.J_transformed[1, 1]) and \
                smpl_verts[idx, 1] > float(self.m.J_transformed[2, 1]):
                torso_idx_list.append(idx)


        print "assigning faces..."
        l_lowerleg_face_list = []
        r_lowerleg_face_list = []
        l_upperleg_face_list = []
        r_upperleg_face_list = []
        l_forearm_face_list = []
        r_forearm_face_list = []
        l_upperarm_face_list = []
        r_upperarm_face_list = []
        head_face_list = []
        torso_face_list = []
        for face_idx in range(13776):
            print face_idx,
            #print smpl_faces[face_idx, :], l_lowerleg_idx_list
            if all(i in l_lowerleg_idx_list for i in smpl_faces[face_idx, :]):
                l_lowerleg_face_list.append([l_lowerleg_idx_list.index(smpl_faces[face_idx, 0]),
                                            l_lowerleg_idx_list.index(smpl_faces[face_idx, 1]),
                                            l_lowerleg_idx_list.index(smpl_faces[face_idx, 2])])
            elif all(i in r_lowerleg_idx_list for i in smpl_faces[face_idx, :]):
                r_lowerleg_face_list.append([r_lowerleg_idx_list.index(smpl_faces[face_idx, 0]),
                                            r_lowerleg_idx_list.index(smpl_faces[face_idx, 1]),
                                            r_lowerleg_idx_list.index(smpl_faces[face_idx, 2])])
            elif all(i in l_upperleg_idx_list for i in smpl_faces[face_idx, :]):
                l_upperleg_face_list.append([l_upperleg_idx_list.index(smpl_faces[face_idx, 0]),
                                            l_upperleg_idx_list.index(smpl_faces[face_idx, 1]),
                                            l_upperleg_idx_list.index(smpl_faces[face_idx, 2])])
            elif all(i in r_upperleg_idx_list for i in smpl_faces[face_idx, :]):
                r_upperleg_face_list.append([r_upperleg_idx_list.index(smpl_faces[face_idx, 0]),
                                            r_upperleg_idx_list.index(smpl_faces[face_idx, 1]),
                                            r_upperleg_idx_list.index(smpl_faces[face_idx, 2])])

            elif all(i in l_forearm_idx_list for i in smpl_faces[face_idx, :]):
                l_forearm_face_list.append([l_forearm_idx_list.index(smpl_faces[face_idx, 0]),
                                            l_forearm_idx_list.index(smpl_faces[face_idx, 1]),
                                            l_forearm_idx_list.index(smpl_faces[face_idx, 2])])
            elif all(i in r_forearm_idx_list for i in smpl_faces[face_idx, :]):
                r_forearm_face_list.append([r_forearm_idx_list.index(smpl_faces[face_idx, 0]),
                                            r_forearm_idx_list.index(smpl_faces[face_idx, 1]),
                                            r_forearm_idx_list.index(smpl_faces[face_idx, 2])])
            elif all(i in l_upperarm_idx_list for i in smpl_faces[face_idx, :]):
                l_upperarm_face_list.append([l_upperarm_idx_list.index(smpl_faces[face_idx, 0]),
                                            l_upperarm_idx_list.index(smpl_faces[face_idx, 1]),
                                            l_upperarm_idx_list.index(smpl_faces[face_idx, 2])])
            elif all(i in r_upperarm_idx_list for i in smpl_faces[face_idx, :]):
                r_upperarm_face_list.append([r_upperarm_idx_list.index(smpl_faces[face_idx, 0]),
                                            r_upperarm_idx_list.index(smpl_faces[face_idx, 1]),
                                            r_upperarm_idx_list.index(smpl_faces[face_idx, 2])])

            elif all(i in head_idx_list for i in smpl_faces[face_idx, :]):
                head_face_list.append([head_idx_list.index(smpl_faces[face_idx, 0]),
                                        head_idx_list.index(smpl_faces[face_idx, 1]),
                                        head_idx_list.index(smpl_faces[face_idx, 2])])

            elif all(i in torso_idx_list for i in smpl_faces[face_idx, :]):
                torso_face_list.append([torso_idx_list.index(smpl_faces[face_idx, 0]),
                                        torso_idx_list.index(smpl_faces[face_idx, 1]),
                                        torso_idx_list.index(smpl_faces[face_idx, 2])])

        segmented_dict = {}
        segmented_dict['l_lowerleg_idx_list'] = l_lowerleg_idx_list
        segmented_dict['r_lowerleg_idx_list'] = r_lowerleg_idx_list
        segmented_dict['l_upperleg_idx_list'] = l_upperleg_idx_list
        segmented_dict['r_upperleg_idx_list'] = r_upperleg_idx_list
        segmented_dict['l_forearm_idx_list'] = l_forearm_idx_list
        segmented_dict['r_forearm_idx_list'] = r_forearm_idx_list
        segmented_dict['l_upperarm_idx_list'] = l_upperarm_idx_list
        segmented_dict['r_upperarm_idx_list'] = r_upperarm_idx_list
        segmented_dict['head_idx_list'] = head_idx_list
        segmented_dict['torso_idx_list'] = torso_idx_list
        segmented_dict['l_lowerleg_face_list'] = l_lowerleg_face_list
        segmented_dict['r_lowerleg_face_list'] = r_lowerleg_face_list
        segmented_dict['l_upperleg_face_list'] = l_upperleg_face_list
        segmented_dict['r_upperleg_face_list'] = r_upperleg_face_list
        segmented_dict['l_forearm_face_list'] = l_forearm_face_list
        segmented_dict['r_forearm_face_list'] = r_forearm_face_list
        segmented_dict['l_upperarm_face_list'] = l_upperarm_face_list
        segmented_dict['r_upperarm_face_list'] = r_upperarm_face_list
        segmented_dict['head_face_list'] = head_face_list
        segmented_dict['torso_face_list'] = torso_face_list

        pkl.dump(segmented_dict, open(os.path.join('./segmented_mesh_idx_faces.p'), 'wb'))

        return segmented_dict



    def render_mesh(self):

        #get SMPL mesh
        smpl_verts = np.array(self.m.r)
        smpl_faces = np.array(self.m.f)

        #self.segmented_dict = self.get_vtx_idx_and_faces()
        self.segmented_dict = load_pickle('segmented_mesh_idx_faces.p')

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

        #print np.min(human_mesh_vtx_parts[0][:, 0]), np.max(human_mesh_vtx_parts[0][:, 0])
        #print np.min(human_mesh_vtx_parts[0][:, 1]), np.max(human_mesh_vtx_parts[0][:, 1])
        #print np.min(human_mesh_vtx_parts[0][:, 2]), np.max(human_mesh_vtx_parts[0][:, 2])
        #for idx in range(np.shape(human_mesh_vtx_parts[0])[0]):
        #    if human_mesh_vtx_parts[0][idx, 0] > -0.11 and human_mesh_vtx_parts[0][idx, 0] < -0.08:
        #        if human_mesh_vtx_parts[0][idx, 1] < -0.2 and human_mesh_vtx_parts[0][idx, 1] > -0.5:
        #            if human_mesh_vtx_parts[0][idx, 2] > 0:
       #                 print idx, human_mesh_vtx_parts[0][idx, :]


        #throw_out = 1350 #heart
        #throw_out = 3166 #mouth
        #throw_out = 6566 #appendix

        #for idx in range(np.shape(human_mesh_face_parts[0])[0]):
        #    #print human_mesh_face_parts[9][idx]
        #    if human_mesh_face_parts[0][idx][0] == throw_out or human_mesh_face_parts[0][idx][1] == throw_out or human_mesh_face_parts[0][idx][2] == throw_out:
        #        human_mesh_face_parts[0][idx][0] = 0
        #        human_mesh_face_parts[0][idx][1] = 1
        #        human_mesh_face_parts[0][idx][2] = 2


        tm_list = []
        for idx in range(len(human_mesh_vtx_parts)):
            tm_list.append(trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_parts[idx]), faces = np.array(human_mesh_face_parts[idx])))

        #smpl_tm = trimesh.base.Trimesh(vertices=smpl_verts, faces=smpl_faces)

        mesh_list = []
        for idx in range(len(tm_list)):
            if len(tm_list) == 1:
                mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[idx], material = self.human_mat, wireframe = True))
            else:
                mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[idx], material = self.mesh_parts_mat_list[idx], wireframe = True))


        #print "Viewing"
        if self.first_pass == True:

            for mesh_part in mesh_list:
                self.scene.add(mesh_part)

            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
            self.first_pass = False

            self.node_list = []
            for mesh_part in mesh_list:
                for node in self.scene.get_nodes(obj=mesh_part):
                    self.node_list.append(node)
        else:
            self.viewer.render_lock.acquire()

            #reset the human mesh

            for idx in range(len(mesh_list)):

                self.scene.remove_node(self.node_list[idx])
                self.scene.add(mesh_list[idx])
                for node in self.scene.get_nodes(obj=mesh_list[idx]):
                    self.node_list[idx] = node

            #print self.scene.get_nodes()
            self.viewer.render_lock.release()



if __name__ == "__main__":
    generator = GeneratePose(sampling = "UNIFORM", sigma = 0, one_side_range = 0)

    #while True:
    generator.render_mesh()