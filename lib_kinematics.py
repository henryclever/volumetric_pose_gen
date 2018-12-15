import numpy as np

#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from ikpy import geometry_utils

INTER_SENSOR_DISTANCE = 0.0286#metres




class JointLimbFiller():
    def __init__(self):

        self.T = np.zeros((4, 1))
        self.H = np.zeros((4, 1))
        self.N = np.zeros((4, 1))
        self.B = np.zeros((4, 1))

        self.r_S = np.zeros((4, 1))
        self.r_E = np.zeros((4, 1))
        self.r_H = np.zeros((4, 1))
        self.r_G = np.zeros((4, 1))
        self.r_K = np.zeros((4, 1))
        self.r_A = np.zeros((4, 1))

        self.l_S = np.zeros((4, 1))
        self.l_E = np.zeros((4, 1))
        self.l_H = np.zeros((4, 1))
        self.l_G = np.zeros((4, 1))
        self.l_K = np.zeros((4, 1))
        self.l_A = np.zeros((4, 1))

        self.pseudoheight = {'NTTQ6': 1.53, 'FMNGQ': 1.42, 'WM9KJ': 1.52, 'WFGW9': 1.63, 'TX887': 1.66, \
                             'ZV7TE': 1.59, 'GRTJK': 1.49, '40ESJ': 1.53, \
                             'TSQNA': 1.69, 'RQCLC': 1.58, 'A4G4Y': 1.64, '1YJAG': 1.45, '4ZZZQ': 1.58, \
                             'G55Q1': 1.67, 'GF5Q3': 1.63, 'WCNOM': 1.48, 'WE2SZ': 1.43, '5LDJG': 1.54}

    def get_lengths_pseudotargets(self, targets, subject, bedangle):
        self.H[0, 0] = targets[0, 0]
        self.H[1, 0] = targets[0, 1]
        self.H[2, 0] = targets[0, 2]
        self.H[3, 0] = 1.

        self.T[0, 0] = targets[1, 0]
        self.T[1, 0] = targets[1, 1]
        self.T[2, 0] = targets[1, 2]
        self.T[3, 0] = 1.

        self.r_E[0, 0] = targets[2, 0]
        self.r_E[1, 0] = targets[2, 1]
        self.r_E[2, 0] = targets[2, 2]
        self.r_E[3, 0] = 1.

        self.l_E[0, 0] = targets[3, 0]
        self.l_E[1, 0] = targets[3, 1]
        self.l_E[2, 0] = targets[3, 2]
        self.l_E[3, 0] = 1.

        self.r_H[0, 0] = targets[4, 0]
        self.r_H[1, 0] = targets[4, 1]
        self.r_H[2, 0] = targets[4, 2]
        self.r_H[3, 0] = 1.

        self.l_H[0, 0] = targets[5, 0]
        self.l_H[1, 0] = targets[5, 1]
        self.l_H[2, 0] = targets[5, 2]
        self.l_H[3, 0] = 1.

        self.r_K[0, 0] = targets[6, 0]
        self.r_K[1, 0] = targets[6, 1]
        self.r_K[2, 0] = targets[6, 2]
        self.r_K[3, 0] = 1.

        self.l_K[0, 0] = targets[7, 0]
        self.l_K[1, 0] = targets[7, 1]
        self.l_K[2, 0] = targets[7, 2]
        self.l_K[3, 0] = 1.

        self.r_A[0, 0] = targets[8, 0]
        self.r_A[1, 0] = targets[8, 1]
        self.r_A[2, 0] = targets[8, 2]
        self.r_A[3, 0] = 1.

        self.l_A[0, 0] = targets[9, 0]
        self.l_A[1, 0] = targets[9, 1]
        self.l_A[2, 0] = targets[9, 2]
        self.l_A[3, 0] = 1.

        pseudotargets = np.zeros((5, 3))

        lengths = {}
        lengths['torso_spine1'] = 0.10
        lengths['spine1_neck'] = 0.2065 * self.pseudoheight[str(subject)] - 0.0529  # about 0.25. torso vert
        lengths['neck_l_shoulder'] = 0.13454 * self.pseudoheight[str(subject)] - 0.03547  # about 0.15. shoulder left
        lengths['neck_r_shoulder'] = 0.13454 * self.pseudoheight[str(subject)] - 0.03547  # about 0.15. shoulder right

        # here we construct pseudo ground truths for the neck and shoulders by making fixed translations from the torso
        self.N[0, 0] = self.T[0, 0]
        self.N[1, 0] = self.T[1, 0] + lengths['spine1_neck'] * np.cos(np.deg2rad(bedangle * 0.75))
        self.N[2, 0] = self.T[2, 0] - lengths['torso_spine1'] + lengths['spine1_neck'] * np.sin(np.deg2rad(bedangle * 0.75))
        self.N[3, 0] = 1

        self.r_S[0, 0] = self.T[0, 0] - lengths['neck_r_shoulder']
        self.r_S[1, 0] = self.T[1, 0] + lengths['spine1_neck'] * np.cos(np.deg2rad(bedangle * 0.75))
        self.r_S[2, 0] = self.T[2, 0] - lengths['torso_spine1'] + lengths['spine1_neck'] * np.sin(np.deg2rad(bedangle * 0.75))
        self.r_S[3, 0] = 1

        self.l_S[0, 0] = self.T[0, 0] + lengths['neck_l_shoulder']
        self.l_S[1, 0] = self.T[1, 0] + lengths['spine1_neck'] * np.cos(np.deg2rad(bedangle * 0.75))
        self.l_S[2, 0] = self.T[2, 0] - lengths['torso_spine1'] + lengths['spine1_neck'] * np.sin(np.deg2rad(bedangle * 0.75))
        self.l_S[3, 0] = 1

        pseudotargets[0, 0] = self.N[0, 0]
        pseudotargets[0, 1] = self.N[1, 0]
        pseudotargets[0, 2] = self.N[2, 0]

        pseudotargets[1, 0] = self.r_S[0, 0]
        pseudotargets[1, 1] = self.r_S[1, 0]
        pseudotargets[1, 2] = self.r_S[2, 0]

        pseudotargets[2, 0] = self.l_S[0, 0]
        pseudotargets[2, 1] = self.l_S[1, 0]
        pseudotargets[2, 2] = self.l_S[2, 0]

        # get the length of the right shoulder to right elbow
        lengths['l_shoulder_elbow'] = np.linalg.norm(self.l_E - self.l_S)
        lengths['r_shoulder_elbow'] = np.linalg.norm(self.r_E - self.r_S)

        # parameter for the length between hand and elbow. Should be around 0.2 meters.
        lengths['l_elbow_wrist'] = np.linalg.norm(self.l_H - self.l_E)
        lengths['r_elbow_wrist'] = np.linalg.norm(self.r_H - self.r_E)

        # get the length between the neck and head
        lengths['neck_head'] = np.linalg.norm(self.H - self.N)


        lengths['torso_spine2'] = 0.14
        lengths['spine2_tail'] = 0.1549 * self.pseudoheight[str(subject)] - 0.03968  # about 0.25. torso vert
        lengths['tail_l_glute'] = 0.08072 * self.pseudoheight[
            str(subject)] - 0.02128  # Equal to 0.6 times the equivalent neck to shoulder. glute left
        lengths['tail_r_glute'] = 0.08072 * self.pseudoheight[
            str(subject)] - 0.02128  # Equal to 0.6 times the equivalent neck to shoulder. glute right

        self.B[0, 0] = self.T[0, 0]
        self.B[1, 0] = self.T[1, 0] - lengths['spine2_tail'] * np.cos(np.deg2rad(bedangle * 0.6))
        self.B[2, 0] = self.T[2, 0] - lengths['torso_spine2'] - lengths['spine2_tail'] * np.sin(np.deg2rad(bedangle * 0.6))
        self.B[3, 0] = 1

        # here we construct pseudo ground truths for the shoulders by making fixed translations from the torso
        self.r_G[0, 0] = self.T[0, 0] - lengths['tail_r_glute']
        self.r_G[1, 0] = self.T[1, 0] - lengths['spine2_tail'] * np.cos(np.deg2rad(bedangle * 0.6))
        self.r_G[2, 0] = self.T[2, 0] - lengths['torso_spine2'] - lengths['spine2_tail'] * np.sin(np.deg2rad(bedangle * 0.6))
        self.r_G[3, 0] = 1

        self.l_G[0, 0] = self.T[0, 0] + lengths['tail_l_glute']
        self.l_G[1, 0] = self.T[1, 0] - lengths['spine2_tail'] * np.cos(np.deg2rad(bedangle * 0.6))
        self.l_G[2, 0] = self.T[2, 0] - lengths['torso_spine2'] - lengths['spine2_tail'] * np.sin(np.deg2rad(bedangle * 0.6))
        self.l_G[3, 0] = 1

        pseudotargets[3, 0] = self.r_G[0, 0]
        pseudotargets[3, 1] = self.r_G[1, 0]
        pseudotargets[3, 2] = self.r_G[2, 0]

        pseudotargets[4, 0] = self.l_G[0, 0]
        pseudotargets[4, 1] = self.l_G[1, 0]
        pseudotargets[4, 2] = self.l_G[2, 0]

        # get the length of the right shoulder to right elbow
        lengths['r_glute_knee'] = np.linalg.norm(self.r_K - self.r_G)
        lengths['l_glute_knee'] = np.linalg.norm(self.l_K - self.l_G)

        # parameter for the length between hand and elbow. Should be around 0.2 meters.
        lengths['r_knee_ankle'] = np.linalg.norm(self.r_A - self.r_K)
        lengths['l_knee_ankle'] = np.linalg.norm(self.l_A - self.l_K)

        return lengths, pseudotargets


def world_to_mat(w_data, p_world_mat, R_world_mat):
    '''Converts a vector in the world frame to a vector in the map frame.
    Depends on the calibration of the MoCap room. Be sure to change this
    when the calibration file changes. This function mainly helps in
    visualizing the joint coordinates on the pressure mat.
    Input: w_data: which is a 3 x 1 vector in the world frame'''
    # The homogenous transformation matrix from world to mat
    # O_m_w = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    O_m_w = np.matrix(np.reshape(R_world_mat, (3, 3)))
    p_mat_world = O_m_w.dot(-np.asarray(p_world_mat))
    B_m_w = np.concatenate((O_m_w, p_mat_world.T), axis=1)
    last_row = np.array([[0, 0, 0, 1]])
    B_m_w = np.concatenate((B_m_w, last_row), axis=0)

    w_data = np.hstack([w_data, np.ones([len(w_data), 1])])

    # Convert input to the mat frame vector
    m_data = B_m_w * w_data.T
    m_data = np.squeeze(np.asarray(m_data[:3, :].T))

    try:
        m_data[:, 0] = m_data[:, 0] + 10 * INTER_SENSOR_DISTANCE
        m_data[:, 1] = m_data[:, 1] + 10 * INTER_SENSOR_DISTANCE
    except:
        m_data[0] = m_data[0] + 10 * INTER_SENSOR_DISTANCE
        m_data[1] = m_data[1] + 10 * INTER_SENSOR_DISTANCE

    return m_data

def ikpy_leg_analytic(origins, current):
    lengths = np.zeros((4))  # torso height, glute right, thigh right, calf right
    angles = np.zeros((4))  # hip right roll, hip right pitch,  hip right yaw, knee right

    lengths[0] = 0#0.14  # torso height
    # lengths[1] = 0.2065 * self.pseudoheight[str(subject)] - 0.0529  # about 0.25. torso vert
    # .85 is .0354, .0241
    # .8 is .0328, .0239
    # .75 is .0314, .0239

    T = np.zeros((4, 2))
    r_G = np.zeros((4, 2))
    r_K = np.zeros((4, 2))
    r_A = np.zeros((4, 2))
    Pr_GK = np.zeros((4, 2))
    Pr_KA = np.zeros((4, 2))
    r_GK = np.zeros((3, 2))
    r_GKn = np.zeros((3, 2))
    r_GA = np.zeros((3, 1))
    r_GAn = np.zeros((3, 1))
    r_Gpxn = np.zeros((3, 1))
    r_Gpxm = np.zeros((3, 1))
    r_Gpxzn = np.zeros((3, 1))
    r_Gpx = np.zeros((3, 1))
    r_GA_pp = np.zeros((3, 1))
    r_Gpx_pp = np.zeros((3, 1))

    T[0:3, 0] = origins[0, :].T
    T[0:3, 1] = current[0, :].T
    T[3, 0:2] = 1.

    r_G[0:3, 0] = origins[1, :].T
    r_G[0:3, 1] = current[1, :].T
    r_G[3, 0:2] = 1

    r_K[0:3, 0] = origins[2, :].T
    r_K[0:3, 1] = current[2, :].T
    r_K[3, 0:2] = 1.

    r_A[0:3, 0] = origins[3, :].T
    r_A[0:3, 1] = current[3, :].T
    r_A[3, 0:2] = 1.


    lengths[1] = np.linalg.norm(r_G[:, 0] - T[:, 0])
    lengths[2] = np.linalg.norm(r_K[:, 0] - r_G[:, 0])
    lengths[3] = np.linalg.norm(r_A[:, 0] - r_K[:, 0])
    print(lengths, 'lengths')


    rGK_mag = np.copy(lengths[2])
    r_GK = r_G[0:3, :] - r_K[0:3, :]
    if rGK_mag > 0: r_GKn = np.copy(r_GK) / rGK_mag
    print(r_GKn)

    print(np.arcsin(r_GKn[2, 0]))
    print(np.arcsin(r_GKn[2, 1]))
    angles[1] =(-np.arcsin(r_GKn[2, 0])+np.arcsin(r_GKn[2, 1]))

    print(angles, 'angles')


def ikpy_leg(origins, current):

    knee_chain = Chain(name='knee', links=[
        OriginLink(),
        URDFLink(name="hip_1", translation_vector=origins[1, :] - origins[0, :], orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="hip_3", translation_vector=[.0, 0, 0], orientation=[0, 0, 0], rotation=[0, 0, 1], ),
        URDFLink(name="knee", translation_vector=origins[2, :] - origins[1, :], orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(0, np.pi), ), ]
                             )

    Knee_Pos = current[2, :] - current[0, :]
    IK_K = knee_chain.inverse_kinematics([
        [1., 0., 0., Knee_Pos[0]],
        [0., 1., 0., Knee_Pos[1]],
        [0., 0., 1., Knee_Pos[2]],
        [0., 0., 0., 1.]])


    ankle_chain = Chain(name='ankle', links=[
        OriginLink(),
        URDFLink(name="hip_1", translation_vector=origins[1, :] - origins[0, :],
                 orientation=[IK_K[1], 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi), ),
        URDFLink(name="hip_2", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, IK_K[2]], rotation=[0, 0, 1], bounds=(-np.pi, np.pi), ),
        URDFLink(name="hip_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-np.pi, np.pi), ),
        URDFLink(name="knee", translation_vector=origins[2, :] - origins[1, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(0, np.pi), ),
        URDFLink(name="ankle", translation_vector=origins[3, :] - origins[2, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi), ), ],
                              active_links_mask=[False, False, False, True, True, False]
                              )

    Ankle_Pos = current[3, :] - current[0, :]
    IK_A = ankle_chain.inverse_kinematics([
        [1., 0., 0., Ankle_Pos[0]],
        [0., 1., 0., Ankle_Pos[1]],
        [0., 0., 1., Ankle_Pos[2]],
        [0., 0., 0., 1.]])

    hip_1 = geometry_utils.Rx_matrix(IK_K[1])
    hip_2 = geometry_utils.Rz_matrix(IK_K[2])
    hip_3 = geometry_utils.Ry_matrix(IK_A[3])
    #print hip_1, 'hip 1'
    #print hip_2, 'hip 2'
    #print hip_3, 'hip 3'

    rot_hip = hip_1.dot(hip_2).dot(hip_3)
    Tr_hip = rot_hip[0,0] + rot_hip[1,1] + rot_hip[2,2]
    theta_inv_hip = np.arccos((Tr_hip - 1)/2)
    omega_hip = np.array([rot_hip[2,1]-rot_hip[1,2], rot_hip[0,2]-rot_hip[2,0], rot_hip[1,0]-rot_hip[0,1]])
    omega_hip = (1/(2*np.sin(theta_inv_hip)))*omega_hip

    #rot_knee = geometry_utils.Rx_matrix(IK_A[4])
    #print(rot_knee)
    #Tr_knee = rot_knee[0,0] + rot_knee[1,1] + rot_knee[2,2]
    #print Tr_knee, 'Tr knee'
    #theta_inv_knee = np.arccos((Tr_knee - 1)/2)
    #print theta_inv_knee, 'th inv knee'
    #omega_knee = np.array([rot_knee[2,1]-rot_knee[1,2], rot_knee[0,2]-rot_knee[2,0], rot_knee[1,0]-rot_knee[0,1]])
    #print omega_knee, 'omega knee'
    #omega_knee = (1/(2*np.sin(theta_inv_knee)))*omega_knee


    return omega_hip, knee_chain, IK_K, ankle_chain, IK_A



def ikpy_arm(origin, current):


    #shoulder_rot1 = (origin[3, :] - origin[2, :])/np.linalg.norm(origin[3, :] - origin[2, :])
    #shoulder_rot1 = list(shoulder_rot1)


    elbow_chain = Chain(name='elbow', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=origin[1, :] - origin[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi),),
        URDFLink(name="shoulder_1", translation_vector=origin[2, :] - origin[1, :],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], bounds=(-np.pi, np.pi),),
        URDFLink(name="shoulder_2", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-np.pi, np.pi),),
        URDFLink(name="elbow", translation_vector=origin[3, :] - origin[2, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-np.pi, np.pi),), ],
                              active_links_mask=[False, False, True, True, False]
                              )


    Elbow_Pos = current[3, :] - current[0, :]
    #print current[3, :] - current[0, :]
    #print origin[3, :] - origin[0, :], 'origin'
    IK_E = elbow_chain.inverse_kinematics([
        [1., 0., 0., Elbow_Pos[0]],
        [0., 1., 0., Elbow_Pos[1]],
        [0., 0., 1., Elbow_Pos[2]],
        [0., 0., 0., 1.]])

    #print IK_RE, 'RE'
    #IK_E *= 0

    wrist_chain = Chain(name='wrist', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=origin[1, :] - origin[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi),),
        URDFLink(name="shoulder_1", translation_vector=origin[2, :] - origin[1, :],
                 orientation=[0, 0, IK_E[2]], rotation=[0, 0, 1], bounds=(-np.pi, np.pi),),
        URDFLink(name="shoulder_2", translation_vector=[0, 0, 0],
                 orientation=[0, IK_E[3], 0], rotation=[0, 1, 0], bounds=(-np.pi, np.pi),),
        URDFLink(name="shoulder_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi),),
        URDFLink(name="elbow", translation_vector=origin[3, :] - origin[2, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-np.pi, np.pi),),
        URDFLink(name="wrist", translation_vector=origin[4, :] - origin[3, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi),), ],
                              active_links_mask=[False, False, False, False, True, True, False]
                              )

    Wrist_Pos = current[4, :] - current[0, :]
    IK_W = wrist_chain.inverse_kinematics([
        [1., 0., 0., Wrist_Pos[0]],
        [0., 1., 0., Wrist_Pos[1]],
        [0., 0., 1., Wrist_Pos[2]],
        [0., 0., 0., 1.]])

    #IK_W *= 0
   # print IK_RW, 'RW'

    shoulder_1 = geometry_utils.Rz_matrix(IK_E[2])
    shoulder_2 = geometry_utils.Ry_matrix(IK_E[3])
    shoulder_3 = geometry_utils.Rx_matrix(IK_W[4])
    #print shoulder_1, 'shoulder 1'
    #print shoulder_2, 'shoulder 2'
    #print shoulder_3, 'shoulder 3'

    rot_shoulder = shoulder_1.dot(shoulder_2).dot(shoulder_3)
    Tr_shoulder = rot_shoulder[0, 0] + rot_shoulder[1, 1] + rot_shoulder[2, 2]
    theta_inv_shoulder = np.arccos((Tr_shoulder - 1) / 2)
    #print(theta_inv_shoulder, 'theta inv should')
    omega_shoulder = np.array([rot_shoulder[2, 1] - rot_shoulder[1, 2], rot_shoulder[0, 2] - rot_shoulder[2, 0], rot_shoulder[1, 0] - rot_shoulder[0, 1]])
    omega_shoulder = (1 / (2 * np.sin(theta_inv_shoulder))) * omega_shoulder * theta_inv_shoulder

    #IK_W[2] = 0#np.pi/4
    #IK_W[4] = -np.pi/2
    #IK_W[5] = -np.pi/4

    return omega_shoulder, elbow_chain, IK_E, wrist_chain, IK_W


def ikpy_head(origin, current):

    head_chain = Chain(name='elbow', links=[
        OriginLink(),
        URDFLink(name="neck_1", translation_vector=origin[1, :] - origin[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi),),
        URDFLink(name="neck_2", translation_vector=[0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], bounds=(-np.pi, np.pi),),
        URDFLink(name="head", translation_vector=origin[2, :] - origin[1, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi),), ],
                              active_links_mask=[False, True, True, False]
                              )


    Head_Pos = current[2, :] - current[0, :]
    #print current[3, :] - current[0, :]
    #print origin[3, :] - origin[0, :], 'origin'
    IK_H = head_chain.inverse_kinematics([
        [1., 0., 0., Head_Pos[0]],
        [0., 1., 0., Head_Pos[1]],
        [0., 0., 1., Head_Pos[2]],
        [0., 0., 0., 1.]])


    neck_1 = geometry_utils.Rx_matrix(IK_H[1])
    neck_2 = geometry_utils.Rz_matrix(IK_H[2])
    #print neck_1, 'neck 1'
    #print neck_2, 'neck 2'

    rot_neck = neck_1.dot(neck_2)
    Tr_neck = rot_neck[0, 0] + rot_neck[1, 1] + rot_neck[2, 2]
    theta_inv_neck = np.arccos((Tr_neck - 1) / 2)
    print(theta_inv_neck, 'theta inv neck')
    omega_neck = np.array([rot_neck[2, 1] - rot_neck[1, 2], rot_neck[0, 2] - rot_neck[2, 0], rot_neck[1, 0] - rot_neck[0, 1]])
    omega_neck = (1 / (2 * np.sin(theta_inv_neck))) * omega_neck * theta_inv_neck

    return omega_neck, head_chain, IK_H