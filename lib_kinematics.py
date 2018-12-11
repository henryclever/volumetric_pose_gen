import numpy as np

#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
from ikpy import geometry_utils


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
    Tr = rot_hip[0,0] + rot_hip[1,1] + rot_hip[2,2]
    theta_inv = np.arccos((Tr - 1)/2)
    #print(theta_inv)
    omega = np.array([rot_hip[2,1]-rot_hip[1,2], rot_hip[0,2]-rot_hip[2,0], rot_hip[1,0]-rot_hip[0,1]])
    omega = (1/(2*np.sin(theta_inv)))*omega
    print(omega*theta_inv, 'axis angle solution')



    return knee_chain, IK_K, ankle_chain, IK_A



def ikpy_arm(origin, current):


    right_elbow_chain = Chain(name='right_elbow', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=origin[1, :] - origin[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_shoulder_1", translation_vector=origin[2, :] - origin[1, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_shoulder_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], ),
        URDFLink(name="r_elbow", translation_vector=origin[3, :] - origin[2, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ), ],
                              active_links_mask=[False, False, True, True, False]
                              )


    R_Elbow_Pos = current[3, :] - current[0, :]
    #print current[3, :] - current[0, :]
    #print origin[3, :] - origin[0, :], 'origin'
    IK_RE = right_elbow_chain.inverse_kinematics([
        [1., 0., 0., R_Elbow_Pos[0]],
        [0., 1., 0., R_Elbow_Pos[1]],
        [0., 0., 1., R_Elbow_Pos[2]],
        [0., 0., 0., 1.]])

    #print IK_RE, 'RE'

    right_wrist_chain = Chain(name='right_wrist', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=origin[1, :] - origin[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_shoulder_1", translation_vector=origin[2, :] - origin[1, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_shoulder_2", translation_vector=[.0, 0, 0],
                 orientation=[0, IK_RE[2], 0], rotation=[0, 1, 0], ),
        URDFLink(name="r_shoulder_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, IK_RE[3]], rotation=[0, 0, 1], ),
        URDFLink(name="r_elbow", translation_vector=origin[3, :] - origin[2, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ),
        URDFLink(name="r_wrist", translation_vector=origin[4, :] - origin[3, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ), ],
                              active_links_mask=[False, False, True, False, False, True, False]
                              )

    R_Wrist_Pos = current[4, :] - current[0, :]
    IK_RW = right_wrist_chain.inverse_kinematics([
        [1., 0., 0., R_Wrist_Pos[0]],
        [0., 1., 0., R_Wrist_Pos[1]],
        [0., 0., 1., R_Wrist_Pos[2]],
        [0., 0., 0., 1.]])

   # print IK_RW, 'RW'
    return right_elbow_chain, IK_RE, right_wrist_chain, IK_RW


def other(m):

    left_elbow_chain = Chain(name='left_elbow', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=m.J[12, :] - m.J[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="l_shoulder_2", translation_vector=m.J[16, :] - m.J[12, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ),
        URDFLink(name="l_shoulder_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], ),
        URDFLink(name="l_elbow", translation_vector=m.J[18, :] - m.J[16, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ), ],
                             active_links_mask=[False, False, True, True, False]
                             # we only solve for pitch and yaw of shoulder
                             )

    L_Elbow_Pos = m.J_transformed[18, :] - m.J_transformed[0, :]
    IK_LE = left_elbow_chain.inverse_kinematics([
        [1., 0., 0., L_Elbow_Pos[0]],
        [0., 1., 0., L_Elbow_Pos[1]],
        [0., 0., 1., L_Elbow_Pos[2]],
        [0., 0., 0., 1.]])

    print
    IK_LE, 'LE'

    left_wrist_chain = Chain(name='left_wrist', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=m.J[12, :] - m.J[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="l_shoulder_1", translation_vector=m.J[16, :] - m.J[12, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="l_shoulder_2", translation_vector=[.0, 0, 0],
                 orientation=[0, IK_LE[2], 0], rotation=[0, 1, 0], ),
        URDFLink(name="l_shoulder_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, IK_LE[3]], rotation=[0, 0, 1], ),
        URDFLink(name="l_elbow", translation_vector=m.J[18, :] - m.J[16, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ),
        URDFLink(name="l_wrist", translation_vector=m.J[20, :] - m.J[18, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ), ],
                             active_links_mask=[False, False, True, False, False, True, False]
                             # solve just for shoulder roll and elbow pitch
                             )

    L_Wrist_Pos = m.J_transformed[20, :] - m.J_transformed[0, :]
    IK_LW = left_wrist_chain.inverse_kinematics([
        [1., 0., 0., L_Wrist_Pos[0]],
        [0., 1., 0., L_Wrist_Pos[1]],
        [0., 0., 1., L_Wrist_Pos[2]],
        [0., 0., 0., 1.]])

    print
    IK_LW, 'LW'
