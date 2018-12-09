import numpy as np

#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


def ik_leg(origins, current):

    knee_chain = Chain(name='knee', links=[
        OriginLink(),
        URDFLink(name="hip_1", translation_vector=origins[1, :] - origins[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="hip_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], ),
        URDFLink(name="knee", translation_vector=origins[2, :] - origins[1, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(0, np.pi), ), ]
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
                 orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-np.pi, np.pi), ),
        URDFLink(name="hip_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, IK_K[2]], rotation=[0, 0, 1], bounds=(-np.pi, np.pi), ),
        URDFLink(name="knee", translation_vector=origins[2, :] - origins[1, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(0, np.pi), ),
        URDFLink(name="ankle", translation_vector=origins[3, :] - origins[2, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi), ), ],
                              active_links_mask=[False, False, True, False, True, False]
                              )

    Ankle_Pos = current[3, :] - current[0, :]
    IK_A = ankle_chain.inverse_kinematics([
        [1., 0., 0., Ankle_Pos[0]],
        [0., 1., 0., Ankle_Pos[1]],
        [0., 0., 1., Ankle_Pos[2]],
        [0., 0., 0., 1.]])


    return knee_chain, IK_K, ankle_chain, IK_A



def ik_arm(origin, current):


    right_elbow_chain = Chain(name='right_elbow', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=origin[1, :] - origin[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_shoulder_2", translation_vector=origin[2, :] - origin[1, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ),
        URDFLink(name="r_shoulder_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], ),
        URDFLink(name="r_elbow", translation_vector=origin[3, :] - origin[2, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ), ],
                              active_links_mask=[False, False, True, True, False]
                              )

    R_Elbow_Pos = current[3, :] - current[0, :]
    IK_RE = right_elbow_chain.inverse_kinematics([
        [1., 0., 0., R_Elbow_Pos[0]],
        [0., 1., 0., R_Elbow_Pos[1]],
        [0., 0., 1., R_Elbow_Pos[2]],
        [0., 0., 0., 1.]])

    print IK_RE, 'RE'

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

    print IK_RW, 'RW'
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
