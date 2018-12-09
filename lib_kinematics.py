#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


def ik_right_leg(m):

    right_knee_chain = Chain(name='right_knee', links=[
        OriginLink(),
        URDFLink(name="r_hip_1", translation_vector=m.J[2, :] - m.J[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_hip_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], ),
        URDFLink(name="r_knee", translation_vector=m.J[5, :] - m.J[2, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(0, np.pi), ), ]
                             )

    R_Knee_Pos = m.J_transformed[5, :] - m.J_transformed[0, :]
    IK_RK = right_knee_chain.inverse_kinematics([
        [1., 0., 0., R_Knee_Pos[0]],
        [0., 1., 0., R_Knee_Pos[1]],
        [0., 0., 1., R_Knee_Pos[2]],
        [0., 0., 0., 1.]])

    right_ankle_chain = Chain(name='right_ankle', links=[
        OriginLink(),
        URDFLink(name="r_hip_1", translation_vector=m.J[2, :] - m.J[0, :],
                 orientation=[IK_RK[1], 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi), ),
        URDFLink(name="r_hip_2", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-np.pi, np.pi), ),
        URDFLink(name="r_hip_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, IK_RK[2]], rotation=[0, 0, 1], bounds=(-np.pi, np.pi), ),
        URDFLink(name="r_knee", translation_vector=m.J[5, :] - m.J[2, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(0, np.pi), ),
        URDFLink(name="r_ankle", translation_vector=m.J[8, :] - m.J[5, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi), ), ],
                              active_links_mask=[False, False, True, False, True, False]
                              )

    R_Ankle_Pos = m.J_transformed[8, :] - m.J_transformed[0, :]
    IK_RA = right_ankle_chain.inverse_kinematics([
        [1., 0., 0., R_Ankle_Pos[0]],
        [0., 1., 0., R_Ankle_Pos[1]],
        [0., 0., 1., R_Ankle_Pos[2]],
        [0., 0., 0., 1.]])


    return right_knee_chain, IK_RK, right_ankle_chain, IK_RA



def ik_left_leg(m):

    left_knee_chain = Chain(name='left_knee', links=[
        OriginLink(),
        URDFLink(name="l_hip_1", translation_vector=m.J[1, :] - m.J[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="l_hip_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], ),
        URDFLink(name="l_knee", translation_vector=m.J[4, :] - m.J[1, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(0, np.pi), ), ]
                            )

    # print left_knee_chain.forward_kinematics([0] * 4)
    L_Knee_Pos = m.J_transformed[4, :] - m.J_transformed[0, :]
    IK_LK = left_knee_chain.inverse_kinematics([
        [1., 0., 0., L_Knee_Pos[0]],
        [0., 1., 0., L_Knee_Pos[1]],
        [0., 0., 1., L_Knee_Pos[2]],
        [0., 0., 0., 1.]])

    print
    IK_LK, 'LK'

    left_ankle_chain = Chain(name='left_ankle', links=[
        OriginLink(),
        URDFLink(name="l_hip_1", translation_vector=m.J[1, :] - m.J[0, :],
                 orientation=[IK_LK[1], 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi), ),
        URDFLink(name="l_hip_2", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], bounds=(-np.pi, np.pi), ),
        URDFLink(name="l_hip_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, IK_LK[2]], rotation=[0, 0, 1], bounds=(-np.pi, np.pi), ),
        URDFLink(name="l_knee", translation_vector=m.J[4, :] - m.J[1, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(0, np.pi), ),
        URDFLink(name="l_ankle", translation_vector=m.J[7, :] - m.J[4, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], bounds=(-np.pi, np.pi), ), ],
                             active_links_mask=[False, False, True, False, True, False]
                             )

    # print left_ankle_chain.forward_kinematics([0] * 6)
    L_Ankle_Pos = m.J_transformed[7, :] - m.J_transformed[0, :]
    IK_LA = left_ankle_chain.inverse_kinematics([
        [1., 0., 0., L_Ankle_Pos[0]],
        [0., 1., 0., L_Ankle_Pos[1]],
        [0., 0., 1., L_Ankle_Pos[2]],
        [0., 0., 0., 1.]])

    print
    IK_LA, 'LA'

    right_elbow_chain = Chain(name='right_elbow', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=m.J[12, :] - m.J[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_shoulder_2", translation_vector=m.J[17, :] - m.J[12, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ),
        URDFLink(name="r_shoulder_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, 0], rotation=[0, 0, 1], ),
        URDFLink(name="r_elbow", translation_vector=m.J[19, :] - m.J[17, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ), ],
                              active_links_mask=[False, False, True, True, False]
                              )

    R_Elbow_Pos = m.J_transformed[19, :] - m.J_transformed[0, :]
    IK_RE = right_elbow_chain.inverse_kinematics([
        [1., 0., 0., R_Elbow_Pos[0]],
        [0., 1., 0., R_Elbow_Pos[1]],
        [0., 0., 1., R_Elbow_Pos[2]],
        [0., 0., 0., 1.]])

    print
    IK_RE, 'RE'

    right_wrist_chain = Chain(name='right_wrist', links=[
        OriginLink(),
        URDFLink(name="neck", translation_vector=m.J[12, :] - m.J[0, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_shoulder_1", translation_vector=m.J[17, :] - m.J[12, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ),
        URDFLink(name="r_shoulder_2", translation_vector=[.0, 0, 0],
                 orientation=[0, IK_RE[2], 0], rotation=[0, 1, 0], ),
        URDFLink(name="r_shoulder_3", translation_vector=[.0, 0, 0],
                 orientation=[0, 0, IK_RE[3]], rotation=[0, 0, 1], ),
        URDFLink(name="r_elbow", translation_vector=m.J[19, :] - m.J[17, :],
                 orientation=[0, 0, 0], rotation=[0, 1, 0], ),
        URDFLink(name="r_wrist", translation_vector=m.J[21, :] - m.J[19, :],
                 orientation=[0, 0, 0], rotation=[1, 0, 0], ), ],
                              active_links_mask=[False, False, True, False, False, True, False]
                              )

    R_Wrist_Pos = m.J_transformed[21, :] - m.J_transformed[0, :]
    IK_RW = right_wrist_chain.inverse_kinematics([
        [1., 0., 0., R_Wrist_Pos[0]],
        [0., 1., 0., R_Wrist_Pos[1]],
        [0., 0., 1., R_Wrist_Pos[2]],
        [0., 0., 0., 1.]])

    print
    IK_RW, 'RW'

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

    # right_knee_chain.plot(IK_RK, ax)
    right_ankle_chain.plot(IK_RA, ax)
    left_ankle_chain.plot(IK_LA, ax)
    # right_elbow_chain.plot(IK_RE, ax)
    right_wrist_chain.plot(IK_RW, ax)
    # left_elbow_chain.plot(IK_LE, ax)
    left_wrist_chain.plot(IK_LW, ax)

    plt.show()