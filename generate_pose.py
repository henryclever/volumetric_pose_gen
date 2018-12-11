import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model

import lib_visualization as libVisualization
import lib_kinematics as libKinematics

#ROS
import rospy
import tf

import tensorflow as tensorflow

#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL



render = 'ikpy'



## Load SMPL model (here we load the female model)
model_path = '../SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
m = load_model(model_path)

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * 0.
m.betas[:] = np.random.rand(m.betas.size) * .0
#m.betas[5] = 20.

m.pose[0] = 0 #pitch rotation of the person in space. 0 means the person is upside down facing back. pi is standing up facing forward
m.pose[1] = 0 #roll of the person in space. -pi/2 means they are tilted to their right side
m.pose[2] = 0 #-np.pi/4 #yaw of the person in space, like turning around normal to the ground

m.pose[3] = np.pi/4 #left hip extension (i.e. leg bends back for np.pi/2)
m.pose[4] = 0 #left leg yaw about hip, where np.pi/2 makes bowed leg
m.pose[5] = 0#-np.pi/8 #left leg abduction (POS) /adduction (NEG)

m.pose[6] = np.pi/2 #right hip extension (i.e. leg bends back for np.pi/2)
m.pose[7] = -np.pi/2
m.pose[8] = 0#-np.pi/2 #right leg abduction (NEG) /adduction (POS)

m.pose[9] = 0 #bending of spine at hips. np.pi/2 means person bends down to touch the ground
m.pose[10] = 0 #twisting of spine at hips. body above spine yaws normal to the ground
m.pose[11] = 0 #bending of spine at hips. np.pi/2 means person bends down sideways to touch the ground 3

m.pose[12] = np.pi/4 #left knee extension. (i.e. knee bends back for np.pi/2)
m.pose[13] = 0 #twisting of knee normal to ground. KEEP AT ZERO
m.pose[14] = 0 #bending of knee sideways. KEEP AT ZERO

m.pose[15] = np.pi/2 #right knee extension (i.e. knee bends back for np.pi/2)

m.pose[18] = 0 #bending at mid spine. makes person into a hunchback for positive values
m.pose[19] = 0#twisting of midspine. body above midspine yaws normal to the ground
m.pose[20] = 0 #bending of midspine, np.pi/2 means person bends down sideways to touch ground 6

m.pose[21] = 0 #left ankle flexion/extension
m.pose[22] = 0 #left ankle yaw about leg
m.pose[23] = 0 #left ankle twist KEEP CLOSE TO ZERO

m.pose[24] = 0 #right ankle flexion/extension
m.pose[25] = 0 #right ankle yaw about leg
m.pose[26] = np.pi/4 #right ankle twist KEEP CLOSE TO ZERO

m.pose[27] = 0 #bending at upperspine. makes person into a hunchback for positive values
m.pose[28] = 0#twisting of upperspine. body above upperspine yaws normal to the ground
m.pose[29] = 0 #bending of upperspine, np.pi/2 means person bends down sideways to touch ground 9

m.pose[30] = 0 #flexion/extension of left ankle midpoint

m.pose[33] = 0 #flexion/extension of right ankle midpoint

m.pose[36] = 0 #flexion/extension of neck. i.e. whiplash 12
m.pose[37] = 0 #yaw of neck

m.pose[39] = 0 #left inner shoulder roll
m.pose[40] = 0 #left inner shoulder yaw, negative moves forward
m.pose[41] = 0 #left inner shoulder pitch, positive moves up

m.pose[42] = 0
m.pose[43] = 0 #right inner shoulder yaw, positive moves forward
m.pose[44] = 0 #right inner shoulder pitch, positive moves down

m.pose[45] = 0 #flexion/extension of head 15

m.pose[48] = 0 #left outer shoulder roll
#m.pose[49] = -np.pi/4
m.pose[50] = np.pi/4 #left outer shoulder pitch

m.pose[51] = 0. #right outer shoulder roll
m.pose[52] = 0.#np.pi/2
m.pose[53] = 0.#np.pi/3

m.pose[54] = 0 #left elbow roll KEEP AT ZERO
#m.pose[55] = -np.pi/4 #left elbow flexion/extension. KEEP NEGATIVE
m.pose[56] = 0 #left elbow KEEP AT ZERO

m.pose[57] = 0
m.pose[58] = 0#np.pi/4 #right elbow flexsion/extension KEEP POSITIVE

m.pose[60] = 0 #left wrist roll

m.pose[63] = 0 #right wrist roll
#m.pose[65] = np.pi/5

m.pose[66] = 0 #left hand roll

m.pose[69] = 0 #right hand roll
#m.pose[71] = np.pi/5 #right fist


m.betas[0] = 0. #overall body size. more positive number makes smaller, negative makes larger with bigger belly
m.betas[1] = 0. #positive number makes person very skinny, negative makes fat
m.betas[2] = 3. #muscle mass. higher makes person less physically fit
m.betas[3] = 0. #proportion for upper vs lower bone lengths. more negative number makes legs much bigger than arms
m.betas[4] = 3. #neck. more negative seems to make neck longer and body more skinny
m.betas[5] = 6. #size of hips. larger means bigger hips
m.betas[6] = 0. #proportion of belly with respect to rest of the body. higher number is larger belly
m.betas[7] = 0
m.betas[8] = -4
m.betas[9] = 0

print m.pose.shape
print m.pose, 'pose'
print m.betas, 'betas'




def standard_render():

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


def solve_ik_tree_smpl():

    # print the origin
    print m.J - m.J[0, :]

    ax = plt.figure().add_subplot(111, projection='3d')

    #grab the joint positions
    r_leg_pos_origins = np.array([np.array(m.J[0, :]), np.array(m.J[2, :]), np.array(m.J[5, :]), np.array(m.J[8, :])])
    r_leg_pos_current = np.array([np.array(m.J_transformed[0, :]), np.array(m.J_transformed[2, :]), np.array(m.J_transformed[5, :]), np.array(m.J_transformed[8, :])])


    #get the IKPY solution
    r_knee_chain, IK_RK, r_ankle_chain, IK_RA = libKinematics.ikpy_leg(r_leg_pos_origins, r_leg_pos_current)


    #get the RPH values
    r_hip_angles = [IK_RK[1], IK_RA[2], IK_RK[2]]
    r_knee_angle = [np.copy(IK_RA[4])]

    #grab the joint angle ground truth
    r_hip_angles_GT = m.pose[6:9]
    r_knee_angle_GT = m.pose[15]
    IK_RK_GT = np.copy(IK_RK)
    IK_RK_GT[1] = r_hip_angles_GT[0]
    IK_RK_GT[2] = r_hip_angles_GT[2]

    IK_RA_GT = np.copy(IK_RA)
    IK_RA_GT[1] = r_hip_angles_GT[0]
    IK_RA_GT[2] = r_hip_angles_GT[1]
    IK_RA_GT[3] = r_hip_angles_GT[2]
    IK_RA_GT[4] = r_knee_angle_GT

    print IK_RK, 'IK_RK'
    print IK_RK_GT, 'IK_RK_GT'
    print IK_RA, "IK_RA"
    print IK_RA_GT, 'IK_RA_GT'

   # #IK_RK_GT = np.copy(IK_RK)
    #IK_RK_GT[2] = r_hip_angles_GT[1]
    #IK_RK_GT[4] = r_knee_angle_GT

    #IK_RA[3] = 1.0


    print "right hip GT: ", r_hip_angles_GT
    print "right hip estimated: ", r_hip_angles
    print "right knee GT: ", r_knee_angle_GT
    print "right knee estimated: ", r_knee_angle


    #grab the joint positions
    #l_leg_pos_origins = np.array([np.array(m.J[0, :]), np.array(m.J[1, :]), np.array(m.J[4, :]), np.array(m.J[7, :])])
    #l_leg_pos_current = np.array([np.array(m.J_transformed[0, :]), np.array(m.J_transformed[1, :]), np.array(m.J_transformed[4, :]), np.array(m.J_transformed[7, :])])

    #get the IK solution
    #l_knee_chain, IK_LK, l_ankle_chain, IK_LA  = libKinematics.ikpy_leg(l_leg_pos_origins, l_leg_pos_current)

    #get the RPH values
    #l_hip_angles = [IK_LK[1], IK_LA[2], IK_LK[2]]
    #l_knee_angle = [IK_LA[4]]

    #grab the joint angle ground truth
    #l_hip_angles_GT = m.pose[3:6]
    #l_knee_angle_GT = m.pose[12]
    #IK_LA_GT = np.copy(IK_LA)
    #IK_LA_GT[2] = l_hip_angles_GT[1]
    #IK_LA_GT[4] = l_knee_angle_GT

    #print "left hip GT: ", l_hip_angles_GT
    #print "left hip estimated: ", l_hip_angles
    #print "left knee GT: ", l_knee_angle_GT
    #print "left knee estimated: ", l_knee_angle



    #grab the joint positions
    r_arm_pos_origins = np.array([np.array(m.J[0, :]), np.array(m.J[12, :]), np.array(m.J[17, :]), np.array(m.J[19, :]), np.array(m.J[21, :]) ])
    r_arm_pos_current = np.array([np.array(m.J_transformed[0, :]), np.array(m.J_transformed[12, :]), np.array(m.J_transformed[17, :]), np.array(m.J_transformed[19, :]), np.array(m.J_transformed[21, :])])

    #get the IK solution
    r_elbow_chain, IK_RE, r_wrist_chain, IK_RW = libKinematics.ikpy_arm(r_arm_pos_origins, r_arm_pos_current)

    #get the RPH values
    r_shoulder_angles = [IK_RW[2], IK_RE[2], IK_RE[3]]
    r_elbow_angle = [IK_RW[5]]

    #grab the joint angle ground truth
    r_shoulder_angles_GT = m.pose[51:54]
    r_elbow_angle_GT = m.pose[58]
    IK_RW_GT = np.copy(IK_RW)
    IK_RW_GT[2] = r_shoulder_angles_GT[1]
    IK_RW_GT[5] = r_elbow_angle_GT


    IK_RE_GT = np.copy(IK_RE)
    IK_RE_GT[2] = r_shoulder_angles_GT[0]
    IK_RE_GT[3] = r_shoulder_angles_GT[1]
    IK_RE_GT[4] = r_shoulder_angles_GT[2]


    print "right shoulder GT: ", r_shoulder_angles_GT
    print "right shoulder estimated: ", r_shoulder_angles
    print "right elbow GT: ", r_elbow_angle_GT
    print "right elbow estimated: ", r_elbow_angle




    r_knee_chain.plot(IK_RK, ax)
    r_ankle_chain.plot(IK_RA, ax)
    #l_ankle_chain.plot(IK_LA, ax)
    #r_elbow_chain.plot(IK_RE, ax)
    #r_wrist_chain.plot(IK_RW, ax)
    # left_elbow_chain.plot(IK_LE, ax)
    #left_wrist_chain.plot(IK_LW, ax)


    Jt = np.array(m.J_transformed-m.J_transformed[0,:])
    ax.plot(Jt[:,0],Jt[:,1],Jt[:,2], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.5)
    plt.show()

    #with tensorflow.Session() as sess:
    #    smpl = SMPL(model_path)
    #    J_transformed = smpl(m.betas, m.pose)

    #print J_transformed
    #print m.J_transformed



def render_rviz():

    print m.J_transformed
    rospy.init_node('smpl_model')

    shift_sideways = np.zeros((24,3))
    shift_sideways[:, 0] = 1.0

    for i in range(0, 10):
        #libVisualization.rviz_publish_output(None, np.array(m.J_transformed))
        time.sleep(0.5)

        concatted = np.concatenate((np.array(m.J_transformed), np.array(m.J_transformed) + shift_sideways), axis = 0)
        #print concatted
        #libVisualization.rviz_publish_output(None, np.array(m.J_transformed) + shift_sideways)
        libVisualization.rviz_publish_output(None, concatted)
        time.sleep(0.5)


if __name__ == "__main__":
    solve_ik_tree_smpl()