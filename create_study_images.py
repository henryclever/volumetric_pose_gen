#Written by Henry M. Clever. November 15, 2018.


PLOT = True
RENDER_FLEX = False
COLOR_BAR = True
DART_TO_FLEX_CONV = 2.58872



from time import time
from time import sleep
from random import random
from random import randint
import numpy as np

class pyFlex():

    def assign_human_meshing(self, m, mTrans, resting_pose_data):


        capsule_angles = resting_pose_data[0].tolist()
        root_joint_pos_list = resting_pose_data[1]
        body_shape_list = resting_pose_data[2]

        for shape_param in range(10):
            m.betas[shape_param] = float(body_shape_list[shape_param])


        #print human_shiftUD, capsule_angles[3:6]

        m.pose[0:3] = capsule_angles[0:3]
        m.pose[3:6] = capsule_angles[6:9]
        m.pose[6:9] = capsule_angles[9:12]
        m.pose[9:12] = capsule_angles[12:15]
        m.pose[12] = capsule_angles[15]
        m.pose[15] = capsule_angles[16]
        m.pose[18:21] = capsule_angles[17:20]
        m.pose[21:24] = capsule_angles[20:23]
        m.pose[24:27] = capsule_angles[23:26]
        m.pose[27:30] = capsule_angles[26:29]
        m.pose[36:39] = capsule_angles[29:32] # neck
        m.pose[39:42] = capsule_angles[32:35]
        m.pose[42:45] = capsule_angles[35:38]
        m.pose[45:48] = capsule_angles[38:41]  # head
        m.pose[48:51] = capsule_angles[41:44]
        m.pose[51:54] = capsule_angles[44:47]
        m.pose[55] = capsule_angles[47]
        m.pose[58] = capsule_angles[48]
        m.pose[60:63] = capsule_angles[49:52]
        m.pose[63:66] = capsule_angles[52:55]

        return m





if __name__ == '__main__':

    from smpl.smpl_webuser.serialization import load_model
    import time
    import lib_pyrender as libPyRender
    pyRender = libPyRender.pyRenderMesh()

    gender = "f"
    posture = "lay"
    num_data = 2000
    stiffness = "none"

    resting_pose_data_list = np.load("/home/henry/data/resting_poses/resting_pose_roll0_plo_hbh_f_lay_set2_1769_of_2601_none_stiff_fix.npy", allow_pickle = True)
    model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_'+gender+'_lbs_10_207_0_v1.0.0.pkl'
    m = load_model(model_path)

    PERSON_SCALE = 50.0

    # first get the offsets
    mTransX = m.r[0, 0] * DART_TO_FLEX_CONV / (PERSON_SCALE * 0.1)  # X appears to move sideways
    mTransY = m.r[0, 1] * DART_TO_FLEX_CONV / (PERSON_SCALE * 0.1)  # Y trans appears to move up in air
    mTransZ = m.r[0, 2] * DART_TO_FLEX_CONV / (PERSON_SCALE * 0.1)  # Z trans appears to move forward
    mTrans = [mTransX, mTransY, mTransZ]


    #resting_pose_data_list = np.load("/home/henry/data/resting_poses/resting_pose_" + gender +"_"+posture+"_" + str(num_data) + "_"+stiffness+"_stiff.npy")
    p_mat_list = []

    for resting_pose_data in resting_pose_data_list[200:]:

        pF = pyFlex()

        m = pF.assign_human_meshing(m, mTrans, resting_pose_data)

        pyRender.render_only_human_gt(m)
        #time.sleep(100000)
