
import numpy as np


#ROS
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf



def rviz_publish_output(scores):

    print scores.shape

    ScoresArray = MarkerArray()
    for joint in range(0, scores.shape[0]):
        print joint
        scoresPublisher = rospy.Publisher("/scores", MarkerArray)
        Smarker = Marker()
        Smarker.header.frame_id = "map"
        Smarker.type = Smarker.SPHERE
        Smarker.action = Smarker.ADD
        Smarker.scale.x = 0.06
        Smarker.scale.y = 0.06
        Smarker.scale.z = 0.06
        Smarker.color.a = 1.0

        if joint == 0:
            Smarker.color.r = 1.0
        else:
            Smarker.color.r = 1.0
            Smarker.color.g = 1.0
        Smarker.color.b = 0.0

        Smarker.pose.orientation.w = 1.0
        Smarker.pose.position.x = scores[joint, 0]
        Smarker.pose.position.y = scores[joint, 1]
        Smarker.pose.position.z = scores[joint, 2]
        ScoresArray.markers.append(Smarker)
        sid = 0
        for m in ScoresArray.markers:
            m.id = sid
            sid += 1

    #print ScoresArray
    scoresPublisher.publish(ScoresArray)


def rviz_publish_output_limbs_direct(scores, count = 0):

    #if LimbArray == None or count <= 2:
    LimbArray = MarkerArray()

    limbs = {}
    limbs['left_glute'] = [scores[0,0], scores[0,1], scores[0,2], scores[1,0], scores[1,1], scores[1,2]]
    limbs['right_glute'] = [scores[0,0], scores[0,1], scores[0,2], scores[2,0], scores[2,1], scores[2,2]]
    limbs['left_thigh'] = [scores[1,0], scores[1,1], scores[1,2], scores[4,0], scores[4,1], scores[4,2]]
    limbs['right_thigh'] = [scores[2,0], scores[2,1], scores[2,2], scores[5,0], scores[5,1], scores[5,2]]
    limbs['left_calf'] = [scores[4,0], scores[4,1], scores[4,2], scores[7,0], scores[7,1], scores[7,2]]
    limbs['right_calf'] = [scores[5,0], scores[5,1], scores[5,2], scores[8,0], scores[8,1], scores[8,2]]
    limbs['spine1'] = [scores[0,0], scores[0,1], scores[0,2], scores[3,0], scores[3,1], scores[3,2]]
    limbs['left_foot'] = [scores[7,0], scores[7,1], scores[7,2], scores[10,0], scores[10,1], scores[10,2]]
    limbs['right_foot'] = [scores[8,0], scores[8,1], scores[8,2], scores[11,0], scores[11,1], scores[11,2]]
    limbs['spine2'] = [scores[3,0], scores[3,1], scores[3,2], scores[6,0], scores[6,1], scores[6,2]]
    limbs['spine3'] = [scores[6,0], scores[6,1], scores[6,2], scores[9,0], scores[9,1], scores[9,2]]
    limbs['spine4'] = [scores[9,0], scores[9,1], scores[9,2], scores[12,0], scores[12,1], scores[12,2]]
    limbs['left_inner_should'] = [scores[9,0], scores[9,1], scores[9,2], scores[13,0], scores[13,1], scores[13,2]]
    limbs['right_inner_should'] = [scores[9,0], scores[9,1], scores[9,2], scores[14,0], scores[14,1], scores[14,2]]
    limbs['neck'] = [scores[12,0], scores[12,1], scores[12,2], scores[15,0], scores[15,1], scores[15,2]]
    limbs['left_outer_should'] = [scores[13,0], scores[13,1], scores[13,2], scores[16,0], scores[16,1], scores[16,2]]
    limbs['right_outer_should'] = [scores[14,0], scores[14,1], scores[14,2], scores[17,0], scores[17,1], scores[17,2]]
    limbs['left_upperarm'] = [scores[16,0], scores[16,1], scores[16,2], scores[18,0], scores[18,1], scores[18,2]]
    limbs['right_upperarm'] = [scores[17,0], scores[17,1], scores[17,2], scores[19,0], scores[19,1], scores[19,2]]
    limbs['left_forearm'] = [scores[18, 0], scores[18, 1], scores[18, 2],scores[20,0], scores[20,1], scores[20,2]]
    limbs['right_forearm'] = [scores[19, 0], scores[19, 1], scores[19, 2], scores[21, 0], scores[21, 1], scores[21, 2]]
    limbs['left_wrist'] = [scores[20,0], scores[20,1], scores[20,2], scores[22,0], scores[22,1], scores[22,2]]
    limbs['right_wrist'] = [scores[21,0], scores[21,1], scores[21,2], scores[23,0], scores[23,1], scores[23,2]]

    for limb in limbs:
        sx1 = limbs[limb][0]
        sy1 = limbs[limb][1]
        sz1 = limbs[limb][2]
        sx2 = limbs[limb][3]
        sy2 = limbs[limb][4]
        sz2 = limbs[limb][5]

        limbscorePublisher = rospy.Publisher("/limbscoresdirect", MarkerArray)
        Lmarker = Marker()
        Lmarker.header.frame_id = "map"
        Lmarker.type = Lmarker.CYLINDER
        Lmarker.action = Lmarker.ADD
        x_origin = np.array([1., 0., 0.])
        z_vector = np.array([(sx2-sx1), (sy2-sy1), (sz2-sz1)])
        z_mag = np.linalg.norm(z_vector)
        z_vector = z_vector / z_mag

        y_orth = np.cross(z_vector, x_origin)
        y_orth = y_orth / np.linalg.norm(y_orth)

        x_orth = np.cross(y_orth, z_vector)
        x_orth = x_orth / np.linalg.norm(x_orth)

        ROT_mat = np.matrix(np.eye(4))
        ROT_mat[0:3, 0] = np.copy(np.reshape(x_orth, [3,1]))
        ROT_mat[0:3, 1] = np.copy(np.reshape(y_orth, [3,1]))
        ROT_mat[0:3, 2] = np.copy(np.reshape(z_vector, [3,1]))


        Lmarker.scale.z = z_mag

        if count <= 0:
            Lmarker.color.a = 1.0
            Lmarker.scale.x = 0.025
            Lmarker.scale.y = 0.025
        else:
            Lmarker.color.a = 0.5
            Lmarker.scale.x = 0.015
            Lmarker.scale.y = 0.015

        Lmarker.color.r = 1.0
        Lmarker.color.g = 1.0
        Lmarker.color.b = 0.0
        Lmarker.pose.orientation.x = tf.transformations.quaternion_from_matrix(ROT_mat)[0]
        Lmarker.pose.orientation.y = tf.transformations.quaternion_from_matrix(ROT_mat)[1]
        Lmarker.pose.orientation.z = tf.transformations.quaternion_from_matrix(ROT_mat)[2]
        Lmarker.pose.orientation.w = tf.transformations.quaternion_from_matrix(ROT_mat)[3]

        Lmarker.pose.position.x = (sx1+sx2)/2
        Lmarker.pose.position.y = (sy1+sy2)/2
        Lmarker.pose.position.z = (sz1+sz2)/2
        LimbArray.markers.append(Lmarker)
        lid = 0
        for m in LimbArray.markers:
            m.id = lid
            lid += 1


    limbscorePublisher.publish(LimbArray)

    return LimbArray
