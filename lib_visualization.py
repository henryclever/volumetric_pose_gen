
import numpy as np


#ROS
import rospy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf



def rviz_publish_output(targets, scores, scores_std=None):
    TargetArray = MarkerArray()
    if targets is not None:
        for joint in range(0, targets.shape[0]):
            targetPublisher = rospy.Publisher("/targets", MarkerArray)
            Tmarker = Marker()
            Tmarker.header.frame_id = "map"
            Tmarker.type = Tmarker.SPHERE
            Tmarker.action = Tmarker.ADD
            Tmarker.scale.x = 0.07
            Tmarker.scale.y = 0.07
            Tmarker.scale.z = 0.07
            Tmarker.color.a = 1.0
            Tmarker.color.r = 0.0
            Tmarker.color.g = 0.69
            Tmarker.color.b = 0.0
            Tmarker.pose.orientation.w = 1.0
            Tmarker.pose.position.x = targets[joint, 0]
            Tmarker.pose.position.y = targets[joint, 1]
            Tmarker.pose.position.z = targets[joint, 2]
            TargetArray.markers.append(Tmarker)
            tid = 0
            for m in TargetArray.markers:
                m.id = tid
                tid += 1
        targetPublisher.publish(TargetArray)

    ScoresArray = MarkerArray()
    for joint in range(0, scores.shape[0]):
        scoresPublisher = rospy.Publisher("/scores", MarkerArray)
        Smarker = Marker()
        Smarker.header.frame_id = "map"
        Smarker.type = Smarker.SPHERE
        Smarker.action = Smarker.ADD
        Smarker.scale.x = 0.06
        Smarker.scale.y = 0.06
        Smarker.scale.z = 0.06
        Smarker.color.a = 1.0
        if scores_std is not None:
            # print scores_std[joint], 'std of joint ', joint
            # std of 3 is really uncertain
            Smarker.color.r = 1.0
            Smarker.color.g = 1.0 - scores_std[joint] / 0.05
            Smarker.color.b = scores_std[joint] / 0.05

        else:
            if joint == 17:
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
    scoresPublisher.publish(ScoresArray)

