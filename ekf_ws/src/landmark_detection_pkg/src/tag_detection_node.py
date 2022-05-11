#!/usr/bin/env python3

"""
This node will subscribe to detected AprilTag poses,
and publish the ID + position of each.
We assume landmarks are orientation invariant.
"""

import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from std_msgs.msg import Float32MultiArray
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from math import tan

############ GLOBAL VARIABLES ###################
DT = 1 # timer period.
tag_pub = None
########### TF INFO ############
tf_listener = None
tf_buffer = None
TF_ORIGIN = 'map'
TF_CAMERA = 'camera_link'
##########################################


def get_tag_detection(tag_msg):
    """
    Detect AprilTag pose(s) relative to the camera frame.
    @param tag_msg: an AprilTagDetectionArray message containing a list of detected tags.
     - This list is empty but still published when no tags are detected.
    """
    # verify there is at least one tag detected.
    if len(tag_msg.detections) == 0:
        return

    # create the message.
    msg = Float32MultiArray()
    msg.data = []
    
    # do for all detected tags.
    for i in range(len(tag_msg.detections)):
        tag_id = tag_msg.detections[i].id
        tag_pose = tag_msg.detections[i].pose.pose.pose
        # extract translation (t) and orientation quaternion (q).
        t = [tag_pose.position.x,tag_pose.position.y,tag_pose.position.z]
        q = [tag_pose.orientation.w, tag_pose.orientation.x, tag_pose.orientation.y, tag_pose.orientation.z]
        # make it into an affine matrix.
        r = R.from_quat(q).as_matrix()
        # make affine matrix for transformation. (apriltag relative to camera frame)
        T_AC = np.array([[r[0][0],r[0][1],r[0][2],t[0]],
                        [r[1][0],r[1][1],r[1][2],t[1]],
                        [r[2][0],r[2][1],r[2][2],t[2]],
                        [0,0,0,1]])

        # convert this into 2D relative range, bearing to camera.
        tag_range = (t[0]**2 + t[1]**2)**(1/2)
        tag_bearing = tan(t[1] / t[2])
    
        # concatenate lists [id, range, bearing] for each.
        msg.data += [tag_id, tag_range, tag_bearing]
    # publish all the info for this detection.
    tag_pub.publish(msg)


def get_transform(TF_TO:str, TF_FROM:str):
    """
    Get the expected transform from tf.
    Use translation and quaternion from tf to construct a pose in SE(3).
    """
    try:
        # get most recent relative pose from the tf service.
        pose = tf_buffer.lookup_transform(TF_TO, TF_FROM, rospy.Time(0), rospy.Duration(4))
    except Exception as e:
        # requested transform was not found.
        print("Transform from " + TF_FROM + " to " + TF_TO + " not found.")
        print("Exception: ", e)
        return None
    
    # extract translation and quaternion from tf pose.
    t = [pose.transform.translation.x, pose.transform.translation.y, pose.transform.translation.z]
    q = (pose.transform.rotation.x, pose.transform.rotation.y, pose.transform.rotation.z, pose.transform.rotation.w)
    # get equiv rotation matrix from quaternion.
    r = R.from_quat(q).as_matrix()

    # make affine matrix for transformation.
    return np.array([[r[0][0],r[0][1],r[0][2],t[0]],
                    [r[1][0],r[1][1],r[1][2],t[1]],
                    [r[2][0],r[2][1],r[2][2],t[2]],
                    [0,0,0,1]])


def main():
    global tag_pub, tf_listener, tf_buffer
    rospy.init_node('tag_tracking_node')

    # setup TF service.
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(1))
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # subscribe to apriltag detections.
    rospy.Subscriber("/tag_detections", AprilTagDetectionArray, get_tag_detection, queue_size=1)

    # create publisher for simplified landmark details [id1,r1,b1,...idN,rN,bN]
    tag_pub = rospy.Publisher("/landmark/apriltag", Float32MultiArray, queue_size=100)

    # rospy.Timer(rospy.Duration(DT), timer_callback)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass