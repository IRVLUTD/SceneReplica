"""collect images from Fetch"""

import rospy
import message_filters
import cv2
import argparse
import threading
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import tf
import tf2_ros
import tf.transformations as tra
import scipy.io
from transforms3d.quaternions import mat2quat, quat2mat
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String

from ros_utils import ros_qt_to_rt

lock = threading.Lock()


def make_pose(tf_pose):
    """
    Helper function to get a full matrix out of this pose
    """
    trans, rot = tf_pose
    pose = tra.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose


def make_pose_from_pose_msg(msg):
    trans = (msg.pose.position.x,
             msg.pose.position.y, msg.pose.position.z,)
    rot = (msg.pose.orientation.x,
           msg.pose.orientation.y,
           msg.pose.orientation.z,
           msg.pose.orientation.w,)
    return make_pose((trans, rot))
    

class ImageListener:

    def __init__(self, outdir=None):

        self.cv_bridge = CvBridge()
        self.count = 1
        
        self.input_rgb = None
        self.input_depth = None
        self.input_rgb_pose = None
        self.input_stamp = None
        self.input_frame_id = None
        if not outdir:
            # output dir
            this_dir = osp.dirname(__file__)
            self.outdir = osp.join(this_dir, '..', 'data', 'Fetch')
        else:
            self.outdir = outdir
        self.prev_scene = None
        self.scene = "-1000"

        # self.camera_frame = 'laser_link'


        # initialize a node
        rospy.init_node("image_listener")
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=2)
        # rgb_pose_sub = message_filters.Subscriber('/poserbpf_image_render_00', Image, queue_size=2)        
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=2)        
        # depth_sub = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect_raw', Image, queue_size=2)
        scene_change_sub = message_filters.Subscriber('/scene_change', String, queue_size=10)
        
        self.tfBuffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        # self.tf_listener = tf.TransformListener()
        self.base_frame = 'base_link'
        self.camera_frame = 'head_camera_rgb_optical_frame'

        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic_matrix = K
        print('Intrinsics matrix : ')
        print(self.intrinsic_matrix)        

        queue_size = 1
        slop_seconds = 0.5
        # ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, rgb_pose_sub, depth_sub], queue_size, slop_seconds)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, scene_change_sub], queue_size, slop_seconds, allow_headerless=True)
        ts.registerCallback(self.callback)
        
    # callback function to get images
    def callback(self, rgb, depth, scene_change):
    # def callback(self, rgb, rgb_pose, depth):

        # decode image
        if depth is not None:
            if depth.encoding == '32FC1':
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            elif depth.encoding == '16UC1':
                depth = self.cv_bridge.imgmsg_to_cv2(depth)
                depth_cv = depth.copy().astype(np.float32)
                depth_cv /= 1000.0
            else:
                rospy.logerr_throttle(1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))
                return
        else:
            depth_cv = None

        with lock:
            self.input_depth = depth_cv
            self.input_rgb = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
            # self.input_rgb_pose = self.cv_bridge.imgmsg_to_cv2(rgb_pose, 'bgr8')
            self.input_stamp = rgb.header.stamp
            self.input_frame_id = rgb.header.frame_id
            self.scene = scene_change.data


    def save_data(self):
        if self.scene != self.prev_scene:            
            _tf = self.tfBuffer.lookup_transform(self.base_frame, self.camera_frame, rospy.Time(0))
            _trans, _quat = _tf.transform.translation, _tf.transform.rotation
            trans = [_trans.x, _trans.y, _trans.z]
            quat = [_quat.x, _quat.y, _quat.z, _quat.w]
            print(trans, quat)
            RT_cam = ros_qt_to_rt(quat,trans)
            filename = os.path.join(self.outdir, f"pose_info_{self.count}.npy")
            np.save(filename, RT_cam)

            # write color images
            filename = 'color-%06d.jpg' % self.count
            cv2.imwrite(os.path.join(self.outdir, filename), self.input_rgb )
            print(filename)
            
            # write pose color images
            filename = 'pose-%06d.jpg' % self.count
            # cv2.imwrite(os.path.join(self.outdir, filename), self.input_rgb_pose)
            cv2.imwrite(os.path.join(self.outdir, filename), self.input_rgb)
            print(filename)

            filename = 'depth-%06d.png' % self.count
            cv2.imwrite(os.path.join(self.outdir, filename), self.input_depth)
            print(filename)

            self.count += 1
            self.prev_scene = self.scene
        
def make_args():
    parser = argparse.ArgumentParser(
        description="Subscribe to published scene topic and render rgb and depth images to save", add_help=True
    )
    parser.add_argument(
        "--output_dir",
        default="../../data/Fetch/",
        help="path to save the rgb, pose and depth imgs",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_args()
    out_dir = args.output_dir
    print("Specified output dir:", out_dir)

    # image listener
    listener = ImageListener(outdir=out_dir)  
    os.makedirs(listener.outdir, exist_ok=True)

    while not rospy.is_shutdown():
        if listener.input_rgb is not None:
            # save images
            listener.save_data()           
            # sleep for 0.25 seconds
            # break
            time.sleep(0.25)
