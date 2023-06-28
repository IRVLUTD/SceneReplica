#!/usr/bin/env python
"""ROS image listener"""

import os, sys
import glob
import threading
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
from scipy.io import savemat

import rospy
import tf
import tf2_ros
import message_filters
from tf.transformations import quaternion_matrix
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from geometry_msgs.msg import Pose, PoseArray, Point
from cv_bridge import CvBridge, CvBridgeError
from ros_utils import ros_qt_to_rt, ros_pose_to_rt

from utils_segmentation import visualize_segmentation
from grasp_utils import compute_xyz

lock = threading.Lock()


class ImageListener:

    def __init__(self, camera='Fetch'):

        self.cv_bridge = CvBridge()

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None

        # initialize a node
        self.tf_listener = tf.TransformListener()        

        if camera == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame        
        elif camera == 'Realsense':
            # use RealSense camera
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Azure':
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:
            print('camera %s is not supported in image listener' % camera)
            sys.exit(1)

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.intrinsics = intrinsics
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)


    def callback_rgbd(self, rgb, depth):
    
        # get camera pose in base
        try:
             trans, rot = self.tf_listener.lookupTransform(self.base_frame, self.camera_frame, rospy.Time(0))
             RT_camera = ros_qt_to_rt(rot, trans)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Update failed... " + str(e))
            RT_camera = None             

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = RT_camera


    def get_data(self):

        with lock:
            if self.im is None:
                return None, None, None, None, None, self.intrinsics
            im_color = self.im.copy()
            depth_image = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp
            RT_camera = self.RT_camera.copy()

        xyz_image = compute_xyz(depth_image, self.fx, self.fy, self.px, self.py, self.height, self.width)
        xyz_array = xyz_image.reshape((-1, 3))
        xyz_base = np.matmul(RT_camera[:3, :3], xyz_array.T) + RT_camera[:3, 3].reshape(3, 1)
        xyz_base = xyz_base.T.reshape((self.height, self.width, 3))
        return im_color, depth_image, xyz_image, xyz_base, RT_camera, self.intrinsics


# class to recieve images and segmentation labels
class MsmSegListener:

    def __init__(self, data_dir):

        self.im = None
        self.depth = None
        self.depth_frame_id = None
        self.depth_frame_stamp = None
        self.xyz_image = None
        self.label = None
        self.bbox = None
        self.counter = 0
        self.cv_bridge = CvBridge()
        self.base_frame = 'base_link'
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)        
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
        label_sub = message_filters.Subscriber('/seg_label_refined', Image, queue_size=10)
        score_sub = message_filters.Subscriber('/seg_score', Image, queue_size=10)          
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        self.camera_frame = 'head_camera_rgb_optical_frame'
        self.target_frame = self.base_frame        

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length    
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics
        print(intrinsics)
        
        # camera pose in base
        transform = self.tf_buffer.lookup_transform(self.base_frame,
                                           # source frame:
                                           self.camera_frame,
                                           # get the tf at the time the pose was valid
                                           rospy.Time(0),
                                           # wait for at most 1 second for transform, otherwise throw
                                           rospy.Duration(1.0)).transform
        quat = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        RT = quaternion_matrix(quat)
        RT[0, 3] = transform.translation.x
        RT[1, 3] = transform.translation.y        
        RT[2, 3] = transform.translation.z
        self.camera_pose = RT
        # print(self.camera_pose)

        queue_size = 1
        slop_seconds = 3.0
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, label_sub, score_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)
        
        # data saving directory
        now = datetime.datetime.now()
        seq_name = "listener_{:%m%dT%H%M%S}/".format(now)
        self.save_dir = os.path.join(data_dir, seq_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)        


    def callback_rgbd(self, rgb, depth, label, score):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        label = self.cv_bridge.imgmsg_to_cv2(label)
        score = self.cv_bridge.imgmsg_to_cv2(score)
        
        # compute xyz image
        height = depth_cv.shape[0]
        width = depth_cv.shape[1]
        xyz_image = compute_xyz(depth_cv, self.fx, self.fy, self.px, self.py, height, width)
        
        # compute the 3D bounding box of each object
        mask_ids = np.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        num = len(mask_ids)
        # print('%d objects segmented' % num)
        bbox = np.zeros((num, 8), dtype=np.float32)
        kernel = np.ones((3, 3), np.uint8)          
        for index, mask_id in enumerate(mask_ids):
            mask = np.array(label == mask_id).astype(np.uint8)
            
            # erode mask
            mask2 = cv2.erode(mask, kernel)
            
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 2, 1)
            # plt.imshow(mask)
            # ax = fig.add_subplot(1, 2, 2)
            # plt.imshow(mask2)
            # plt.show()               
            
            mask = (mask2 > 0) & (depth_cv > 0)
            points = xyz_image[mask, :]
            confidence = np.mean(score[mask])
            # convert points to robot base
            points_base = np.matmul(self.camera_pose[:3, :3], points.T) + self.camera_pose[:3, 3].reshape((3, 1))
            points_base = points_base.T
            center = np.mean(points_base, axis=0)
            if points_base.shape[0] > 0:
                x = np.max(points_base[:, 0]) - np.min(points_base[:, 0])
                y = np.max(points_base[:, 1]) - np.min(points_base[:, 1])
                # deal with noises in z values
                z = np.sort(points_base[:, 2])
                num = len(z)
                percent = 0.05
                lower = int(num * percent)
                upper = int(num * (1 - percent))
                if upper > lower:
                    z_selected = z[lower:upper]
                else:
                    z_selected = z
                z = np.max(z_selected) - np.min(z_selected)
            else:
                x = 0
                y = 0
                z = 0
            bbox[index, :3] = center
            bbox[index, 3] = x
            bbox[index, 4] = y
            bbox[index, 5] = z
            bbox[index, 6] = confidence
            bbox[index, 7] = mask_id
            
        # filter box
        index = bbox[:, 5] > 0
        bbox = bbox[index, :]

        with lock:
            self.im = im.copy()        
            self.label = label.copy()
            self.score = score.copy()
            self.depth = depth_cv.copy()
            self.depth_frame_id = depth.header.frame_id
            self.depth_frame_stamp = depth.header.stamp
            self.xyz_image = xyz_image
            self.bbox = bbox               

            
    # save data
    def save_data(self, step: int):
        # save meta data
        factor_depth = 1000.0        
        meta = {'intrinsic_matrix': self.intrinsics, 'factor_depth': factor_depth, 'camera_pose': self.camera_pose}
        filename = self.save_dir + 'meta-{:06}.mat'.format(step)
        savemat(filename, meta, do_compression=True)
        print('save data to {}'.format(filename))

        # convert depth to unit16
        depth_save = np.array(self.depth * factor_depth, dtype=np.uint16)

        # segmentation label image
        im_label = visualize_segmentation(self.im, self.label, return_rgb=True)

        save_name_rgb = self.save_dir + 'color-{:06}.png'.format(step)
        save_name_depth = self.save_dir + 'depth-{:06}.png'.format(step)
        save_name_label = self.save_dir + 'label-{:06}.png'.format(step)
        save_name_label_image = self.save_dir + 'pred-{:06}.png'.format(step)
        save_name_score = self.save_dir + 'score-{:06}.png'.format(step)   
        save_name_bbox = self.save_dir + 'bbox-{:06}.npy'.format(step)     
        cv2.imwrite(save_name_rgb, self.im)
        cv2.imwrite(save_name_depth, depth_save)
        cv2.imwrite(save_name_label, self.label.astype(np.uint8))
        cv2.imwrite(save_name_label_image, im_label)
        cv2.imwrite(save_name_score, self.score.astype(np.uint8))
        np.save(save_name_bbox, self.bbox)


# class to recieve images and segmentation labels
class UoisSegListener:

    def __init__(self, data_dir):

        self.im = None
        self.depth = None
        self.depth_frame_id = None
        self.depth_frame_stamp = None
        self.xyz_image = None
        self.label = None
        self.bbox = None
        self.counter = 0
        self.cv_bridge = CvBridge()
        self.base_frame = 'base_link'
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)        
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
        label_sub = message_filters.Subscriber('/seg_label_refined', Image, queue_size=10)
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        self.camera_frame = 'head_camera_rgb_optical_frame'
        self.target_frame = self.base_frame        

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length    
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics
        print("Camera Intrinsics: ", intrinsics)
        
        # camera pose in base
        transform = self.tf_buffer.lookup_transform(self.base_frame,
                                           # source frame:
                                           self.camera_frame,
                                           # get the tf at the time the pose was valid
                                           rospy.Time(0),
                                           # wait for at most 1 second for transform, otherwise throw
                                           rospy.Duration(1.0)).transform
        quat = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        RT = quaternion_matrix(quat)
        RT[0, 3] = transform.translation.x
        RT[1, 3] = transform.translation.y        
        RT[2, 3] = transform.translation.z
        self.camera_pose = RT
        # print(self.camera_pose)

        queue_size = 1
        slop_seconds = 3.0
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, label_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)
        
        # data saving directory
        now = datetime.datetime.now()
        seq_name = "listener_{:%m%dT%H%M%S}/".format(now)
        self.save_dir = os.path.join(data_dir, seq_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)        


    def callback_rgbd(self, rgb, depth, label):

        if depth.encoding == '32FC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
        label = self.cv_bridge.imgmsg_to_cv2(label)
        # score = self.cv_bridge.imgmsg_to_cv2(score)
        
        # compute xyz image
        height = depth_cv.shape[0]
        width = depth_cv.shape[1]
        xyz_image = compute_xyz(depth_cv, self.fx, self.fy, self.px, self.py, height, width)
        
        # compute the 3D bounding box of each object
        mask_ids = np.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        num = len(mask_ids)
        # print('%d objects segmented' % num)
        bbox = np.zeros((num, 8), dtype=np.float32)
        kernel = np.ones((3, 3), np.uint8)          
        for index, mask_id in enumerate(mask_ids):
            mask = np.array(label == mask_id).astype(np.uint8)
            
            # erode mask
            mask2 = cv2.erode(mask, kernel)
            
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 2, 1)
            # plt.imshow(mask)
            # ax = fig.add_subplot(1, 2, 2)
            # plt.imshow(mask2)
            # plt.show()               
            
            mask = (mask2 > 0) & (depth_cv > 0)
            points = xyz_image[mask, :]
            confidence = -1 # np.mean(score[mask])
            # convert points to robot base
            points_base = np.matmul(self.camera_pose[:3, :3], points.T) + self.camera_pose[:3, 3].reshape((3, 1))
            points_base = points_base.T
            center = np.mean(points_base, axis=0)
            if points_base.shape[0] > 0:
                x = np.max(points_base[:, 0]) - np.min(points_base[:, 0])
                y = np.max(points_base[:, 1]) - np.min(points_base[:, 1])
                # deal with noises in z values
                z = np.sort(points_base[:, 2])
                num = len(z)
                percent = 0.05
                lower = int(num * percent)
                upper = int(num * (1 - percent))
                if upper > lower:
                    z_selected = z[lower:upper]
                else:
                    z_selected = z
                z = np.max(z_selected) - np.min(z_selected)
            else:
                x = 0
                y = 0
                z = 0
            bbox[index, :3] = center
            bbox[index, 3] = x
            bbox[index, 4] = y
            bbox[index, 5] = z
            bbox[index, 6] = confidence
            bbox[index, 7] = mask_id
            
        # filter box
        index = bbox[:, 5] > 0
        bbox = bbox[index, :]

        with lock:
            self.im = im.copy()        
            self.label = label.copy()
            self.score = -1 # score.copy()
            self.depth = depth_cv.copy()
            self.depth_frame_id = depth.header.frame_id
            self.depth_frame_stamp = depth.header.stamp
            self.xyz_image = xyz_image
            self.bbox = bbox               

            
    # save data
    def save_data(self, step: int):
        # save meta data
        factor_depth = 1000.0        
        meta = {'intrinsic_matrix': self.intrinsics, 'factor_depth': factor_depth, 'camera_pose': self.camera_pose}
        filename = self.save_dir + 'meta-{:06}.mat'.format(step)
        savemat(filename, meta, do_compression=True)
        print('save data to {}'.format(filename))

        # convert depth to unit16
        depth_save = np.array(self.depth * factor_depth, dtype=np.uint16)

        # segmentation label image
        im_label = visualize_segmentation(self.im, self.label, return_rgb=True)

        save_name_rgb = self.save_dir + 'color-{:06}.png'.format(step)
        save_name_depth = self.save_dir + 'depth-{:06}.png'.format(step)
        save_name_label = self.save_dir + 'label-{:06}.png'.format(step)
        save_name_label_image = self.save_dir + 'pred-{:06}.png'.format(step)        
        # save_name_score = self.save_dir + 'score-{:06}.png'.format(step)   
        save_name_bbox = self.save_dir + 'bbox-{:06}.npy'.format(step)     
        cv2.imwrite(save_name_rgb, self.im)
        cv2.imwrite(save_name_depth, depth_save)
        cv2.imwrite(save_name_label, self.label.astype(np.uint8))
        cv2.imwrite(save_name_label_image, im_label)
        # cv2.imwrite(save_name_score, self.score.astype(np.uint8))
        np.save(save_name_bbox, self.bbox)


# class to publish rgb, depth, and masked label images
class CropImgPublisher:
    def __init__(self, rgb_img, depth_img, label):

        self.im = rgb_img
        self.depth = depth_img
        self.label = label
        self.counter = 0
        self.cv_bridge = CvBridge()

        self.rgb_pub = rospy.Publisher('/selected_rgb', Image, queue_size=10)        
        self.depth_pub = rospy.Publisher('/selected_depth', Image, queue_size=10)
        self.label_pub = rospy.Publisher('/selected_label', Image, queue_size=10)

        self.imrgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_img, 'rgb8')
        self.label_msg = self.cv_bridge.cv2_to_imgmsg(label.astype(np.uint8))

    def run(self):
        while True:
            imrgb_c = self.rgb_pub.get_num_connections()
            depth_c = self.depth_pub.get_num_connections()
            label_c = self.label_pub.get_num_connections()
            rospy.loginfo(f"rgb, depth, lab pub connections: {imrgb_c}, {depth_c}, {label_c}")
            if (imrgb_c > 0) and (depth_c > 0) and (label_c > 0):
                print("Found a subscriber for the crop img message!")
                # TODO: Implement publishing of rgb, depth and selected_label image
                break


# class to publish point cloud corresponding to an object
class ObjPointPublisher:

    def __init__(self, data_dir):
        self.points_pub = rospy.Publisher('/selected_objpts', PointCloud, queue_size=5)
        self.pc_all_pub = rospy.Publisher('/all_objpts_cam', PointCloud, queue_size=5)
        self.points_pub_base = rospy.Publisher('/selected_objpts_base', PointCloud, queue_size=5)
        # data saving directory
        now = datetime.datetime.now()
        seq_name = "pointpublisher_{:%m%dT%H%M%S}/".format(now)
        self.save_dir = os.path.join(data_dir, seq_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)    

    def save_data(self, object_points, step, pc_all=None):
        save_name_pts = self.save_dir + 'objpts-{:06}.npy'.format(step)
        save_name_pts_all = os.path.join(self.save_dir, 'pc_all-{:06}.npy'.format(step))
        np.save(save_name_pts, object_points)
        if np.any(pc_all):
            np.save(save_name_pts_all, pc_all)

    def run(self, object_points, obj_pts_base=None, pc_all=None):
        """
        object_points : in camera frame
        pc_all : entire scene's point cloud in camera frame
        """
        points_msg = PointCloud()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'head_camera_rgb_optical_frame'
        points_msg.header = header
        for i in range(object_points.shape[0]):
            pt = object_points[i]
            points_msg.points.append(Point(pt[0], pt[1], pt[2]))

        if obj_pts_base is not None:
            # publish pts in base frame
            pts_msg_base = PointCloud()
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'base_link'
            pts_msg_base.header = header
            for i in range(obj_pts_base.shape[0]):
                pt = obj_pts_base[i]
                pts_msg_base.points.append(Point(pt[0], pt[1], pt[2]))

        if pc_all is not None:
            # publish entire scene's point cloud
            pts_msg_pc_all = PointCloud()
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'head_camera_rgb_optical_frame'
            pts_msg_pc_all.header = header
            for i in range(pc_all.shape[0]):
                pt = pc_all[i]
                pts_msg_pc_all.points.append(Point(pt[0], pt[1], pt[2]))
            print(f"LEN: {len(pts_msg_pc_all.points)} points published for entire scene!")
        # self.points_pub.publish(points_msg)
        # num_c = 1
        while True:
            num_c = self.points_pub.get_num_connections()
            # print(num_c)
            # rospy.loginfo(f"points pub connections: {num_c}")
            if num_c:
                print("Publishing the PointCloud message!")
                self.points_pub.publish(points_msg)                
                if obj_pts_base is not None:
                    self.points_pub_base.publish(pts_msg_base)
                    print("Finished publishing PC msg for obj points in base frame!")
                if pc_all is not None:
                    self.pc_all_pub.publish(pts_msg_pc_all)
                    print("Finished publishing PC msg for ENTIRE SCENCE PC in camera frame!")
                break


# class to listen to a pose array consisting of different grasp poses
class GraspPoseListener:

    def __init__(self, data_dir):
        self.grasp_poses = None
        self.prev_poses_mean = None
        self.counter = 0
        self.cv_bridge = CvBridge()
        self.base_frame = 'base_link'
        pose_sub = message_filters.Subscriber('/pose_6dof', PoseArray, queue_size=10) 
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        self.camera_frame = 'head_camera_rgb_optical_frame'
        self.target_frame = self.base_frame        

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length    
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        self.intrinsics = intrinsics
        print("Camera Intrinsics: ", intrinsics)
        
        # camera pose in base
        transform = self.tf_buffer.lookup_transform(self.base_frame,
                                           # source frame:
                                           self.camera_frame,
                                           # get the tf at the time the pose was valid
                                           rospy.Time(0),
                                           # wait for at most 1 second for transform, otherwise throw
                                           rospy.Duration(1.0)).transform
        quat = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        RT = quaternion_matrix(quat)
        RT[0, 3] = transform.translation.x
        RT[1, 3] = transform.translation.y        
        RT[2, 3] = transform.translation.z
        self.camera_pose = RT
        # print(self.camera_pose)

        queue_size = 1
        slop_seconds = 3.0
        ts = message_filters.ApproximateTimeSynchronizer([pose_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_pose)
        
        # data saving directory
        now = datetime.datetime.now()
        seq_name = "poselistener_{:%m%dT%H%M%S}/".format(now)
        self.save_dir = os.path.join(data_dir, seq_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)        

    def callback_pose(self, pose_array):
        n = len(pose_array.poses)
        if n == 0:
            # Catch this when listening to grasp poses in main grasping loop
            # Init with an empty array
            self.grasp_poses = np.zeros((0,4,4))
            print("NO SUITABLE GRASP POSES FOUND!")
            return
        assert n > 0
        grasp_poses = np.zeros((n, 4, 4), dtype=np.float32)
        for i, pose in enumerate(pose_array.poses):
            grasp_poses[i, :, :] = ros_pose_to_rt(pose)
        # with lock:
        curr_mean = np.mean(grasp_poses[:, :3, 3], axis=0)
        if self.prev_poses_mean is None:
            self.prev_poses_mean = curr_mean
        else:
            dist = np.linalg.norm(self.prev_poses_mean - curr_mean)
            if dist > 0.08:
                print("Grasp poses mean differ by ", dist)
                self.prev_poses_mean = curr_mean
        self.grasp_poses = grasp_poses.copy()
 
    # save data
    def save_data(self, step: int):
        # save meta data
        factor_depth = 1000.0        
        meta = {'intrinsic_matrix': self.intrinsics, 'factor_depth': factor_depth, 'camera_pose': self.camera_pose}
        filename = self.save_dir + 'posemeta-{:06}.mat'.format(step)
        savemat(filename, meta, do_compression=True)
        print('save data to {}'.format(filename))
        save_name_pose = self.save_dir + 'grasp_poses-{:06}.npy'.format(step)
        np.save(save_name_pose, self.grasp_poses)


def test_basic_img():
    # image listener
    rospy.init_node("image_listener")    
    listener = ImageListener()
    while 1:
        im_color, depth_image, xyz_image, xyz_base, RT_camera, intrinsics = listener.get_data()
        if im_color is None:
            continue
    
        # visualization
        fig = plt.figure()
    
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(im_color)
    
        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(depth_image)
    
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        pc = xyz_image.reshape((-1, 3))
        index = np.isfinite(pc[:, 2])
        pc = pc[index, :]
        n = pc.shape[0]
        index = np.random.choice(n, 5000)        
        ax.scatter(pc[index, 0], pc[index, 1], pc[index, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('sampled point cloud in camera frame')
        
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        pc = xyz_base.reshape((-1, 3))
        index = np.isfinite(pc[:, 2])
        pc = pc[index, :]
        n = pc.shape[0]
        index = np.random.choice(n, 5000)        
        ax.scatter(pc[index, 0], pc[index, 1], pc[index, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title('sampled point cloud in base frame')
        plt.show()


def test_point_publisher():
    rospy.init_node("point_publisher")    
    datadir = "../data/graspnet_test/"
    publisher = ObjPointPublisher(datadir)
    for npy_file in glob.glob(os.path.join(datadir, "obj_npy_data", "*.npy")):
        data = np.load(npy_file, allow_pickle=True, encoding="latin1").item()
        object_pc = data['smoothed_object_pc']
        print("Loaded object pc...waiting to publish")
        publisher.run(object_pc)
        print("Finished publish object's pc info\n")
        _tmp = input("Waiting to go to next object???")

if __name__ == '__main__':
    # test_basic_img()
    test_point_publisher()

