#!/usr/bin/env python
import sys, os
import cv2
import scipy.io
import rospy
import numpy as np
import tf

from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class ImageListener:
    def __init__(self):
        self.object_names = (
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "006_mustard_bottle",
            "007_tuna_fish_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
            "011_banana",
            "021_bleach_cleanser",
            "024_bowl",
            "025_mug",
            "035_power_drill",
            "037_scissors",
            "040_large_marker",
            "052_extra_large_clamp",
        )
        self.num_objects = len(self.object_names)
        self.data_reference = None
        self.cv_bridge = CvBridge()
        self.br = tf.TransformBroadcaster()

        # pose image publisher
        self.pose_image_pub = rospy.Publisher("image_overlay", Image, queue_size=1)

        # marker publishers
        self.marker_pubs = []
        for i in range(self.num_objects):
            name = "marker_" + self.object_names[i]
            self.marker_pubs.append(rospy.Publisher(name, Marker, queue_size=2))

        rgb_sub = rospy.Subscriber(
            "/head_camera/rgb/image_raw", Image, self.callback, queue_size=2
        )

    def setup_reference(self, data):
        self.data_reference = data

    def callback(self, rgb):
        # get color images
        im = self.cv_bridge.imgmsg_to_cv2(rgb, "bgr8")

        if self.data_reference is not None:
            # publish pose image
            image_reference = self.data_reference["image_pose"]
            image_disp = 0.4 * im.astype(np.float32) + 0.6 * image_reference.astype(
                np.float32
            )
            image_disp = np.clip(image_disp, 0, 255).astype(np.uint8)

            pose_msg = self.cv_bridge.cv2_to_imgmsg(image_disp)
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = rgb.header.frame_id
            pose_msg.encoding = "bgr8"
            self.pose_image_pub.publish(pose_msg)
            print("publish reference image at /image_overlay")

            # publish object markers
            object_names = self.data_reference["object_names"]
            for i in range(len(object_names)):
                object_name = object_names[i].strip()
                pose = self.data_reference["poses"][i]

                # publish tf
                t_bo = pose[:3]
                q_bo = pose[3:]  # assumes quat is sent as (w,x,y,z) format
                name = "reference/" + object_name
                self.br.sendTransform(
                    t_bo,
                    [q_bo[1], q_bo[2], q_bo[3], q_bo[0]],
                    rospy.Time.now(),
                    name,
                    "base_link",
                )

                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = rospy.Time.now()
                marker.type = 10
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.pose.position.x = pose[0]
                marker.pose.position.y = pose[1]
                marker.pose.position.z = pose[2]
                marker.pose.orientation.w = pose[3]
                marker.pose.orientation.x = pose[4]
                marker.pose.orientation.y = pose[5]
                marker.pose.orientation.z = pose[6]

                marker.mesh_resource = os.path.join(
                    "package://fetch_gazebo/models/", object_name, "textured_simple.obj"
                )
                print(marker.mesh_resource)
                marker.mesh_use_embedded_materials = True
                index = self.object_names.index(object_name)
                self.marker_pubs[index].publish(marker)
        else:
            print("no reference data")


def read_data(dirname, index):
    # color image
    im_file = os.path.join(dirname, "color-%06d.png" % index)
    im = cv2.imread(im_file)
    print(im_file)

    # pose image
    filename = os.path.join(dirname, "pose-%06d.png" % index)
    im_pose = cv2.imread(filename)

    # # depth image
    # filename = os.path.join(dirname, "depth-%06d.png" % index)
    # depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)

    # meta data
    filename = os.path.join(dirname, "meta-%06d.mat" % index)
    data = scipy.io.loadmat(filename)
    # factor_depth = data["factor_depth"]
    # depth /= factor_depth

    # contruct data
    data["image"] = im
    data["image_pose"] = im_pose
    # data["depth"] = depth
    return data


import argparse


def make_args():
    parser = argparse.ArgumentParser(description="Test scene setup", add_help=True)
    parser.add_argument("-i", "--index", type=int, default=0)
    parser.add_argument(
        "-d",
        "--datadir",
        type=str,
        default="/home/benchmark/Projects/posecnn-pytorch/data/Fetch/",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    Main function to run the code
    """
    args = make_args()
    rospy.init_node("setup_scene")
    dirname = args.datadir
    # read a reference scene
    index = args.index
    data = read_data(dirname, index)

    # image listener
    listener = ImageListener()
    listener.setup_reference(data)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
