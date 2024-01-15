import rospy
import tf2_ros
import numpy as np
from tf.transformations import quaternion_matrix


def get_tf_pose(target_frame, base_frame=None, is_matrix=False):
    transform = tf_buffer.lookup_transform(
        base_frame, target_frame, rospy.Time.now(), rospy.Duration(1.0)).transform
    quat = [
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ]
    RT_obj = quaternion_matrix(quat)
    RT_obj[:3, 3] = np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z]
    )
    return RT_obj


if __name__ == "__main__":

    # Create a node
    rospy.init_node("query_pose")
    tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(2)

    topic_name = 'head_camera_rgb_optical_frame'
    RT = get_tf_pose(topic_name, 'base_link')
    print(RT)