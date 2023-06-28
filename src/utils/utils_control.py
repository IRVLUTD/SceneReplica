import numpy as np
import rospy
import actionlib

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from control_msgs.msg import PointHeadAction, PointHeadGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

# Send a trajectory to controller
class FollowTrajectoryClient(object):
    def __init__(self, name, joint_names):
        self.client = actionlib.SimpleActionClient(
            "%s/follow_joint_trajectory" % name, FollowJointTrajectoryAction
        )
        rospy.loginfo("Waiting for %s..." % name)
        self.client.wait_for_server()
        self.joint_names = joint_names

    def move_to(self, positions, duration=5.0):
        if len(self.joint_names) != len(positions):
            print("Invalid trajectory position")
            return False
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = positions
        trajectory.points[0].velocities = [0.0 for _ in positions]
        trajectory.points[0].accelerations = [0.0 for _ in positions]
        trajectory.points[0].time_from_start = rospy.Duration(duration)
        follow_goal = FollowJointTrajectoryGoal()
        follow_goal.trajectory = trajectory

        self.client.send_goal(follow_goal)
        self.client.wait_for_result()


# Point the head using controller
class PointHeadClient(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient(
            "head_controller/point_head", PointHeadAction
        )
        rospy.loginfo("Waiting for head_controller...")
        self.client.wait_for_server()

    def look_at(self, x, y, z, frame, duration=1.0):
        """
        Turning head to look at x,y,z
        :param x: x location
        :param y: y location
        :param z: z location
        :param frame: the frame of reference
        :param duration: given time for operation to calcualte the motion plan
        :return:
        """
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()


class JointListener:

    """
    Listens on a particular message topic.
    """

    def __init__(self, topic_name_joint="/joint_states", queue_size=100):
        self.topic_name_joint = topic_name_joint
        self.robot_state = None
        self.joint_position = None
        self._sub = rospy.Subscriber(
            self.topic_name_joint,
            JointState,
            self.robot_state_callback,
            queue_size=queue_size,
        )

    def robot_state_callback(self, data):
        self.robot_state = data
        self.joint_position = np.array(data.position)
