#!/usr/bin/env python
import sys
import rospy
import argparse
import numpy as np

from geometry_msgs.msg import PoseStamped
import rospy
import sys
import moveit_commander
from geometry_msgs.msg import PoseStamped

from utils_control import FollowTrajectoryClient
from grasp_utils import user_confirmation


def reset_arm_tuck(group):
    """
    reset arm to a tucking position for pushing
    :param group: the moveit group of joints
    :return:
    """
    pose = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.1]
    group.set_joint_value_target(pose)
    plan = group.plan()
    if not plan[0]:
        print("NO PLAN FOUND FOR Tuck! Exiting...")
        sys.exit()
    if user_confirmation("Resetting arm to tuck position"):
        pass
    else:
        sys.exit(1)
    # group.go(pose, wait=True)
    trajectory = plan[1]
    input("execute? Tucking?")
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()
    rospy.sleep(1)
    return


def reset_arm_stow(group, direction="right"):
    """
    reset arm to a stowing position for pushing
    :param group: the moveit group of joints
    :return:
    """

    if direction == "right":
        pose = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.1]  # right stow
    else:
        pose = [-1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.1]  # left stow

    group.set_joint_value_target(pose)
    group.plan()
    if user_confirmation("Resetting arm to stow position"):
        pass
    else:
        sys.exit(1)
    group.go(pose, wait=True)
    rospy.sleep(2)
    return


def rotate_shoulder(group, angle: float):
    """
    Rotates the shoulder joint by some degrees
    angle: rotation angle in degrees
    """
    group.stop()
    group.clear_pose_targets()
    joint_goal = group.get_current_joint_values()
    joint_goal[0] = min(1.32, np.radians(angle))
    group.go(joint_goal, wait=True)
    group.stop()
    group.clear_pose_targets()
    rospy.sleep(0.25)


def make_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a PoseCNN network")
    parser.add_argument("--stow", dest="stow", help="Stow arm", action="store_true")
    parser.add_argument("--sdir", type=str, help="Direction to Stow", default="right")
    parser.add_argument("--tuck", dest="tuck", help="Tuck arm", action="store_true")
    parser.add_argument("--rotate", action="store_true", help="Rotate Shoulder")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()

    if args.tuck and args.stow:
        print("Give either stow or tuck, not both!")
        sys.exit()
    if args.sdir not in {"left", "right"}:
        print("Give 'left' or 'right' as the stow direction")
        sys.exit()

    stow_dir = args.sdir

    rospy.init_node("StowingOrTucking")
    group = moveit_commander.MoveGroupCommander("arm")
    group.set_max_velocity_scaling_factor(1)
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()

    rospy.sleep(2)
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()
    p.pose.position.x = 0
    p.pose.position.y = 0
    p.pose.position.z = -0.13
    scene.add_box("base", p, (1.3, 1, 1))

    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    torso_action.move_to(
        [
            0.4,
        ]
    )

    if args.stow:
        rospy.loginfo("Stowing arm")
        reset_arm_stow(group, args.sdir)
    elif args.tuck:
        reset_arm_tuck(group)
    elif args.rotate:
        rotate_shoulder(group, 60)
