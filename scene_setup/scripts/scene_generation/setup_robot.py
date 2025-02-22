import sys, argparse
import numpy as np
from scipy.io import savemat
import rospy

sys.path.append("../utils/")
from utils_control import FollowTrajectoryClient, PointHeadClient


def make_args():
    parser = argparse.ArgumentParser(
        description="Make the robot lift torse and look at a particular point in space",
        add_help=True,
    )
    parser.add_argument(
        "-x",
        "--xloc",
        type=float,
        default=0.45,
        help="X-coordinate to which robot will look at",
    )
    parser.add_argument(
        "-y",
        "--yloc",
        type=float,
        default=0,
        help="Y-coordinate to which robot will look at",
    )
    parser.add_argument(
        "-z",
        "--zloc",
        type=float,
        default=0.75,
        help="Z-coordinate to which robot will look at",
    )
    args = parser.parse_args()
    return args


def main(args):
    xloc = args.xloc
    yloc = args.yloc
    zloc = args.zloc
    rospy.init_node("viz")
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    head_action = PointHeadClient()
    rospy.loginfo("Raising torso...")
    torso_action.move_to(
        [
            0.4,
        ]
    )
    for _ in range(5):
        head_action.look_at(xloc, yloc, zloc, "base_link")


if __name__ == "__main__":
    args = make_args()
    main(args)
