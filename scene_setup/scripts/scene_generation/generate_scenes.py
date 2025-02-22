import sys, os
import argparse
import pickle
import datetime
import numpy as np

import rospy
import moveit_commander
import geometry_msgs.msg
import tf.transformations as tra
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetModelState

sys.path.append("../utils/")
from utils_control import FollowTrajectoryClient, PointHeadClient
from ros_utils import ros_pose_to_rt, rt_to_ros_qt, rt_to_ros_pose
from gripper import Gripper
from grasp_utils import parse_grasps, sort_grasps, sort_and_filter_grasps
from stow_or_tuck_arm import reset_arm_stow
from utils_scene import WorldService, ObjectService, SceneMaker


def plan_to_pose(group, quat, trans):
    """
    quat is with format (x, y, z, w) for a quanternion
    trans is the 3D translation (x, y, z)
    group is the moveit group interface
    """
    # use moveit to plan to trajectory towards the gripper pose defined by (quat, trans)
    # refer to https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html

    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = trans[0]
    target_pose.position.y = trans[1]
    target_pose.position.z = trans[2]
    target_pose.orientation.x = quat[0]
    target_pose.orientation.y = quat[1]
    target_pose.orientation.z = quat[2]
    target_pose.orientation.w = quat[3]

    group.set_pose_target(target_pose)

    plan = group.plan()
    return plan


def plan_grasp(group, RT_grasps_base, grasp_index):
    """
    RT_grasps_base is with shape (50, 4, 4): 50 grasps in the robot base frame
    The plan_grasp function tries to plan a trajectory to each grasp. It stops when a plan is found.
    A standoff is a gripper pose with a short distance along x-axis of the gripper frame before grasping the object.
    """
    # number of grasps
    n = RT_grasps_base.shape[0]
    reach_tail_len = 10
    # define the standoff distance as 10cm
    standoff_dist = 0.10

    # compute standoff pose
    offset = -standoff_dist * np.linspace(0, 1, reach_tail_len, endpoint=False)[::-1]
    offset = np.append(offset, [0.04])

    reach_tail_len += 1
    pose_standoff = np.tile(np.eye(4), (reach_tail_len, 1, 1))
    pose_standoff[:, 0, 3] = offset

    # for each grasp
    for i in range(n):
        RT_grasp = RT_grasps_base[i]
        grasp_idx = grasp_index[i]

        standoff_grasp_global = np.matmul(RT_grasp, pose_standoff)

        # Calling `stop()` ensures that there is no residual movement
        group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        group.clear_pose_targets()

        # plan to the standoff
        quat, trans = rt_to_ros_qt(standoff_grasp_global[0, :, :])  # xyzw for quat
        plan = plan_to_pose(group, quat, trans)
        trajectory = plan[1]
        if plan[0]:
            # print("found a plan for grasp")
            # print(RT_grasp)
            # print("grasp idx", grasp_idx)
            # print("grasp index", grasp_index)
            return RT_grasp, grasp_idx

    return None, -1


def grasp(gripper, group, scene, object_name, RT_grasp):
    # first plan to the standoff pose, then move the the grasping pose
    reach_tail_len = 10
    standoff_dist = 0.10

    # compute standoff pose
    offset = -standoff_dist * np.linspace(0, 1, reach_tail_len, endpoint=False)[::-1]
    offset = np.append(offset, [0.04])

    reach_tail_len += 1
    pose_standoff = np.tile(np.eye(4), (reach_tail_len, 1, 1))
    pose_standoff[:, 0, 3] = offset

    # plan to grasp
    standoff_grasp_global = np.matmul(RT_grasp, pose_standoff)

    # Calling `stop()` ensures that there is no residual movement
    group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    group.clear_pose_targets()

    # plan to the standoff
    quat, trans = rt_to_ros_qt(standoff_grasp_global[0, :, :])  # xyzw for quat
    plan = plan_to_pose(group, quat, trans)
    trajectory = plan[1]
    if not plan[0]:
        print("no plan found")
        return 0
    # else:
    #     return 1
    input("execute?")
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()

    # # remove the target from the planning scene for grasping
    # scene.remove_world_object(object_name)

    # waypoints = []
    # wpose = group.get_current_pose().pose
    # for i in range(1, reach_tail_len):
    #     wpose = rt_to_ros_pose(wpose, standoff_grasp_global[i])
    #     # print(wpose)
    #     waypoints.append(copy.deepcopy(wpose))
    # (plan_standoff, fraction) = group.compute_cartesian_path(
    #     waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
    # )  # jump_threshold
    # trajectory = plan_standoff

    # input("execute?")
    # group.execute(trajectory, wait=True)
    # group.stop()
    # group.clear_pose_targets()

    print("Stowing the Gripper...")
    reset_arm_stow(group)
    return 1


# lift the robot arm
def lift_arm(group):
    # lift the object
    offset = -0.2
    rospy.loginfo("lift object")
    pose = group.get_current_joint_values()
    pose[1] += offset
    group.set_joint_value_target(pose)
    plan = group.plan()

    if not plan[0]:
        print("no plan found in lifting")
        sys.exit(1)

    input("execute?")
    trajectory = plan[1]
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()


def get_pose_gazebo(model_name, relative_entity_name=""):
    # Query pose of frames from the Gazebo environment

    def gms_client(model_name, relative_entity_name):
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            gms = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            resp1 = gms(model_name, relative_entity_name)
            return resp1
        except (rospy.ServiceException, e):
            print("Service call failed: %s" % e)

    # query the object pose in Gazebo world T_wo
    res = gms_client(model_name, relative_entity_name)
    T_wo = ros_pose_to_rt(res.pose)

    # query fetch base link pose in Gazebo world T_wb
    res = gms_client(model_name="fetch", relative_entity_name="base_link")
    T_wb = ros_pose_to_rt(res.pose)

    T_bo = np.linalg.inv(T_wb) @ T_wo
    return T_bo


def plan_all_grasps(group, object_name, grasp_dir, gripper, scene, try_grasp=False):
    """
    Returns the success code, grasp index
    """
    # Get object pose
    RT_obj = get_pose_gazebo(model_name=object_name)
    # trans = RT_obj[:3, 3]
    # print(f"RT_obj trans: {trans}")
    # Load grasps
    grasp_filename = os.path.join(grasp_dir, f"fetch_gripper-{object_name}.json")
    RT_grasps = parse_grasps(grasp_filename)
    # print(RT_grasps.shape)  # (N, 4, 4) grasps in object centric frame.
    # Current gripper pose
    RT_gripper = get_pose_gazebo(
        model_name="fetch", relative_entity_name="wrist_roll_link"
    )
    # Sort grasps according to distances to gripper ; RT_grasps_base contains
    # all the grasps in the robot base frame
    # RT_grasps_base, grasp_index = sort_grasps(RT_obj, RT_gripper, RT_grasps)
    RT_grasps_base, grasp_index, _ = sort_and_filter_grasps(
        RT_obj, RT_gripper, RT_grasps, table_height=TABLE_HEIGHT
    )
    RT_grasp, g_idx = plan_grasp(group, RT_grasps_base, grasp_index)
    if RT_grasp is not None:
        print(f"{object_name}: Found a plan with {g_idx} g_idx!")
        if try_grasp:
            _status = grasp(gripper, group, scene, object_name, RT_grasp)
        return 1, g_idx
    else:
        print(f"{object_name}: Found NO plans!")
        return 0, -1


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/home/benchmark/Datasets/benchmarking/",
        help="Path to root data dir",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=7,
        help="Number of rows in the grid for spawning objects on table",
    )
    parser.add_argument(
        "-c",
        "--cols",
        type=int,
        default=7,
        help="Number of cols in the grid for spawning objects on table",
    )
    parser.add_argument(
        "-g",
        "--grasp",
        action="store_true",
        help="Use this flag if you also need to execute the motion plan to a standoff grasp pose",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()
    data_dir = args.data_dir
    grid_r = args.rows
    grid_c = args.cols
    grasp_flag = args.grasp

    models_path = os.path.join(data_dir, "gazebo_models_all_simple")
    grasp_dir = os.path.join(data_dir, "grasp_data", "refined_grasps")
    output_dir = os.path.join(
        data_dir,
        "scene_gen",
        "iter_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    # Create a directory to store the scene generation results
    os.makedirs(output_dir)

    ws = WorldService()
    objs = ObjectService(models_base_path=models_path)

    z_offset = -0.03  # difference between Real World and Gazebo table
    TABLE_HEIGHT = 0.78 + z_offset
    grid_size = (grid_r, grid_c)
    table_position = [0.8, 0, z_offset]

    objs.add_object(
        "cafe_table_org",
        [*table_position, 0, 0, 0, 1],
    )
    model_names = [
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
        # "051_large_clamp",
        "052_extra_large_clamp",
    ]
    scene_m = SceneMaker(
        model_names,
        models_path,
        grid_size,
        table_position,
        TABLE_HEIGHT,
        stable_pose_f=os.path.join(data_dir, "pose_data", "selected_poses.pk"),
    )

    # ----------------- Motion Planning ------------- #
    # Create a node
    rospy.init_node("fetch_grasping")
    # # Setup clients
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    head_action = PointHeadClient()
    gripper = Gripper()
    # Raise the torso using just a controller
    rospy.loginfo("Raising torso...")
    torso_action.move_to(
        [
            0.4,
        ]
    )
    # --------- initialize moveit components ------
    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander("arm")
    scene = moveit_commander.PlanningSceneInterface()
    scene.clear()
    robot = moveit_commander.RobotCommander()
    # look at table
    head_action.look_at(0.45, 0, 0.75, "base_link")

    def setup_scene_base(scene, table_height, object_names, models_path):
        """
        Sets the Motion Planning scene for MoveIt!
        """
        # Clear Planning Scene
        scene.clear()
        rospy.sleep(1)
        scene.remove_world_object()
        rospy.sleep(1)

        # # sleep before adding objects
        # # dimension of each default(1,1,1) box is 1x1x1m
        # -------- planning scene set-up -------
        rospy.loginfo("adding table object into planning scene")
        print("adding table object into planning scene")
        rospy.sleep(1.0)
        p = PoseStamped()
        p.header.frame_id = robot.get_planning_frame()
        p.pose.position.x = table_position[0]
        p.pose.position.y = 0
        p.pose.position.z = (
            table_height - 0.5
        )  # 0.5 = half length of moveit obstacle box
        scene.add_box("table", p, (1, 5, 1))
        # add a box for robot base
        p.pose.position.x = 0
        p.pose.position.y = 0
        p.pose.position.z = 0.18
        scene.add_box("base", p, (0.56, 0.56, 0.4))

        for obj_name in object_names:
            RT_obj = get_pose_gazebo(model_name=obj_name)
            p.pose = rt_to_ros_pose(p.pose, RT_obj)
            pose = [
                p.pose.position.x,
                p.pose.position.y,
                p.pose.position.z,
                p.pose.orientation.x,
                p.pose.orientation.y,
                p.pose.orientation.z,
                p.pose.orientation.w,
            ]
            # print(f"gazebo {obj_name} pose: {pose}")
            obj_mesh_path = os.path.join(models_path, obj_name, "textured_simple.obj")
            if not os.path.exists(obj_mesh_path):
                print(f"Given mesh path does not exist! {obj_mesh_path}")
            scene.add_mesh(obj_name, p, obj_mesh_path)

    scene_count = 0
    MAX_SCENE_COUNT = 2000
    while scene_count < MAX_SCENE_COUNT:
        print("-----------------------------")
        sample_scene, stable_pose_idx = scene_m.create_scene()
        object_names = list(sample_scene.keys())
        # add objects to gazebo scene
        for obj, pose in sample_scene.items():
            # print(f"adding {obj} pose: {pose}")
            objs.add_object(obj, pose)

        # confirmation = input("Continue with planning? y or n...")
        # if confirmation == "y":
        total_success = 0
        # Setup base scene elements: table and robot base collision boxes
        setup_scene_base(scene, TABLE_HEIGHT, object_names, models_path)
        grasp_index = {}  # Storing graspit grasp index if motion plan found
        gazebo_pose = (
            {}
        )  # Storing the pose of obj in gazebo (maybe changed after spawning due to  physics)
        for curr_obj in object_names:
            print(f"Planning for object: {curr_obj}")
            curr_success, g_idx = plan_all_grasps(
                group, curr_obj, grasp_dir, gripper, scene, try_grasp=grasp_flag
            )
            if curr_success != 1:
                # no need to continue with a scene which fails a motion plan for even single object
                break
            total_success += curr_success
            grasp_index[curr_obj] = g_idx
            gazebo_pose[curr_obj] = objs.get_state(curr_obj)[1]
        if total_success == len(object_names):
            print("Found motion plans for all objects!")
            scene_count += 1
            with open(
                os.path.join(output_dir, f"scene_id_{scene_count}.pk"), "wb"
            ) as f_handle:
                scene_info = {}
                # TODO: Query current pose of object in gazebo and also save it
                scene_info["gz_obj_poses"] = gazebo_pose
                scene_info["obj_poses"] = sample_scene
                scene_info["graspit_grasp_index"] = grasp_index
                scene_info["table_pos"] = table_position
                scene_info["grid_dims"] = (grid_r, grid_c)
                scene_info["stable_pose_idx"] = stable_pose_idx
                pickle.dump(scene_info, f_handle)

        # confirmation = input("Continue with next scene? y or n...")
        # cleanup for next scene!
        for obj in sample_scene:
            objs.delete_object(obj)
