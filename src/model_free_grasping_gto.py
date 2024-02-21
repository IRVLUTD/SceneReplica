#!/usr/bin/env python
import math
import moveit_msgs.msg
import sys
import os
import copy
import argparse
import datetime
import time
from scipy.io import loadmat
import imageio.v2 as imageio

import cv2
import numpy as np
import threading
from matplotlib import pyplot as plt
import munkres as munkres

import rospy
from geometry_msgs.msg import PoseStamped
import moveit_commander
import geometry_msgs.msg
from tf.transformations import euler_matrix, quaternion_matrix
import tf2_ros

sys.path.append("./utils/")
from ros_utils import ros_pose_to_rt, rt_to_ros_qt, rt_to_ros_pose
from gripper import Gripper
from utils_control import FollowTrajectoryClient, PointHeadClient, JointListener
from stow_or_tuck_arm import reset_arm_stow
from image_listener import (
    ImageListener,
    MsmSegListener,
    UoisSegListener,
    GraspPoseListener,
    ObjPointPublisher,
)
from utils_scene import write_pickle_file
from utils_log import get_custom_logger
from utils_segmentation import visualize_segmentation, process_label_image, compute_segments_assignment
from grasp_utils import (
    model_free_sort_and_filter_grasps,
    model_free_top_down_grasp,
    compute_oriented_bbox,
    get_standoff_wp_poses,
    lift_arm_joint,
    lift_arm_pose,
    lift_arm_cartesian,
    rotate_gripper,
    move_arm_to_dropoff,
    user_confirmation,
    convert_plan_to_trajectory_toppra,
)
from random import shuffle

# GTO planner
import _init_paths
from mesh_to_sdf.depth_point_cloud import DepthPointCloud
from optas.visualize import Visualizer
from gto.gto_models import GTORobotModel
from gto.gto_planner import GTOPlanner
from gto.ik_solver import IKSolver
from gto.utils import load_yaml, get_root_dir, visualize_plan


def compute_obstacle_for_object(bbox, camera_pose, label, xyz_image):
    scene_boxes = [] # stores markers
    for i in range(bbox.shape[0]):
        mask_id = bbox[i, -1]
        target_pts = get_target_pts(camera_pose, mask_id, label, xyz_image)
        print(target_pts.shape, mask_id)
        center, xlen, ylen, zlen, theta = compute_oriented_bbox(
            target_pts
        )
        if center is None:
            continue
        p = PoseStamped()
        p.header.frame_id = robot.get_planning_frame()
        p.pose.position.x = center[0]
        p.pose.position.y = center[1]
        p.pose.position.z = center[2]
        # rotate around Z axis by angle theta. axis: (0,0,1) angle: theta -->
        # quaternion is (cos t/2, 0, 0, sin t/2) in [wxyz] format
        p.pose.orientation.w = np.cos(theta / 2)
        p.pose.orientation.x = 0
        p.pose.orientation.y = 0
        p.pose.orientation.z = np.sin(theta / 2)
        marker_info = {}
        marker_info['marker_name'] = f"object_{i}"
        marker_info['pose'] = p
        marker_info['size'] = (xlen, ylen, zlen)
        marker_info['mask_id'] = mask_id
        scene_boxes.append(marker_info)
    # ------------------------------------------------- #
    return scene_boxes


def add_markers_to_scene(scene, scene_boxes):
    for marker_info in scene_boxes:
        marker_name = marker_info['marker_name']
        p = marker_info['pose']
        (xlen, ylen, zlen) = marker_info['size']
        scene.add_box(marker_name, p, (xlen, ylen, zlen))


def remove_markers_from_scene(scene, scene_boxes):
    for marker_info in scene_boxes:
        marker_name = marker_info['marker_name']
        scene.remove_world_object(marker_name)


def plan_grasp(robot,
                group,
                scene,
                RT_grasps_base,
                bbox,
                scene_boxes):
    n = RT_grasps_base.shape[0]
    pose_standoff = get_standoff_wp_poses(standoff_dist=0.15, extra_off=0.015)
    flag_plan = False

    # Here there is no question of "good", pre-selected grasps, we simply iterate
    _grasps = list(range(n))
    shuffle(_grasps)

    if n > 40:
        _grasps = _grasps[:40]
    for _ixx, i in enumerate(_grasps):
        RT_grasp = RT_grasps_base[i]
        print(f"{_ixx}: Planning....")
        # Add Markers to Scene
        add_markers_to_scene(scene, scene_boxes)
        standoff_grasp_global = np.matmul(RT_grasp, pose_standoff)
        # Calling `stop()` ensures that there is no residual movement
        group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        group.clear_pose_targets()
        # plan to the standoff
        quat, trans = rt_to_ros_qt(standoff_grasp_global[0, :, :])  # xyzw for quat
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = quat[3]
        pose_goal.orientation.x = quat[0]
        pose_goal.orientation.y = quat[1]
        pose_goal.orientation.z = quat[2]
        pose_goal.position.x = trans[0]
        pose_goal.position.y = trans[1]
        pose_goal.position.z = trans[2]
        group.set_pose_target(pose_goal)
        
        # -------- adding a marker -------
        cube = PoseStamped()
        cube.header.frame_id = robot.get_planning_frame()
        cube.pose.position.x = trans[0]
        cube.pose.position.y = trans[1]
        cube.pose.position.z = trans[2]
        # scene.add_box("standoff_cube", cube, (0.05, 0.05, 0.05))
        # _tmp = input("Showing Cube for grasp standoff waypoint!")
        cube.pose.orientation.w = quat[3]
        cube.pose.orientation.x = quat[0]
        cube.pose.orientation.y = quat[1]
        cube.pose.orientation.z = quat[2]        
        # scene.add_mesh("standoff", cube, 'fetch_real_world.stl')
        # _tmp = input("Showing gripper mesh for grasp standoff waypoint!")
        # scene.remove_world_object("standoff")

        plan = group.plan()
        trajectory = plan[1]
        if plan[0]:
            print("found a plan for grasp")
            flag_plan = True
            # remove obstacles
            remove_markers_from_scene(scene, scene_boxes)
            waypoints = []
            wpose = group.get_current_pose().pose
            for ixx in range(1, standoff_grasp_global.shape[0]):
                wpose = rt_to_ros_pose(wpose, standoff_grasp_global[ixx])
                waypoints.append(copy.deepcopy(wpose))
            (plan_final, fraction) = group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )  # jump_threshold
            print(f"Iteration: {_ixx} | Gidx {i} fraction: {fraction}")
            if fraction >= 0.9:
                print("Found Complete PLAN!")
                return RT_grasp, i, trajectory, plan_final
            # else go to next iteration
        else:
            # Show a cube with marker!
            print(f"no plan found to standoff for index {i}")

    if not flag_plan:
        print("Not SINGLE complete plan found in plan_grasp()")
    return None, -1, None, None


def grasp(
    robot,
    gripper,
    group,
    arm_action,
    scene,
    bbox,
    scene_boxes,
    RT_grasp,
    gripper_width,
    object_name,
    display_trajectory_publisher,
    confirm=True,
    slow_execution=False,
):
    """
    Method to perform a top down grasp of an object
    """
    # Compute standoff pose
    pose_standoff = get_standoff_wp_poses(standoff_dist=0.15, extra_off=0.015)
    standoff_grasp_global = np.matmul(RT_grasp, pose_standoff)
    # Calling `stop()` ensures that there is no residual movement
    group.stop()
    # It is always good to clear your targets after planning with poses.
    # Note: there is no equivalent function for clear_joint_value_targets()
    group.clear_pose_targets()


        # NOTE: We dont add here but rather save the info, so that we can add/remove at will
        # scene.add_box(_marker_name, p, (xlen, ylen, zlen))
    # -------------------------------------------------------------------- # 
    # Add moveit obstacles to scene
    add_markers_to_scene(scene, scene_boxes)

    quat, trans = rt_to_ros_qt(standoff_grasp_global[0, :, :])  # xyzw for quat
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = quat[3]
    pose_goal.orientation.x = quat[0]
    pose_goal.orientation.y = quat[1]
    pose_goal.orientation.z = quat[2]
    pose_goal.position.x = trans[0]
    pose_goal.position.y = trans[1]
    pose_goal.position.z = trans[2]
    group.set_pose_target(pose_goal)

    plan = group.plan()
    trajectory = plan[1]
    if not plan[0]:
        print("no plan found to standoff")
        # -------- adding a marker -------
        cube = PoseStamped()
        cube.header.frame_id = robot.get_planning_frame()
        cube.pose.position.x = trans[0]
        cube.pose.position.y = trans[1]
        cube.pose.position.z = trans[2]
        cube.pose.orientation.w = quat[3]
        cube.pose.orientation.x = quat[0]
        cube.pose.orientation.y = quat[1]
        cube.pose.orientation.z = quat[2]
        # scene.add_box("standoff", cube, (0.05, 0.05, 0.05))
        # scene.add_mesh("standoff", cube, 'fetch_real_world.stl')
        # _tmp = input("Showing gripper mesh for grasp standoff waypoint!")
        # scene.remove_world_object("standoff")
        return False

    if confirm:
        if user_confirmation("Move to standoff pose"):
            pass
        else:
            return False
     # visualize plan
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(trajectory)
    # Publish
    display_trajectory_publisher.publish(display_trajectory)

    # ------------------------Now Plan to Final Grasp ------------------------ #
    input("execute grasp?")
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()
    print("Gripper width", gripper_width)
    # gripper.open_position(gripper_width)

    # remove the obstacles for objects from the scene
    remove_markers_from_scene(scene, scene_boxes)

    # plan to final grasping pose slowly
    group.set_max_velocity_scaling_factor(0.1)
    if object_name is not None:
        scene.remove_world_object(object_name)

    waypoints = []
    wpose = group.get_current_pose().pose
    for i in range(1, standoff_grasp_global.shape[0]):
        wpose = rt_to_ros_pose(wpose, standoff_grasp_global[i])
        waypoints.append(copy.deepcopy(wpose))
    (plan_standoff, fraction) = group.compute_cartesian_path(
        waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
    )  # jump_threshold
    print(f"Fraction of waypoints success: {fraction}")
    if fraction < 0.9:
        print("[ERROR]: Cannot reach all specified waypoints, aborting!")
        return False

    trajectory = plan_standoff
    # visualize plan
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(trajectory)
    # Publish
    display_trajectory_publisher.publish(display_trajectory)
    input("execute?")
    input("execute?")
    if confirm:
        input("execute?")
        if user_confirmation("Move to grasping pose"):
            pass
        else:
            return False

    # ---------------------- Execute Final Trajectory ------------------------ #
    if slow_execution:
        num = len(trajectory.joint_trajectory.points)
        for i in range(num):
            positions = trajectory.joint_trajectory.points[i].positions
            arm_action.move_to(positions, 0.5)
            rospy.sleep(0.1)
    else:
        group.execute(trajectory, wait=True)
    group.set_max_velocity_scaling_factor(1.0)
    group.stop()
    group.clear_pose_targets()

    # Close Gripper
    print("Closing Gripper")
    gripper.close(max_effort=100)
    rospy.sleep(1)

    return True


def execute_trajectory(
    group, display_trajectory_publisher, trajectory
):
    """
    A method the included the actions of pushing and sweeping according to direction.
    It first set its arm to the left/right of the given location, then sweeps to the left/right of the cube to achieve
    the pushing motion
    :param group: the moveit group of joints
    :param display_trajectory_publisher: for visualization of the planned trajectory
    :param RT_grasp: gripper pose for grasping
    :return:
    """

    # visualize plan
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    robot_trajectory = moveit_msgs.msg.RobotTrajectory()
    robot_trajectory.joint_trajectory = trajectory
    display_trajectory.trajectory.append(robot_trajectory)
    # Publish
    display_trajectory_publisher.publish(display_trajectory)

    input("execute?")
    # for point in trajectory.points:
    #     arm_action.move_to(point.positions, duration=dt, velocities=point.velocities)
    # arm_action.follow_traj(trajectory)
    group.stop()
    group.clear_pose_targets()
    group.execute(robot_trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()


def process_boxes(bbox, order="random"):
    """
    From a set of bboxes, pick an object (bbox) to grasp, either nearest bbox
    first or a random bbox

    Input:
    - bbox: (N,8) array with bboxes for N segmented objects
    - order (str): 'random' or 'nearest_first' way to select grasping object

    Returns:
    - bbox_grasp (8, ) array with bbox info for object to grasp
    - idx: index into the provided bbox_list
    """
    if (bbox is None) or (bbox.shape[0] < 1):
        print("no graspable object detected")
        return None, None

    print(f"Using {order} order to choose object for grasping")
    confidence = bbox[:, 6]
    if order == "random":  # randomly select an object to grasp
        idx = np.random.permutation(len(confidence))[0]
    else:
        xdist_to_base = bbox[:, 0]
        idx = np.argmax(xdist_to_base)
    bbox_grasp = bbox[idx, :]
    print(f"grasp object with score {confidence[idx]}")
    return bbox_grasp, idx


def associate_bbox(b1, b2):
    """
    data association between two sets of bboxes
    """
    n1 = b1.shape[0]
    n2 = b2.shape[0]
    # compute distance between two sets of bboxes
    D = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        c1 = b1[i, :3]
        for j in range(n2):
            c2 = b2[j, :3]
            D[i, j] = np.linalg.norm(c1 - c2)

    ### Compute the Hungarian assignment ###
    m = munkres.Munkres()
    assignments = m.compute(D)  # list of (y,x) indices into F (these are the matchings)
    return assignments


def add_table_base_to_planning(scene, table_height, table_position=(0.8, 0, 0)):
    rospy.loginfo("adding table object into planning scene")
    print("adding table object into planning scene")

    # # sleep before adding objects
    # # dimension of each default(1,1,1) box is 1x1x1m
    # -------- planning scene set-up -------
    rospy.sleep(1)
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()
    p.pose.position.x = table_position[0]
    p.pose.position.y = 0
    p.pose.position.z = table_height - 0.51
    scene.add_box("table", p, (1.3, 3, 1))

    rospy.sleep(1)
    p = PoseStamped()
    p.header.frame_id = robot.get_planning_frame()
    p.pose.position.x = 0
    p.pose.position.y = 0
    p.pose.position.z = -0.13
    scene.add_box("base", p, (1.3, 1, 1))


def get_target_pts(camera_pose, mask_id, label, xyz_image, frame='base'):
    # target mask
    mask = (label == mask_id).astype(np.uint8)
    # erode mask
    kernel = np.ones((3, 3), np.uint8)
    mask2 = cv2.erode(mask, kernel)
    # process all the points
    depth = xyz_image[:, :, 2]
    # process points for the target
    index = (mask2 > 0) & (depth > 0)
    points = xyz_image[index, :]
    if frame == "camera":
        print(points.shape)
        points_tf = np.reshape(points, (-1, 3))
        print(points_tf.shape)
        return points_tf
    else:
        points_base = np.matmul(camera_pose[:3, :3], points.T) + camera_pose[:3, 3].reshape(
            (3, 1)
        )
        points_base = points_base.T
        return points_base


def get_scene_pc(xyz_image):
    _tf_img = xyz_image.copy()
    depth = _tf_img[:, :, 2]
    index = (depth > 0) & (depth < 1.8)
    _tf_img = _tf_img[index, :]
    points = np.reshape(_tf_img, (-1, 3))
    return points


def get_gripper_rt(tf_buffer):
    try:
        transform = tf_buffer.lookup_transform(
            "base_link", "wrist_roll_link", rospy.Time.now(), rospy.Duration(1.0)
        ).transform
        quat = [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
        RT_gripper = quaternion_matrix(quat)
        RT_gripper[:3, 3] = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z]
        )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        RT_gripper = None
    return RT_gripper


def make_args():
    parser = argparse.ArgumentParser(
        description="Process the args like scene, threshold, ycbid"
    )

    parser.add_argument(
        "--seg_method",
        type=str,
        required=True,
        help='Specify segmentation method: {"msmformer", "uois"}',
    )

    parser.add_argument(
        "--grasp_method",
        type=str,
        required=True,
        help='Specify 6dof grasping method: {"graspnet", "contact_gnet"}',
    )

    parser.add_argument(
        "--safe_mode",
        action="store_true",
        help="Whether to ask user for confirmation before important steps and VIZ things",
    )

    parser.add_argument(
        "--table_height",
        type=float,
        default=0.74,
        help="table height based on which the object point cloud will be filtered (along Z-dim",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ----------------------- Experiment Args  ------------------------------- #
    VALID_SEG_METHODS = {"msmformer", "uois"}
    VALID_GRASP_METHODS = {"graspnet", "contact_gnet"}
    
    CLASSESS_ALL = ('003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
               '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', \
               '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '037_scissors', '040_large_marker', \
               '052_extra_large_clamp')
               
    CLASS_COLORS_ALL = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                    (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                    (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0)]

    args = make_args()
    table_height = args.table_height
    seg_method = args.seg_method
    grasp_method = args.grasp_method
    user_confirm = args.safe_mode

    # Data dir to hold the logs and results for experiments
    exp_root_dir = os.path.join(
        os.path.abspath("./"), "data", "experiments", "bench_6dof_seg_cg"
    )
    # hyper-params: create a dir name with this and timestamp
    # seg_method, scene_id, order
    curr_time = datetime.datetime.now()
    exp_time = "{:%y-%m-%d_T%H%M%S}".format(curr_time)
    exp_args = (
        f"grasp-{grasp_method}_seg-{seg_method}_model_free"
    )
    exp_dir = os.path.join(exp_root_dir, exp_time + "_" + exp_args)

    if seg_method not in VALID_SEG_METHODS:
        print(f"Incorrect seg method {args.seg_method}! See --help for details.")
        sys.exit(0)
    if grasp_method not in VALID_GRASP_METHODS:
        print(f"Incorrect grasp method: {grasp_method}! See --help for details.")
        sys.exit(0)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    logger = get_custom_logger(os.path.join(exp_dir, str(curr_time) + ".log"))
    logger.inform(f"grasp_method:{grasp_method}")
    logger.inform(f"seg_method:{seg_method}")
    logger.inform(f"table_height:{table_height}")

    # LOAD DATA FOR SCENES
    experiment_data_file = os.path.join(exp_dir, "exp_data.pk")
    experiment_data_file = os.path.join(exp_dir, "exp_data.pk")
    experiment_data = {}
    experiment_data["metadata"] = {
        "grasp_method": grasp_method,
        "seg_method": seg_method,
        "table_height": table_height,
    }

    import logging

    root_handlers = logging.root.handlers[:]
    # ----------------------- ROSPY Stuff ------------------------------------ #
    rospy.init_node("Seg6DofGrasping")  # create ros node
    logging.root.handlers = root_handlers
    logger.setLevel(200)
    tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
    rospy.sleep(2)
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(2)
    # Setup clients
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    head_action = PointHeadClient()
    rospy.loginfo("Raising torso...")
    torso_action.move_to(
        [
            0.4,
        ]
    )  # Raise the torso using just a controller
    head_action.look_at(0.45, 0, table_height + 0.58, "base_link")  # Look at fixed loc

    # image listener
    image_listener = ImageListener()

    # ---------------- Initialize moveit components -------------------------- #
    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander("arm")
    group.set_max_velocity_scaling_factor(0.5)
    group_grp = moveit_commander.MoveGroupCommander("gripper")  # Gripper's group
    scene = moveit_commander.PlanningSceneInterface()
    scene.remove_world_object()
    robot = moveit_commander.RobotCommander()
    arm_action = FollowTrajectoryClient("arm_controller", group.get_joints())
    # Setup the MoveIt Planning Scene with table and base
    add_table_base_to_planning(scene, table_height)

    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20,
    )    

    # ---------------- Initialize GTO Planner -------------------------- #
    # load config file
    robot_name = 'fetch'
    root_dir = get_root_dir()
    config_file = os.path.join(root_dir, 'data', 'configs', f'{robot_name}.yaml')
    if not os.path.exists(config_file):
        print(f'robot {robot_name} not supported', config_file)
        sys.exit(1) 
    cfg = load_yaml(config_file)['robot_cfg']
    print(cfg)    
    
    # load robot model
    robot_model_dir = os.path.join(root_dir, 'data', 'robots', cfg['robot_name'])
    urdf_filename = os.path.join(root_dir, cfg['urdf_robot_path']) 
    # define the standoff pose for collision checking
    offset = -0.01
    base_position = [0, 0, 0]

    gto_robot = GTORobotModel(robot_model_dir,
                          urdf_filename=urdf_filename, 
                          time_derivs=[0, 1],  # i.e. joint position/velocity trajectory
                          param_joints=cfg['param_joints'],
                          collision_link_names=cfg['collision_link_names'])

    # load robot gripper model
    urdf_filename = os.path.join(robot_model_dir, f"{robot_name}_gripper.urdf")
    gripper_model = GTORobotModel(robot_model_dir, urdf_filename=urdf_filename)    

    # Initialize planner
    print('Initialize planner')
    standoff_offset = -10
    planner = GTOPlanner(gto_robot, cfg['link_ee'], cfg['link_gripper'])
    ik_solver = IKSolver(gto_robot, cfg['link_ee'], cfg['link_gripper'], collision_avoidance=False)
    

    # --------------- Init Listner and Setup Robot and Gripper --------------- #
    # image listener
    if seg_method == "msmformer":
        listener = MsmSegListener(data_dir=exp_dir)
    elif seg_method == "uois":
        listener = UoisSegListener(data_dir=exp_dir)
    point_publisher = ObjPointPublisher(data_dir=exp_dir)
    grasp_listener = GraspPoseListener(data_dir=exp_dir)

    joint_listener = JointListener()
    gripper = Gripper(group_grp)
    gripper.open()

    # -------------------- Main Segmentation Loop ---------------------------- #
    try:           
        step = 0
        success = True
        bbox_prev = None

        while True:

            rospy.loginfo("Removing moveit objects from previous iteration")
            for name in scene.get_known_object_names():
                # print(name, name == "table")
                if name != "table" and name != "base":
                    scene.remove_world_object(name)
                    rospy.sleep(0.001)
            # -----------------------------------------------------------------#
            # Get object segementations and bbox info

            while True:
                im = listener.im
                label = listener.label
                xyz_image = listener.xyz_image
                camera_pose = listener.camera_pose
                bbox = listener.bbox
                if bbox is None:
                    logger.error(f"Step: {step} | No object segmented, trying next loop iteration!\n")
                    rospy.loginfo("No object segmented")
                    continue
                # filter objects
                index = bbox[:, 0] < 1.5
                bbox = bbox[index, :]
                if bbox.shape[0] < 1:
                    rospy.loginfo("Less than one object segmented")
                    rospy.sleep(2.0)
                    continue
                # if (bbox.shape[0] + step ) != 5:
                #     _go_to_next = True
                print(f"{bbox.shape[0]} objects segmented")
                listener.save_data(step)
                break

            # use the first object as target
            bbox_grasp = bbox[0]
            target_maskid_label = bbox[0, -1]
            print(f"Step: {step} | maskid_label_Pred: {target_maskid_label}\n")
            print(f"************************************************")
            
            # ------------------------ GRASP PLANNING --------------------------#
            mask_id = bbox_grasp[-1]
            
            pc_scene_cam = get_scene_pc(xyz_image) # scene's pc (N, 3) in camera frame!
            np.save("/home/ninad/depth_pc_cam.npy", pc_scene_cam)
            rospy.sleep(3)
            obj_pts_base = get_target_pts(camera_pose, mask_id, label, xyz_image)
            obj_pts_cam = get_target_pts(camera_pose, mask_id, label, xyz_image, frame="camera")
            # Publish the object points to 6dof grasp planning node
            print("Obtained target object points, publishing....")
            print(f"PC_ALL_SCENE shape: {pc_scene_cam.shape}")
            point_publisher.run(obj_pts_cam)
            point_publisher.run(obj_pts_cam)
            rospy.sleep(5)
            point_publisher.save_data(obj_pts_cam, step, pc_all=pc_scene_cam)
            rospy.sleep(1)
            # 6dof node will listen to points, sample grasps and publish a PoseArray
            print("Waiting for grasp pose array message...")
            rospy.sleep(5)
            while True:
                RT_grasps = grasp_listener.grasp_poses # in camera frame
                if RT_grasps is None:
                    # rospy.loginfo("Waiting for grasp pose info")
                    continue
                grasp_listener.save_data(step)
                break
            if len(RT_grasps) == 0:
                print("No Grasp Poses returend from Grasp Sampling algorithm")
                logger.error(f"Step: {step} | No Grasp Poses returend from Grasp Sampling algorithm")

            RT_gripper = get_gripper_rt(tf_buffer)
            RT_grasps_base, grasp_index, pruned_ratio = model_free_sort_and_filter_grasps(
                RT_grasps, table_height, RT_cam=camera_pose
            )
            # ----------- Moveit Markers Setup ---------------- #
            scene_boxes = compute_obstacle_for_object(bbox, camera_pose, label, xyz_image)
            # ----------- Moveit Markers Setup ---------------- #
            grasp_log = {
                "step": step,
                "RT_grasps_cam": RT_grasps,
                "filter_RT_grasps_base": RT_grasps_base,
                "pruned_ratio": pruned_ratio,
                "mask_id_grasp": mask_id,
                "xyz_image": xyz_image,
                "camera_pose": camera_pose,
                "label": label,
                "bbox": bbox,
                "scene_boxes": scene_boxes
            }
            experiment_data[f"step_{step}_log"] = grasp_log
            write_pickle_file(experiment_data, experiment_data_file)

            # NOTE: Implemented code similar to model based grasping
            # render image and compute sdf cost field
            im_color, depth_image, xyz_image, xyz_base, cam_pose, intrinsic_matrix = image_listener.get_data()
            depth_image[np.isnan(depth_image)] = np.inf
            depth_pc = DepthPointCloud(depth_image, intrinsic_matrix, cam_pose)

            # compute sdf cost of all points
            gto_robot.setup_points_field(depth_pc.points)           
            world_points = gto_robot.workspace_points
            sdf_cost_all = depth_pc.get_sdf_cost(world_points)

            # compute sdf cost obstacle
            target_mask = (label == target_maskid_label)
            depth_obstacle = depth_image.copy()
            depth_obstacle[target_mask] = 2.0
            depth_pc_obstacle = DepthPointCloud(depth_obstacle, intrinsic_matrix, cam_pose, target_mask)
            sdf_cost_obstacle = depth_pc_obstacle.get_sdf_cost(world_points)

            if user_confirm:
                # show result
                fig = plt.figure()
                ax = fig.add_subplot(2, 2, 1)
                plt.imshow(depth_image)
                ax.set_title('depth image')
                ax = fig.add_subplot(2, 2, 2)
                plt.imshow(target_mask)
                ax.set_title('mask image')
                ax = fig.add_subplot(2, 2, 3)
                plt.imshow(depth_obstacle)
                ax.set_title('depth obstacle')               
                ax = fig.add_subplot(2, 2, 4)
                plt.imshow(im[:, :, (2, 1, 0)])
                plt.show()

            # motion planning
            direct_topdown = False
            RT_grasp = None # Set the final grasp here
            if grasp_index is None:
                logger.error(f"Step: {step} | No Grasps from Planning...Going into Top Down")
                direct_topdown = True
            else:
                # Motion planning for sampled grasps!
                print('start checking collision of grasps')
                start = time.time()
                n = RT_grasps_base.shape[0]
                in_collision = np.zeros((n, ), dtype=np.int32)
                for i in range(n):
                    RT = RT_grasps_base[i]

                    # check if the grasp is in collision
                    RT_off = RT @ gto_robot.get_standoff_pose(offset, cfg['axis_standoff'])
                    gripper_points, normals = gripper_model.compute_fk_surface_points(cfg['gripper_open_offsets'], tf_base=RT_off)
                    sdf = depth_pc_obstacle.get_sdf(gripper_points)

                    ratio = np.sum(sdf < 0) / len(sdf)
                    print(f'grasp {i}, collision ratio {ratio}')
                    if ratio > 0.01:
                        in_collision[i] = 1
                
                RT_grasps_base = RT_grasps_base[in_collision == 0]
                checking_time = time.time() - start
                print('Checking grasp collision time', checking_time)
                print('Among %d grasps, %d in collision, %d collision-free' % (n, np.sum(in_collision), RT_grasps_base.shape[0]))
                if RT_grasps_base.shape[0] == 0:
                    direct_topdown=True
                else:
                    # test IK for remaining grasps
                    print('start IK')
                    start = time.time()
                    ik_solver.setup_optimization()
                    n = RT_grasps_base.shape[0]
                    found_ik = np.zeros((n, ), dtype=np.int32)

                    # get current joint
                    joint_name = joint_listener.joint_name
                    joint_position = joint_listener.joint_position
                    q0 = np.zeros((gto_robot.ndof, 1))
                    for i in range(gto_robot.ndof):
                        name = gto_robot.actuated_joint_names[i]
                        if name in joint_name:
                            q0[i] = joint_position[joint_name.index(name)]
                    # set gripper joint open
                    q0[cfg['finger_index'], 0] = cfg['gripper_open_offsets']

                    q_solutions = np.zeros((gto_robot.ndof, n), dtype=np.float32)
                    for i in range(n):
                        RT = RT_grasps_base[i].copy()
                        q_solution, err_pos, err_rot, cost_collision = ik_solver.solve_ik(q0, RT, sdf_cost_obstacle, base_position)
                        q_solutions[:, i] = q_solution
                        if err_pos < 0.01 and err_rot < 5 and cost_collision < 5:
                            found_ik[i] = 1
                    RT_grasps_base = RT_grasps_base[found_ik == 1]
                    q_solutions = q_solutions[:, found_ik == 1]
                    ik_time = time.time() - start
                    print('IK time', ik_time)
                    print('Among %d grasps, %d found IK' % (n, np.sum(found_ik)))
                    print('IK solutions with shape', q_solutions.shape)
                    if RT_grasps_base.shape[0] == 0:
                        direct_topdown=True
                    else:
                        # plan to a grasp set
                        qc = q0.flatten()
                        print('start planning')
                        start = time.time()
                        plan, dQ, cost = planner.plan_goalset(qc, RT_grasps_base, sdf_cost_all, sdf_cost_obstacle, 
                                                                base_position, q_solutions, use_standoff=True, axis_standoff=cfg['axis_standoff'])
                        planning_time = time.time() - start
                        print('plannnig time', planning_time, 'cost', cost)

                        # check if the robot plan is in collision
                        in_collision = False
                        plan = np.array(plan)
                        for i in range(plan.shape[1]):
                            q = plan[:, i]
                            points_base, _ = gto_robot.compute_fk_surface_points(q)
                            sdf = depth_pc_obstacle.get_sdf(points_base)
                            # at least 10 body points in collision
                            num = np.sum(sdf < 0)
                            if num > 0:
                                print('number of points in collision:', np.sum(sdf < 0))
                            if num > 30:
                                print('****************************** plan in collision ***************************')
                                print('number of points in collision:', np.sum(sdf < 0))
                                print('****************************** plan in collision ***************************')
                                in_collision = True
                                break                        

                        # convert to trajectory
                        if in_collision:
                            trajectory = None
                        else:
                            plan_ros = plan[gto_robot.optimized_joint_indexes, :]
                            dQ = dQ[gto_robot.optimized_joint_indexes, :]
                            # trajectory = convert_plan_to_trajectory(gto_robot.optimized_joint_names, plan_ros, dQ, planner.dt)
                            trajectory = convert_plan_to_trajectory_toppra(gto_robot, gto_robot.optimized_joint_names, plan_ros)
                        
                        # visualize_plan(gto_robot, gripper_model, base_position, plan, depth_pc, depth_pc_obstacle, RT_grasps_base)

                gripper_width = 0.05 # Set a dummy value for 6Dof grasp
            
            if trajectory:
                execute_trajectory(
                    group,
                    display_trajectory_publisher,
                    trajectory,
                )

                # close gripper
                print("close gripper")
                gripper.close()
                rospy.sleep(3)

                # get current joint
                joint_name = joint_listener.joint_name
                joint_position = joint_listener.joint_position
                q0 = np.zeros((gto_robot.ndof, 1))
                for i in range(gto_robot.ndof):
                    name = gto_robot.actuated_joint_names[i]
                    if name in joint_name:
                        q0[i] = joint_position[joint_name.index(name)]
                # set gripper joint close
                q0[cfg['finger_index'], 0] = 0

                # move to standoff
                plan_standoff = plan[:, np.arange(standoff_offset - 20, -1)]
                plan_reverse = plan_standoff[:, ::-1]
                plan_reverse[cfg['finger_index'], :] = 0
                plan_all = np.hstack((q0, plan_reverse))
                plan_ros = plan_all[gto_robot.optimized_joint_indexes, :]
                trajectory = convert_plan_to_trajectory_toppra(gto_robot, gto_robot.optimized_joint_names, plan_ros)

                execute_trajectory(
                    group,
                    display_trajectory_publisher,
                    trajectory,
                )

            ####### remove planning scene objects for lifting
            scene.remove_world_object()
            rospy.sleep(1) # add a delay before querying for gripper open/close status

            # ------------------------ OPEN GRIPPER & STOW ---------------------#
            input("Open Gripper??")
            gripper.open()
            rospy.sleep(1)
            reset_arm_stow(group)
            # Update Step (iteration number)
            step += 1
            if step >= 5:
                print(f"MAX 5 steps allowed!!!!!")
                # sys.exit(0)

            # ------------------------ NEXT STAGE CONFIRMATION ---------------------#
            user_confirm_message = (
                "PROCEED TO NEXT STEP? Press 'y' to go or 'n' to close program: "
            )
            _next_step = input(user_confirm_message) # only proceed with a 'y' or 'n' input
            while _next_step != "y" and _next_step != "n":
                _next_step = input(user_confirm_message)
            if _next_step == "y":
                continue
            elif _next_step == "n":
                print("Shutting down")
                rospy.loginfo("Removing all moveit objects from runs")
                scene.remove_world_object()
                # Calling `stop()` ensures that there is no residual movement
                group.stop()
                # It is always good to clear your targets after planning with poses.
                # Note: there is no equivalent function for clear_joint_value_targets()
                group.clear_pose_targets()
                rospy.signal_shutdown("motion_planning_error")
                break
            # ---------------------------------------------------------------------#
    except KeyboardInterrupt:
        print("Shutting down")
        rospy.loginfo("Removing all moveit objects from runs")
        scene.remove_world_object()
        # Calling `stop()` ensures that there is no residual movement
        group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        group.clear_pose_targets()
        rospy.signal_shutdown("motion_planning_error")