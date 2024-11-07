#!/usr/bin/env python
import sys, os, pickle
import numpy as np
import copy
import argparse
import datetime

from scipy.io import loadmat
import rospy
import moveit_commander
import geometry_msgs.msg
import moveit_msgs.msg
import tf2_ros
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import PoseStamped

sys.path.append("./utils/")
from ros_utils import ros_pose_to_rt, rt_to_ros_qt, rt_to_ros_pose
from stow_or_tuck_arm import reset_arm_stow
from gripper import Gripper
from grasp_utils import (
    get_object_name,
    parse_grasps,
    sort_grasps,
    sort_and_filter_grasps,
    model_based_top_down_grasp,
    get_object_verts,
    get_standoff_wp_poses,
    lift_arm_cartesian,
    lift_arm_joint,
    lift_arm_pose,
    move_arm_to_dropoff,
    rotate_gripper,
    user_confirmation,
)
from utils_control import FollowTrajectoryClient, PointHeadClient
from utils_scene import load_scene, read_pickle_file, write_pickle_file
from image_listener import ImageListener
from moveit_msgs.msg import Constraints
from utils_log import get_custom_logger


def get_tf_pose(target_frame, base_frame=None, is_matrix=False):
    try:
        transform = tf_buffer.lookup_transform(
            base_frame, target_frame, rospy.Time.now(), rospy.Duration(1.0)
        ).transform
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
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ):
        RT_obj = None
    return RT_obj


def plan_grasp(
    group,
    scene,
    display_trajectory_publisher,
    RT_grasps_base,
    grasp_index,
    obj_name,
    RT_obj,
    models_path,
    successful_grasps,
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
    n = RT_grasps_base.shape[0]
    pose_standoff = get_standoff_wp_poses()
    flag_plan = False
    # for each grasp
    grasp_index = list(grasp_index)
    # print(f"grasp index {grasp_index}\n")
    # print(f"successful grasps {successful_grasps}\n")
    successful_grasps = set(grasp_index).intersection(successful_grasps)
    rest_of_grasps = set(grasp_index) - successful_grasps
    
    order_good = [grasp_index.index(g) for g in successful_grasps]
    order_rest = [grasp_index.index(g) for g in rest_of_grasps]
    iter_order = order_good + order_rest
    for i in iter_order:
        grasp_idx = grasp_index[i]
        if grasp_idx not in successful_grasps:
            continue
        RT_grasp = RT_grasps_base[i]

        print(f"{i}: Planning for {obj_name} with grasp_idx {grasp_idx}")
        # print(RT_grasp)
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
        plan = group.plan()
        trajectory = plan[1]
        if plan[0]:
            print("found a plan for grasp")
            flag_plan = True
            # rospy.sleep(1)
            # quat, trans = rt_to_ros_qt(standoff_grasp_global[0, :, :])
            # cube = PoseStamped()
            # cube.header.frame_id = robot.get_planning_frame()
            # cube.pose.position.x = trans[0]
            # cube.pose.position.y = trans[1]
            # cube.pose.position.z = trans[2]
            # cube_name = "marker_cube"
            # scene.add_box(cube_name, cube, (0.02, 0.02, 0.02))
            # cube_ok = input("Cube OK?")
            # scene.remove_world_object(cube_name)
            # Test for waypoints to final grasp pose
            scene.remove_world_object(obj_name)
            waypoints = []
            wpose = group.get_current_pose().pose
            for i in range(1, standoff_grasp_global.shape[0]):
                wpose = rt_to_ros_pose(wpose, standoff_grasp_global[i])
                waypoints.append(copy.deepcopy(wpose))
            print("qwerty")
            # print("sssss", group.cartesian_path())
            print("qwerty2")


            (plan_standoff, fraction) = group.compute_cartesian_path(
                                waypoints,   # waypoints to follow
                                0.01,        # eef_step
                                True     # True to avoid collisions
                                )
            # (plan_final, fraction) = group.compute_cartesian_path(
            #     waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            # )  # jump_threshold
            print(f"Gidx {grasp_idx} fraction: {fraction}")
            if fraction >= 0.9:
                print("Found FULL PLAN!")
                return RT_grasp, grasp_idx, trajectory, plan_standoff
            else:
                obj_mesh_path = os.path.join(
                    models_path, obj_name, "textured_simple.obj"
                )
                p = PoseStamped()
                p.pose = rt_to_ros_pose(p.pose, RT_obj)
                scene.add_mesh(obj_name, p, obj_mesh_path)
        else:
            print("no plan for grasp %d with index %d" % (i, grasp_idx))

    if not flag_plan:
        print("no plan found in plan_grasp()")
    return None, -1, None, None


def grasp_with_rt(
    gripper, group, scene, object_name, display_trajectory_publisher, RT_grasp
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
    pose_standoff = get_standoff_wp_poses()
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
    plan = group.plan()
    trajectory = plan[1]
    if not plan[0]:
        print("no plan found in grasp()")
        # sys.exit()
        return

    # visualize plan
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(trajectory)
    # Publish
    display_trajectory_publisher.publish(display_trajectory)

    input("execute?")
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()

    # remove the target from the planning scene for grasping
    scene.remove_world_object(object_name)

    waypoints = []
    wpose = group.get_current_pose().pose
    for i in range(1, standoff_grasp_global.shape[0]):
        wpose = rt_to_ros_pose(wpose, standoff_grasp_global[i])
        waypoints.append(copy.deepcopy(wpose))
    (plan_standoff, fraction) = group.compute_cartesian_path(
        waypoints, 0.01, True  # waypoints to follow  # eef_step
    )  # jump_threshold
    
    print(f"{object_name}: FRACTION: {fraction}")
    trajectory = plan_standoff

    # visualize plan
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(trajectory)
    # Publish
    display_trajectory_publisher.publish(display_trajectory)

    input("execute?")
    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()

    # close gripper
    print("close gripper")
    gripper.close()
    rospy.sleep(2)


def get_pose(object_name: str, pose_method: str):
    """
    Calls the suitable function depending on pose method
    
    Input:
    - object_name (str) : name of the object mode, e.g '003_cracker_box'
    - pose_method (str) : name of the model based pose method, e.g. 'poserbpf'
        - options : {'gazebo', 'posecnn', 'poserbpf'}
    
    Returns:
    - RT_object (4,4 np.ndarray) : 4x4 transform of the object in robot's base_link frame
    """
    if pose_method == "gazebo":
        return get_pose_gazebo(object_name)
    elif pose_method == "isaac":
        return get_pose_isaac(object_name)    
    elif pose_method == "posecnn":
        return get_pose_posecnn(object_name)
    elif pose_method == "poserbpf":
        return get_pose_poserbpf(object_name)
    else:
        print(f"[ERROR] incorrect pose method provided : {pose_method}. Will return None!")
        return None


def get_pose_posecnn(object_name: str):
    """
    Queries the PoseCNN topic for the given YCB `object_name` and returns
    a 4x4 transform for its pose
    """
    # Example: "posecnn/00_cracker_box_01_refined"
    # NOTE: The object name appears without the ycbid, "cracker_box" instead of "003_cracker_box"
    posecnn_topic_name = f"posecnn/00_{object_name[4:]}_01_refined"
    RT_obj = get_tf_pose(posecnn_topic_name, 'base_link')
    return RT_obj


def get_pose_gazebo(model_name, relative_entity_name=""):
    import roslib

    roslib.load_manifest("gazebo_msgs")
    from gazebo_msgs.srv import GetModelState

    def gms_client(model_name, relative_entity_name):
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            gms = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            resp1 = gms(model_name, relative_entity_name)
            return resp1
        except (rospy.ServiceException, e):
            print("Service call failed: %s" % e)

    res = gms_client(model_name, relative_entity_name)
    RT_obj = ros_pose_to_rt(res.pose)

    # query fetch base line
    res = gms_client(model_name="fetch", relative_entity_name="base_link")
    RT_base = ros_pose_to_rt(res.pose)

    # object pose in robot base
    RT = np.matmul(np.linalg.inv(RT_base), RT_obj)
    return RT


def get_pose_poserbpf(object_name: str):
    """
    Queries the PoseRBPF topic for the given YCB `object_name` and returns
    a 4x4 transform for its pose
    """
    # Example: "poserbpf/00_003_cracker_box_00"
    poserbpf_topic_name = f"poserbpf/00_{object_name}_00"  # f"00_{object_name}_raw"
    RT_obj = get_tf_pose(poserbpf_topic_name, "base_link")
    return RT_obj


def get_pose_isaac(object_name: str):
    """
    Queries the Isaac Sim object topic for the given YCB `object_name` and returns
    a 4x4 transform for its pose
    """
    # Example: "object_003_cracker_box_base_link"
    isaac_topic_name = f"object_{object_name}_base_link"
    RT_obj = get_tf_pose(isaac_topic_name, "base_link")
    return RT_obj


def get_gripper_rt(tf_buffer):
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
    return RT_gripper


def setup_moveit_scene(
    scene,
    table_height,
    object_names,
    models_path,
    pose_method,
    gt_metadata,
    gt_file_path,
    table_position=(0.8, 0, 0)
):
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
    p.pose.position.z = table_height - 0.5  # 0.5 = half length of moveit obstacle box
    scene.add_box("table", p, (1, 2, 1))
    # add a box for robot base
    p.pose.position.x = 0
    p.pose.position.y = 0
    p.pose.position.z = 0.18
    scene.add_box("base", p, (0.56, 0.56, 0.4))

    for obj_name in object_names:
        RT_obj = get_pose(obj_name, pose_method)
        gt_metadata["estimated_poses"][obj_name] = RT_obj
        if RT_obj is not None:
            p.pose = rt_to_ros_pose(p.pose, RT_obj)
            obj_mesh_path = os.path.join(models_path, obj_name, "textured_simple.obj")
            if not os.path.exists(obj_mesh_path):
                print(f"Given mesh path does not exist! {obj_mesh_path}")
            scene.add_mesh(obj_name, p, obj_mesh_path)
        else:
            print(f"Not adding {obj_name} to planning scene! (pose not found)")
    write_pickle_file(gt_metadata, gt_file_path)


def make_args():
    parser = argparse.ArgumentParser(
        description="Process the args like refine_pose, threshold, ycbid"
    )

    parser.add_argument(
        "-s",
        "--scene_idx",
        type=int,
        required=True,
        help="ID for the scene under evaluation",
    )

    parser.add_argument(
        "--pose_method",
        type=str,
        required=True,
        help='Specify the object pose estimation method: "gazebo", "poserbpf", "posecnn", "isaac"',
    )

    parser.add_argument(
        "--obj_order",
        type=str,
        required=True,
        help='Order to grasp object. Choose from {"nearest_first", "random"}',
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/benchmark/Datasets/benchmarking/",
        help="Path to parent of model dataset, grasp and scenes dir",
    )

    parser.add_argument(
        "--scene_dir",
        type=str,
        default="final_scenes",
        help="Path to data dir",
    )
    parser.add_argument(
        "-sg",
        "--sgrasp_file",
        default="sgrasps.pk",
        help="path to successful grasp pkl file",
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
    VALID_POSE_METHODS = {"gazebo", "poserbpf", "posecnn", "isaac"}

    args = make_args()
    table_height = args.table_height
    pose_method = args.pose_method
    scene_idx = args.scene_idx
    ordering = args.obj_order

    # Data dir to hold the logs and results for experiments
    exp_root_dir = os.path.join(
        os.path.abspath("./"), "data", "experiments", "bench_pose"
    )
    # hyper-params: create a dir name with this and timestamp
    # seg_method, scene_id, order
    curr_time = datetime.datetime.now()
    exp_time = "{:%y-%m-%d_T%H%M%S}".format(curr_time)
    exp_args = f"method-{pose_method}_scene-{scene_idx}_ord-{ordering}"
    exp_dir = os.path.join(exp_root_dir, exp_time + "_" + exp_args)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        print(f"directory created")

    logger = get_custom_logger(os.path.join(exp_dir, str(curr_time)+".log"))

    logger.inform(f"scene_id:{args.scene_idx}")
    logger.inform(f"pose_method:{args.pose_method}")
    logger.inform(f"ordering:{args.obj_order}")
    logger.inform(f"table_height:{args.table_height}")

    if ordering not in {"nearest_first", "random"}:
        logger.error("Incorrect option for object ordering. See help!")
        sys.exit(0)

    model_dir = os.path.join(args.data_dir, "models")
    grasp_dir = os.path.join(args.data_dir, "grasp_data", "refined_grasps")
    scene_dir = os.path.join(args.data_dir, args.scene_dir)
    grasp_order_f = os.path.join(scene_dir, args.sgrasp_file)
    experiment_data_file = os.path.join(exp_dir, "exp_data.pk")

    
    # Read the ordering over graspit grasp for all objects in scene
    success_grasp_info = read_pickle_file(grasp_order_f)
    # Read in metadata for correct object order to grasp
    meta_f = "meta-%06d.mat" % scene_idx
    meta = loadmat(os.path.join(scene_dir, "metadata", meta_f))
    objects_in_scene = [obj.strip() for obj in meta["object_names"]]
    # List of objects arranged in order to grasp
    # object_order = meta[ordering]
    object_order = meta[ordering][0].split(",")
    print(object_order)
    print(objects_in_scene)
    # assert set(object_order) == set(objects_in_scene)
    print(f"Grasping in following order: {object_order}")
    if pose_method not in VALID_POSE_METHODS:
        print(
            f"Incorrect Pose method specified: {args.pose_method}. Should be either: gazebo, isaac, poserbpf or posecnn"
        )
        exit(0)
    experiment_data = {}
    experiment_data["metadata"] = {"scene_id":args.scene_idx, "pose_method": args.pose_method, "ordering": args.obj_order, "table_height": args.table_height}
    experiment_data["estimated_poses"] = {objectname: None for objectname in object_order}

    gt_experiment_data = {}
    gt_experiment_data["metadata"] = {"scene_id":args.scene_idx, "pose_method": args.pose_method, "ordering": args.obj_order, "table_height": args.table_height}
    gt_experiment_data["estimated_poses"] = {objectname: None for objectname in object_order}

    import logging
    root_handlers = logging.root.handlers[:]
    # ----------------------------- ROSPY Stuff--------------------------------#
    # Create a node
    rospy.init_node("ramp_grasping")
    logging.root.handlers = root_handlers
    # from importlib import reload
    # import logging
    # reload(logging)
    logger.setLevel(200)
    # cv_bridge = CvBridge()
    tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    image_listener = ImageListener()

    # Setup clients
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    # Raise the torso using just a controller
    rospy.loginfo("Raising torso...")
    torso_action.move_to(
        [
            0.4,
        ]
    )
    rospy.loginfo("Raising torso...done")

    # look at table
    head_action = PointHeadClient()
    rospy.loginfo("Pointing head...")
    if head_action.success:
        head_action.look_at(0.45, 0, table_height, "base_link")
    else:
        head_action = FollowTrajectoryClient("head_controller", ["head_pan_joint", "head_tilt_joint"])
        head_action.move_to([0.009195, 0.908270])
    rospy.loginfo("Pointing head...done")

    # --------------------------- initialize moveit components ----------------#
    moveit_commander.roscpp_initialize(sys.argv)
    group = moveit_commander.MoveGroupCommander("arm")
    group.set_max_velocity_scaling_factor(1.0)
    # group.set_max_acceleration_scaling_factor(1.0)
    group_grp = moveit_commander.MoveGroupCommander("gripper")
    scene = moveit_commander.PlanningSceneInterface()
    scene.remove_world_object()
    robot = moveit_commander.RobotCommander()
    display_trajectory_publisher = rospy.Publisher(
        "/move_group/display_planned_path",
        moveit_msgs.msg.DisplayTrajectory,
        queue_size=20,
    )
    gripper = Gripper(group_grp)
    # ---------------------------- initialize moveit components ---------------#


    for obj_i, object_to_grasp in enumerate(object_order):
        grasp_num, trajectory_standoff, trajectory_final = None, None, None
        gt_experiment_data_file = os.path.join(exp_dir, f"gt_exp_data_{obj_i}.pk")
        gripper.open()
        setup_moveit_scene(
            scene=scene, table_height= table_height, object_names= objects_in_scene, models_path= model_dir, pose_method=pose_method, gt_file_path=gt_experiment_data_file, gt_metadata=gt_experiment_data
        )
        RT_obj = get_pose(object_to_grasp, pose_method)
        if RT_obj is None:
            print(f"{object_to_grasp} pose not found using {pose_method}\n")
            logger.error(f"{object_to_grasp} pose in scene {args.scene_idx} not found using {pose_method} pose estimation")
            logger.error(f"{object_to_grasp} not grasped at all!! ---------------------------------------------\n")
            continue
        # TODO: LOG the estimated grasp Pose to Log file
        logger.pose(f"estimated_{object_to_grasp}_pose: {RT_obj}")
        experiment_data["estimated_poses"][object_to_grasp] = RT_obj
        write_pickle_file(experiment_data, experiment_data_file)
        direct_topdown = False
        if not direct_topdown:
            RT_gripper = get_gripper_rt(tf_buffer)
            print("RT_gripper", RT_gripper)
            # Using Graspit generated grasp to test the Pose Detection (model based grasping)
            grasp_file = os.path.join(grasp_dir, f"fetch_gripper-{object_to_grasp}.json")
            RT_grasps = parse_grasps(grasp_file)
            RT_grasps_base, grasp_index, pruned_ratio = sort_and_filter_grasps(
                RT_obj, RT_gripper, RT_grasps, table_height
            )
            print(f"grasp indbex {grasp_index}, object to grasp {object_to_grasp}")
            # if (grasp_index is None ) and (object_to_grasp == "025_mug"):
            #     print(f"no grasp found valid for bowl.........")
            #     logger.error(f"no grasp found valid for bowl.........")
            #     continue
            if (grasp_index is None ):
                direct_topdown=True
            else:    
                successful_grasps = set(success_grasp_info[scene_idx][object_to_grasp])
                # grasp planning
                RT_grasp, grasp_num, trajectory_standoff, trajectory_final = plan_grasp(
                    group,
                    scene,
                    display_trajectory_publisher,
                    RT_grasps_base,
                    grasp_index,
                    object_to_grasp,
                    RT_obj,
                    model_dir,
                    successful_grasps,
                )
        # Exit code for plan_grasp(): Returning an exit code to catch it here so that we can still continue on to the next trial
        if direct_topdown or (grasp_num == -1 or (not trajectory_standoff) or (not trajectory_final)):
            print("No plans found for direct grasping, trying TOP-DOWN!")
            logger.warning("No plans found for direct grasping, trying TOP-DOWN!")
            mesh_p = os.path.join(model_dir, object_to_grasp, "textured_simple.obj")
            obj_pts = get_object_verts(mesh_p, pose=RT_obj)
            RT_grasp, g_width = model_based_top_down_grasp(obj_pts)
            print(f"Gripper Width: {g_width}")
            if object_to_grasp in {"052_extra_large_clamp", "025_mug"}:
                g_width = -1
            if g_width < (0.1 - 0.002):
                grasp_with_rt(
                    gripper,
                    group,
                    scene,
                    object_to_grasp,
                    display_trajectory_publisher,
                    RT_grasp,
                )
                logger.inform("TOP DOWN SUCCESSFUL")
            else:
                print("TOP DOWN FAILED!! Object too wide.")
                logger.error("TOP DOWN FAILED!! Object too wide.")
        elif trajectory_standoff and trajectory_final and grasp_num != -1:
            grasp_with_rt(
                gripper,
                group,
                scene,
                object_to_grasp,
                display_trajectory_publisher,
                RT_grasp,
            )
        logger.inform(f"{object_to_grasp} reached successfully")

        # ------------------------ LIFTING OBJECT --------------------------#
        if gripper.is_fully_closed() or gripper.is_fully_open():
            print("Gripper fully open/closed (after Grasping)....Not Lifting!")
            # TODO: LOG Grasping failure to log file with scene_id, object name, pose method, and order (all exp params)
            logger.failure_gripper(f"Gripper fully open/closed (after Grasping) scene--{args.scene_idx} object_name--{object_to_grasp} pose_method--{args.pose_method} ordering--{args.obj_order}")
        else:
            print("Trying to lift object")
            RT_gripper = get_gripper_rt(tf_buffer)
            print("RT_gripper before lifting", RT_gripper)
            lift_arm_cartesian(group, RT_gripper)
            
            # ----------------------- MOVING OBJECT ------------------------#
            if gripper.is_fully_closed() or gripper.is_fully_open():
                print("Gripper fully open/closed (after Lifting)....Not Moving!")
                # TODO: LOG Lifting failure to log file
                logger.failure_gripper("Gripper fully open/closed (after Lifting)....Not Moving!")

            else:
                logger.inform(f"{object_to_grasp} successfully lifted")
                print("Trying to move object")
                RT_gripper = get_gripper_rt(tf_buffer)
                rotate_gripper(group, RT_gripper)
                RT_gripper = get_gripper_rt(tf_buffer)
                print("RT_gripper after lift", RT_gripper)
                move_arm_to_dropoff(group, RT_gripper, x_final=0.78)
                if gripper.is_fully_closed() or gripper.is_fully_open():
                    print("Gripper fully open/closed (after Moving)....")
                    # TODO: LOG Moving failure to log file
                    logger.failure_dropoff("Gripper fully open/closed (after Moving).... ")
                logger.inform(f"{object_to_grasp} successfully droppedoff")
        
        # ------------------------ OPEN GRIPPER & STOW ---------------------#
        input("Open Gripper??")
        gripper.open()
        print("STOWING THE GRIPPER")
        reset_arm_stow(group)
        input("proceed next ?")
        input("proceed next ?")
        input("proceed next ?")
        
