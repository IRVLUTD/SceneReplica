import os
import argparse
import sys
from scipy.io import loadmat
import rospy
import numpy as np

# isaac
import carb
from omni.isaac.kit import SimulationApp

FETCH_STAGE_PATH = "/Fetch"
TABLE_STAGE_PATH = "/Table"
CONFIG = {"renderer": "RayTracedLighting", "headless": False}

# set up isaac environment
simulation_app = SimulationApp(CONFIG)

# Example ROS bridge sample demonstrating the manual loading of stages
# and creation of ROS components
import omni.graph.core as og
import usdrt.Sdf
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils import extensions, stage, viewports
from omni.isaac.core.prims.rigid_prim import RigidPrim, RigidPrimView    
from omni.isaac.core.prims.geometry_prim import GeometryPrim

# SceneReplica
sys.path.append("./utils/")
from utils.utils_scene import load_scene
from utils_control import FollowTrajectoryClient
from ros_utils import convert_standard_to_rosqt


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/home/yuxiang/Projects/SceneReplica/data",
        help="Path to data dir",
    )
    parser.add_argument(
        "-s",
        "--scene_dir",
        type=str,
        default="final_scenes",
        help="Path to data dir",
    )
    args = parser.parse_args()
    return args


def default_pose(robot):
    # set robot pose
    print(robot.dof_names)
    # ['l_wheel_joint', 'r_wheel_joint', 'torso_lift_joint', 'bellows_joint', 'head_pan_joint', 
    # 'shoulder_pan_joint', 'head_tilt_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 
    # 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 
    # 'l_gripper_finger_joint', 'r_gripper_finger_joint']
    joint_command = robot.get_joint_positions()

    # arm_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
    #              "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
    # arm_joint_positions  = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]

    # raise torso
    joint_command[2] = 0.4
    # move head
    joint_command[4] = 0.009195
    joint_command[6] = 0.908270
    # move arm
    index = [5, 7, 8, 9, 10, 11, 12]
    joint_command[index] = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
    return joint_command    


def main(args):
    data_dir = args.data_dir
    scene_dir = args.scene_dir
    models_path = os.path.join(data_dir, "models")
    scenes_path = os.path.join(data_dir, scene_dir, "scene_data")
    if not os.path.exists(scenes_path):
        print(f"Path to scenes files does not exist!: {scenes_path}")
        exit(0)

    # enable ROS bridge extension
    extensions.enable_extension("omni.isaac.ros_bridge")
    simulation_app.update()

    # check if rosmaster node is running
    # this is to prevent this sample from waiting indefinetly if roscore is not running
    # can be removed in regular usage
    import rosgraph
    if not rosgraph.is_master_online():
        carb.log_error("Please run roscore before executing this script")
        simulation_app.close()
        exit()

    # set up world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Preparing stage
    viewports.set_camera_view(eye=np.array([3.5, 3.5, 3.5]), target=np.array([0, 0, 0.5]))

    # Loading the fetch robot USD
    asset_path = os.path.join(models_path, "fetch/fetch.usd")
    stage.add_reference_to_stage(usd_path=asset_path, prim_path=FETCH_STAGE_PATH)
    robot = world.scene.add(Articulation(prim_path = FETCH_STAGE_PATH, name="fetch"))
    robot.set_enabled_self_collisions(False)

    simulation_app.update()

    # Creating a action graph with ROS component nodes
    try:
        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
                    ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                    ("PublishJointState", "omni.isaac.ros_bridge.ROS1PublishJointState"),
                    ("SubscribeJointState", "omni.isaac.ros_bridge.ROS1SubscribeJointState"),
                    ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                    ("PublishTF", "omni.isaac.ros_bridge.ROS1PublishTransformTree"),
                    ("PublishClock", "omni.isaac.ros_bridge.ROS1PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "PublishTF.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
                    ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Setting the /Fetch target prim to Articulation Controller node
                    ("ArticulationController.inputs:usePath", True),
                    ("ArticulationController.inputs:robotPath", FETCH_STAGE_PATH),
                    ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(FETCH_STAGE_PATH)]),
                    ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(FETCH_STAGE_PATH)]),
                ],
            },
        )
    except Exception as e:
        print(e)
    simulation_app.update()

    # add table
    asset_path = os.path.join(models_path, "cafe_table_org/cafe_table_org/cafe_table_org.usd")
    stage.add_reference_to_stage(usd_path=asset_path, prim_path=TABLE_STAGE_PATH)
    world.scene.add(GeometryPrim(prim_path=TABLE_STAGE_PATH, name="table"))
    table_position = np.array([0.8, 0, 0.1]).reshape((1, 3))
    table_prim = RigidPrim(prim_path=TABLE_STAGE_PATH, position = table_position)     

    simulation_app.update() 

    # need to initialize physics getting any articulation..etc
    world.initialize_physics()

    # set robot pose
    joint_command = default_pose(robot)
    action = ArticulationAction(joint_positions=joint_command)
    robot.apply_action(action)

    world.play()

    while simulation_app.is_running():

        # Run with a fixed step size
        world.step(render=True)

        # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
        og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)

    world.stop()
    simulation_app.close() 

    return

    z_offset = -0.03  # difference between Real World and Gazebo table
    table_position = [0.8, 0, z_offset]
    objs.add_object(
        "cafe_table_org",
        [*table_position, 0, 0, 0, 1],
    )

    # Read in the selected scene ids
    scenes_list_f = os.path.join(data_dir, scene_dir, "scene_ids.txt")
    with open(scenes_list_f, "r") as f:
        sel_scene_ids = [int(x) for x in f.read().split()]

    while True:
        # for scene_id in sel_scene_ids:
        scene_id = input("Please provide the scene id: ")
        scene_id = int(scene_id)
        if scene_id not in sel_scene_ids:
            print("Provided scene id not in the list of selected scene ids")
            print(f"Valid ids: {sel_scene_ids}")
            continue
        print(f"-----------Scene:{scene_id}---------------")
        scene_file = os.path.join(scenes_path, f"scene_id_{scene_id}.pk")
        if not os.path.exists(scene_file):
            print(f"{scene_file} not found! Exiting...")
            break
        scene_info = load_scene(scenes_path, scene_id)
        # scene = scene_info["obj_poses"]
        scene = scene_info["obj_poses"]
        meta_f = "meta-%06d.mat" % scene_id
        meta = loadmat(os.path.join(data_dir, scene_dir, "metadata", meta_f))
        meta_obj_names = meta["object_names"]
        meta_poses = {}
        for i, obj in enumerate(meta_obj_names):
            meta_poses[obj] = convert_standard_to_rosqt(meta["poses"][i])
        print(f"Objects: {[obj for obj in scene]}---")
        # for obj, pose in scene.items():
        for obj, pose in meta_poses.items():
            objname = obj.strip()
            objs.add_object(objname, pose)
            rospy.sleep(2)
        objects_in_scene = [obj for obj in scene.keys()]
        print(objects_in_scene)

        confirmation = input("Next scene? y or n.....")
        for obj in scene:
            objs.delete_object(obj)
        if confirmation == "y":
            continue
        else:
            break
    # Deleting cafe table
    objs.delete_object("cafe_table_org")


   


if __name__ == "__main__":
    args = make_args()
    main(args)
