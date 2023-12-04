import os
import argparse
import sys
from scipy.io import loadmat
import numpy as np

# isaac
import carb
from omni.isaac.kit import SimulationApp

FETCH_STAGE_PATH = "/World/Fetch"
TABLE_STAGE_PATH = "/World/Table"
CAMERA_STAGE_PATH = FETCH_STAGE_PATH + "/head_camera_rgb_frame/Camera"
CONFIG = {"renderer": "RayTracedLighting", "headless": False}

# set up isaac environment
simulation_app = SimulationApp(CONFIG)

# Example ROS bridge sample demonstrating the manual loading of stages
# and creation of ROS components
import omni
import omni.graph.core as og
import usdrt.Sdf
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils import extensions, stage, viewports
from omni.isaac.core.prims.rigid_prim import RigidPrimView

# SceneReplica
sys.path.append("./utils/")
from utils.utils_scene import load_scene


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
    parser.add_argument(
        "-i",
        "--scene_index",
        type=int,
        default=0,
        help="Index for the scene to be created",
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
    scene_index = args.scene_index
    models_path = os.path.join(data_dir, "models")
    scenes_path = os.path.join(data_dir, scene_dir, "scene_data")
    if not os.path.exists(scenes_path):
        print(f"Path to scenes files does not exist!: {scenes_path}")
        exit(0)
    # Read in the selected scene ids
    scenes_list_f = os.path.join(data_dir, scene_dir, "scene_ids.txt")
    with open(scenes_list_f, "r") as f:
        sel_scene_ids = [int(x) for x in f.read().split()]
    scene_id = sel_scene_ids[scene_index]
    print('Setting up Scene ', scene_id)

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

    ############### Calling Camera publishing functions ###############
    # stage_usd = omni.usd.get_context().get_stage()
    # OpenCV camera matrix and width and height of the camera sensor, from the calibration file
    # https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_camera.html?highlight=set%20camera%20parameter#calibrated-camera-sensors
    width, height = 640, 480
    viewport_api = omni.kit.viewport.utility.get_active_viewport()
    viewport_api.set_texture_resolution((width, height))

    # Creating a action graph with ROS component nodes
    try:
        (ros_camera_graph, nodes, _, _) = og.Controller.edit(
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
                    ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                    ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
                    ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                    ("cameraHelperRgb", "omni.isaac.ros_bridge.ROS1CameraHelper"),
                    ("cameraHelperInfo", "omni.isaac.ros_bridge.ROS1CameraHelper"),
                    ("cameraHelperDepth", "omni.isaac.ros_bridge.ROS1CameraHelper"),                    
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
                    ("OnImpulseEvent.outputs:execOut", "createViewport.inputs:execIn"),                    
                    ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                    ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                    ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                    ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                    ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                    ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
                    ("getRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),                    
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Setting the /Fetch target prim to Articulation Controller node
                    ("ArticulationController.inputs:usePath", True),
                    ("ArticulationController.inputs:robotPath", FETCH_STAGE_PATH),
                    ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(FETCH_STAGE_PATH)]),
                    ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(FETCH_STAGE_PATH)]),
                    ("createViewport.inputs:viewportId", 0),
                    ("cameraHelperRgb.inputs:frameId", "sim_camera"),
                    ("cameraHelperRgb.inputs:topicName", "head_camera/rgb/image_raw"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperInfo.inputs:frameId", "sim_camera"),
                    ("cameraHelperInfo.inputs:topicName", "head_camera/rgb/camera_info"),
                    ("cameraHelperInfo.inputs:type", "camera_info"),
                    ("cameraHelperDepth.inputs:frameId", "sim_camera"),
                    ("cameraHelperDepth.inputs:topicName", "head_camera/depth_registered/image_raw"),
                    ("cameraHelperDepth.inputs:type", "depth"),
                    ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_STAGE_PATH)]),                    
                ],
            },
        )
    except Exception as e:
        print(e)
    tf_int_attr = og.Controller.attribute("inputs:targetPrims", nodes[5])

    # Run the ROS Camera graph once to generate ROS image publishers in SDGPipeline
    og.Controller.evaluate_sync(ros_camera_graph)     
    simulation_app.update()

    # need to initialize physics getting any articulation..etc
    world.initialize_physics()

    # set robot pose
    joint_command = default_pose(robot)
    action = ArticulationAction(joint_positions=joint_command)
    robot.apply_action(action)

    world.play()
    # move the robot by 200 steps first
    for step in range(200):
        world.step(render=True)

    # add table
    asset_path = os.path.join(models_path, "cafe_table_org/cafe_table_org/cafe_table_org.usd")
    stage.add_reference_to_stage(usd_path=asset_path, prim_path=TABLE_STAGE_PATH)
    # z_offset = -0.03  # difference between Real World and Gazebo table
    table_position = np.array([0.8, 0, 0]).reshape((1, 3))
    table_rigid_prim_view = RigidPrimView(prim_paths_expr=TABLE_STAGE_PATH + "/baseLink")
    world.scene.add(table_rigid_prim_view)
    table_rigid_prim_view.set_world_poses(positions=table_position)
    simulation_app.update()

    # set up YCB objects
    while True:
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
        num = len(meta_obj_names)
        
        positions = np.zeros((num ,3), dtype=np.float32)
        orientations = np.zeros((num ,4), dtype=np.float32)
        tf_paths = [usdrt.Sdf.Path(FETCH_STAGE_PATH)]
        for i, obj in enumerate(meta_obj_names):
            objname = obj.strip()
            pose = meta["poses"][i]
            positions[i, :] = pose[:3]
            orientations[i, :] = pose[3:]

            filename = os.path.join(objname, objname + ".usd")
            asset_path = os.path.join(models_path, filename)
            obj_prim_path = "/World/YCB_" + objname
            print(asset_path, obj_prim_path)
            stage.add_reference_to_stage(usd_path=asset_path, prim_path=obj_prim_path)

            print(positions[i], orientations[i])
            rigid_prim_view = RigidPrimView(prim_paths_expr=obj_prim_path + "/object_" + objname + "_base_link", name=obj_prim_path)
            world.scene.add(rigid_prim_view)
            rigid_prim_view.set_world_poses(positions=positions[i].reshape((1,3)), orientations=orientations[i].reshape((1, 4)))
            tf_paths.append(usdrt.Sdf.Path(obj_prim_path))

        # set tf publisher
        print(tf_paths)
        og.DataView.set(attribute=tf_int_attr, value=tf_paths)
        simulation_app.update()

        objects_in_scene = [obj for obj in scene.keys()]
        print(objects_in_scene)
        break

    while simulation_app.is_running():

        # Run with a fixed step size
        world.step(render=True)

        # Tick the Publish/Subscribe JointState, Publish TF and Publish Clock nodes each frame
        og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)

    world.stop()
    simulation_app.close() 


if __name__ == "__main__":
    args = make_args()
    main(args)