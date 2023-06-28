import sys
import os
import argparse
import random
import math
import pickle
import json
from collections import deque

import numpy as np
import trimesh
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from transforms3d.euler import quat2euler
from transforms3d.quaternions import mat2quat, quat2mat

import rospy
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import (
    GetModelState,
    GetWorldProperties,
    SetModelState,
    DeleteModel,
    SpawnModel,
)

import utils_grid

# import utils.utils_grid as utils_grid


class WorldService:
    def __init__(self):
        """
        Initializes the get_world_properties and reset_world services
        """
        self.world_service_name = "/gazebo/get_world_properties"
        rospy.wait_for_service(self.world_service_name)
        self.property_service = rospy.ServiceProxy(
            self.world_service_name, GetWorldProperties
        )

        self.worldreset_service_name = "/gazebo/reset_world"
        rospy.wait_for_service(self.worldreset_service_name)
        self.reset_service = rospy.ServiceProxy(self.worldreset_service_name, Empty)

    def get_object_names_in_scene(self):
        """
        returns list of objetcs names found in the current scene
        """
        object_names = self.property_service().model_names
        print(f"{len(object_names)} found in the scene")
        return object_names

    def reset_world(self):
        """
        resets the objects present in scene to initial positions
        """
        self.reset_service()


class ObjectService:
    def __init__(
        self,
        models_base_path,
        get_service_name="/gazebo/get_model_state",
        set_service_name="/gazebo/set_model_state",
        del_object_service_name="/gazebo/delete_model",
        spawn_object_service_name="/gazebo/spawn_sdf_model",
    ):
        """
        Initializes the get, set model state services
        models_base_path: path to fetch_gazebo models dir (can be custom dir or symlinked).
                          Example: ~/Path_to_ws/src/fetch_gazebo/fetch_gazebo/models/
        """
        self.models_base_path = models_base_path

        rospy.wait_for_service(get_service_name)
        self.obj_get_service = rospy.ServiceProxy(get_service_name, GetModelState)

        rospy.wait_for_service(set_service_name)
        self.obj_set_service = rospy.ServiceProxy(set_service_name, SetModelState)

        rospy.wait_for_service(del_object_service_name)
        self.delete_object_client = rospy.ServiceProxy(
            "/gazebo/delete_model", DeleteModel
        )

        rospy.wait_for_service(spawn_object_service_name)
        self.add_object_client = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", SpawnModel
        )

        self.new_state = ModelState()
        self.pose = Pose()

        self.object_pose_inq = [0, 0, 0, 0, 0, 0, 0]
        self.object_pose_eul = [0, 0, 0, 0, 0, 0]
        self.reference = None

    def convert_quat_to_euler(self):
        """
        converts an orientation in quaternion [x,y,z,w] to euler [theta, phi, psi]
        """
        euler_orientation = quat2euler(self.object_pose_inq[3:])
        self.object_pose_eul[0:3] = self.object_pose_inq[0:3]
        self.object_pose_eul[3:] = euler_orientation

    def _ros_pose_as_qt(self, object_pose):
        """
        Get a list of [posn, quat] from a ros pose
        """
        qt_pose = [0, 0, 0, 0, 0, 0, 1]  # (x,y,z) position and (x,y,z,w) quat
        qt_pose[0] = object_pose.position.x
        qt_pose[1] = object_pose.position.y
        qt_pose[2] = object_pose.position.z
        qt_pose[3] = object_pose.orientation.x
        qt_pose[4] = object_pose.orientation.y
        qt_pose[5] = object_pose.orientation.z
        qt_pose[6] = object_pose.orientation.w
        return qt_pose

    def get_state(self, obj_name):
        """
        arguments: object name
        returns the pose of requested object in 2 formats:
        1. [x,y,z,theta,phi,psi] (for use in gazebo SDF)
        2. [posn, quat] (for general use case as quat is better for rotation)
        """
        object_pose = self.obj_get_service(obj_name, "").pose
        self.object_pose_inq[0] = object_pose.position.x
        self.object_pose_inq[1] = object_pose.position.y
        self.object_pose_inq[2] = object_pose.position.z
        self.object_pose_inq[3] = object_pose.orientation.x
        self.object_pose_inq[4] = object_pose.orientation.y
        self.object_pose_inq[5] = object_pose.orientation.z
        self.object_pose_inq[6] = object_pose.orientation.w

        self.convert_quat_to_euler()
        return self.object_pose_eul, self._ros_pose_as_qt(object_pose)

    def set_state(self, obj_name, target_pose=[0, 0, 0.75, 0, 0, 0, 1]):
        """
        arguments: object name, its pose as [x,y,z,theta,phi,psi]
        sets the given pose to the corresponsing object
        """
        self.new_state.model_name = obj_name
        self.new_state.pose.position.x = target_pose[0]
        self.new_state.pose.position.y = target_pose[1]
        self.new_state.pose.position.z = target_pose[2]
        self.new_state.pose.orientation.x = target_pose[3]
        self.new_state.pose.orientation.y = target_pose[4]
        self.new_state.pose.orientation.z = target_pose[5]
        self.new_state.pose.orientation.w = target_pose[6]
        response = self.obj_set_service(self.new_state)
        if response.success == True:
            print(f"{obj_name}: state set successfully")
        else:
            print(
                f"{obj_name}: failed to set state! object doesn't exists or invalid pose"
            )

    def delete_object(self, obj_name):
        """
        arguments: name of model to delete
        deletes the models from the current scene
        """
        response = self.delete_object_client(obj_name)
        if not response.success:
            print(
                f"{obj_name}: failed to delete! check name, existence of object in current scene"
            )

    def add_object(
        self,
        obj_name,
        spawn_pose=[0, 0, 0, 0, 0, 0, 1],
    ):
        """
        arguments: object name, spawn pose [X,Y,Z,x,y,z,w],
        """
        model_path = os.path.join(self.models_base_path, obj_name, f"model.sdf")

        if not os.path.exists(model_path):
            print(f"model path missing! please check: {model_path}")
        else:
            self.pose.position.x = spawn_pose[0]
            self.pose.position.y = spawn_pose[1]
            self.pose.position.z = spawn_pose[2]
            self.pose.orientation.x = spawn_pose[3]
            self.pose.orientation.y = spawn_pose[4]
            self.pose.orientation.z = spawn_pose[5]
            self.pose.orientation.w = spawn_pose[6]

            response = self.add_object_client(
                model_name=obj_name,
                model_xml=open(model_path, "r").read(),
                initial_pose=self.pose,
            )

            if not response:
                print("Failed to add object", obj_name)


class SdfService:
    def __init__(self, target_path_to_sdf_file, world_name="default"):
        self.file = target_path_to_sdf_file
        self.world_name = world_name

    def create_sdf_template(self):
        self.root = ET.Element("sdf", {"version": "1.4"})
        self.world = ET.Element("world", {"name": self.world_name})
        self.root.append(self.world)

    def add_default_models(self, default_models=["sun"], poses=[None]):
        for model_no, model in enumerate(default_models):
            self.add_model(model, poses[model_no])

    def add_model(self, model_name=None, pose=None):
        if model_name is None:
            print(f"skipping adding model; No Name provided")
        else:
            include_element = ET.SubElement(self.world, "include")
            uri_element = ET.SubElement(include_element, "uri")
            uri_element.text = f"model://{model_name}"
            if pose is not None:
                pose_element = ET.SubElement(include_element, "pose")
                pose = " ".join(str(round(e, 3)) for e in pose)
                pose_element.text = pose

    def write_to_sdf(self):
        xmlstr = minidom.parseString(ET.tostring(self.root)).toprettyxml(indent="  ")
        with open(self.file, "w") as f:
            f.write(xmlstr)


class SceneMaker:
    """
    Helper class to manage scenes and abstract away the logic for spawning collision
    free objects on the table.
    """

    def __init__(
        self,
        model_names,
        models_path,
        grid_size,
        table_position,
        table_height,
        stable_pose_f=None,
    ) -> None:
        self.model_names = model_names
        self.models_path = models_path
        self._grid_size = grid_size
        self._table_pos = table_position
        self._table_height = table_height
        self._stable_poses_f = stable_pose_f
        # Do some preprocessing for scene creation
        self._create_table_grid()
        self._process_object_models()

    def _create_table_grid(self):
        grid_size = self._grid_size
        _verts_org = utils_grid.generate_grid(
            grid_size[0], grid_size[1], self._table_pos, 1
        )
        # Only need the (x,y) locations for the grid.
        self.vertices = _verts_org[:, :-1].reshape(grid_size[0], grid_size[1], 2)

    def _process_object_models(self):
        self.model_meshes = {
            ycbid: utils_grid.load_mesh(self.models_path, ycbid)
            for ycbid in self.model_names
        }
        if self._stable_poses_f is not None:
            with open(self._stable_poses_f, "rb") as f:
                selected_poses = pickle.load(f)
            self.model_stable_poses = {
                ycbid: selected_poses[ycbid] for ycbid in self.model_names
            }
            print("Loaded stable poses for all objects from file")
        else:
            self.model_stable_poses = {
                ycbid: trimesh.poses.compute_stable_poses(
                    self.model_meshes[ycbid], threshold=0.02
                )
                for ycbid in self.model_names
            }
            print("Computed and stored stable poses for all objects!")

    def _rotZ(self, angle: float):
        """
        Gives a 4x4 transform (rotation matrix) about Z axis.
        Input:
        angle (float): rotation angle in radians
        Returns:
        tf_rotZ (np.array) : rotation transform
        """
        tf_rotZ = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return tf_rotZ

    def create_scene(self):
        start_node = (random.choice(range(self._grid_size[1])), 0)
        # start_node = (0, 0)
        seen_nodes = set()
        seen_rects = []
        nodelist = deque()
        nodelist.append(start_node)
        # Returns a dict with object names and their pose in Gazebo
        scene = {}
        stable_pose_idx = {}
        shuffled_models = self.model_names.copy()
        random.shuffle(shuffled_models)  # in-place operation, hence copy first
        # Now choose the first 5 object names in the shuffled list to create scene
        for curr_obj in shuffled_models[:5]:
            curr_mesh = self.model_meshes[curr_obj]
            tfs, _ = self.model_stable_poses[curr_obj]
            tf_idx = random.choice(range(len(tfs)))
            # random_tf = random.choice(tfs)  # change this to sample with prob?
            random_tf = tfs[tf_idx]
            # Apply random Z rotation
            rot_angle = random.uniform(0, math.pi)
            random_tf = self._rotZ(rot_angle) @ random_tf
            # print(f"Rot {curr_obj} about Z by {math.degrees(rot_angle)} degrees")
            obj_quat, _ = utils_grid.rt_to_ros_qt(random_tf)
            tf_mesh = curr_mesh.copy().apply_transform(random_tf)

            x_pos, y_pos = None, None
            while nodelist:
                curr_node = nodelist.popleft()
                seen_nodes.add(curr_node)
                # add neighbors of current node to the queue
                ngs = utils_grid.valid_neighbors(curr_node, self._grid_size, seen_nodes)
                for neighbor in ngs:
                    nodelist.append(neighbor)
                x, y = self.vertices[curr_node[0], curr_node[1]]
                curr_rect = utils_grid.create_rectange_from_bbox(tf_mesh, x, y)
                if utils_grid.check_collision(curr_rect, seen_rects):
                    print(f"Spawn {curr_obj} at {curr_node}")
                    seen_rects.append(curr_rect)
                    x_pos = x
                    y_pos = y
                    break

            # For z-coordinate, we need to have the height above table as
            # approximately half the length along z (for the transformed mesh)
            bbox_mesh = tf_mesh.bounding_box.vertices
            bbox_side = np.max(bbox_mesh, axis=0) - np.min(bbox_mesh, axis=0)
            offset_above_table = bbox_side[-1] / 2.0 + 0.005
            z_pos = self._table_height + offset_above_table
            pose = [
                x_pos,
                y_pos,
                z_pos,
                *obj_quat,
            ]  # pose = [posn, quat_xyzw]
            scene[curr_obj] = pose
            stable_pose_idx[curr_obj] = tf_idx

        return scene, stable_pose_idx


def write_pickle_file(contents, filename):
    with open(filename, "wb") as handle:
        pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    return True


def read_pickle_file(filename):
    with open(filename, "rb") as handle:
        contents = pickle.load(handle)
    handle.close()
    return contents


def get_successful_grasps(filename):
    """
        return a dictionary, with object names as keys
        and value of each key is the list of successful grasp indices
        ex: {'obj1':[1,31,2],'obj2':[22,3],...}
    """
    contents = read_pickle_file(filename)
    success_grasps = {}

    for _, sub_dict in contents.items():
        for key, value in sub_dict.items():
            if key in success_grasps:
                success_grasps[key].append(value)
            else:
                success_grasps[key] = [value]
    return success_grasps

def write_json(contents, filename):
    with open(filename, "w") as handle:
        json.dump(contents, handle)
    return True


def read_json(filename):
    with open(filename, "r") as handle:
        contents = json.load(handle)
    return contents


def load_scene(scene_dir, scene_id):
    scene_f = os.path.join(scene_dir, f"scene_id_{scene_id}.pk")
    with open(scene_f, "rb") as f:
        data = pickle.load(f)
    return data


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "--ws_dir",
        type="str",
        default="/home/ninad/Projects/Benchmarking/bench_ws/",
        help="path to the benchmarking workspace dir",
    )
    parser.add_argument(
        "-o",
        "--object",
        type="str",
        default="003_cracker_box",
        help="name of the ycb object",
    )
    args = parser.parse_args()
    return args


def main(args):
    ws_dir = args.ws_dir
    ycb_obj = args.object_name
    gz_model_path = "fetch_gazebo/fetch_gazebo/models/"
    models_path = os.path.join(ws_dir, "src", gz_model_path)

    ws = WorldService()
    objs = ObjectService(models_base_path=models_path)
    print(ws.get_object_names_in_scene())

    objs.add_object(
        ycb_obj,
        [1, 0, 0.8, 0, 0, 0, 1],
    )


if __name__ == "__main__":
    args = make_args()
    main(args)
