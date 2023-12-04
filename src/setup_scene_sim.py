import os
import argparse
import sys
from scipy.io import loadmat
import rospy

sys.path.append("./utils/")
from utils.utils_scene import ObjectService, load_scene
from utils_control import PointHeadClient, FollowTrajectoryClient
from ros_utils import convert_standard_to_rosqt


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/home/benchmark/Datasets/benchmarking/",
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


def main(args):
    data_dir = args.data_dir
    scene_dir = args.scene_dir
    scenes_path = os.path.join(data_dir, scene_dir, "scene_data")
    if not os.path.exists(scenes_path):
        print(f"Path to scenes files does not exist!: {scenes_path}")
        exit(0)

    models_path = os.path.join(data_dir, "models")
    objs = ObjectService(models_base_path=models_path)
    rospy.init_node("VizSceneSim")
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    head_action = PointHeadClient()
    rospy.loginfo("Raising torso...")
    torso_action.move_to(
        [
            0.4,
        ]
    )
    for i in range(3):
        head_action.look_at(0.45, 0, 0.75, "base_link")
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
