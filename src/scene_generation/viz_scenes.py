import os, argparse, sys
import pickle
from scipy.io import savemat

import rospy
from std_msgs.msg import String

sys.path.append("../utils/")
from utils_scene import ObjectService, load_scene
from utils_control import PointHeadClient, FollowTrajectoryClient
from ros_utils import convert_rosqt_to_standard


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/home/ninad/Datasets/benchmarking/",
        help="Path to data dir",
    )
    parser.add_argument(
        "-s",
        "--scene_dir",
        type=str,
        default="iter_2023-05-15_15-25-42",
        help="Path to data dir",
    )
    parser.add_argument(
        "-i",
        "--interactive_viz",
        action="store_true",
        help="Flag for whethere to visualize interactively, \
              i.e wait for user confirmation before moving to next scene",
    )
    args = parser.parse_args()
    return args


def main(args):
    data_dir = args.data_dir
    scene_dir = args.scene_dir
    iviz_flag = args.interactive_viz
    scenes_path = os.path.join(data_dir, "scene_gen", scene_dir)
    if not os.path.exists(scenes_path):
        print(f"Path to scenes files does not exist!: {scenes_path}")
        exit(0)
    outdir = os.path.join(scenes_path, "metadata")
    os.makedirs(outdir, exist_ok=True)

    models_path = os.path.join(data_dir, "gazebo_models_all_simple")
    objs = ObjectService(models_base_path=models_path)

    rospy.init_node("viz")
    pub = rospy.Publisher("scene_change", String, queue_size=10)
    torso_action = FollowTrajectoryClient("torso_controller", ["torso_lift_joint"])
    head_action = PointHeadClient()
    rospy.loginfo("Raising torso...")
    torso_action.move_to(
        [
            0.4,
        ]
    )

    for i in range(5):
        head_action.look_at(0.45, 0, 0.75, "base_link")

    z_offset = -0.03  # difference between Real World and Gazebo table
    table_position = [0.8, 0, z_offset]

    objs.add_object(
        "cafe_table_org",
        [*table_position, 0, 0, 0, 1],
    )

    for i in range(1, 201):
        print(f"-----------Scene:{i}---------------")
        scene_file = os.path.join(scenes_path, f"scene_id_{i}.pk")
        if not os.path.exists(scene_file):
            print(f"{scene_file} not found! Exiting...")
            break
        scene_info = load_scene(scene_file)
        scene = scene_info["obj_poses"]
        print(f"Objs: {[obj for obj in scene]}---")
        for obj, pose in scene.items():
            objs.add_object(obj, pose)
            print(obj, objs.get_state(obj)[1])
        # print(ws.get_object_names_in_scene())
        rospy.sleep(2)
        pub_msg = f"scene_{i}"
        pub.publish(pub_msg)
        objects_in_scene = [obj for obj in scene.keys()]
        print(objects_in_scene)
        meta = {
            "object_names": objects_in_scene,
            "poses": [
                # convert_rosqt_to_standard(pose) for _,pose in scene.items()
                convert_rosqt_to_standard(objs.get_state(obj)[1])
                for obj in scene.keys()
            ],  # info = (obj_name, pose -- (w, x, y, z))
            "factor_depth": 1000.0,
        }
        fname = "meta-%06d.mat" % i
        savemat(os.path.join(outdir, fname), meta, do_compression=True)

        rospy.sleep(2)
        for obj in scene:
            objs.delete_object(obj)

        if iviz_flag:
            confirmation = input("Next scene? y or n.....")
            if confirmation == "y":
                continue
            else:
                break


if __name__ == "__main__":
    args = make_args()
    main(args)
