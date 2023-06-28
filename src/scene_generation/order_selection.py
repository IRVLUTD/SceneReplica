import os
import sys
import argparse
sys.path.append("../utils")
from utils_scene import read_pickle_file, write_pickle_file
from numpy.linalg import norm
from operator import itemgetter
from random import shuffle


def order(objects_dict, type=0):
    """
        objects = dictionary of objects names, positions
        type = 0 for nearest first, 1 for random order
    """
    # print(f"unsorted {objects_dict}")
    _sort = lambda x: norm(x[0])
    if type==0:
        distance_dict = {object_name: _sort(pos) for object_name, pos in objects_dict.items()}
        sorted_dict = dict(sorted(distance_dict.items(), key=itemgetter(1)))
        sorted_dict = {object_name: objects_dict[object_name] for object_name in sorted_dict.keys()}
        sorted_objects = list(sorted_dict.keys())
    else:
        object_names = list(objects_dict.keys())
        shuffle(object_names)
        sorted_dict = {object_name: objects_dict[object_name] for object_name in object_names}
        sorted_objects = list(sorted_dict.keys())

    # print(f"sorted_dict {sorted_dict}")

    return sorted_objects


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sort pick up order", add_help=True
    )
    parser.add_argument(
        "--scenes_dir",
        default="/home/benchmark/Datasets/benchmarking/scene_gen/",
        help="path to the Scenes dir",
    )

    args = parser.parse_args()
    scenes = [os.path.join(args.scenes_dir,file) for file in os.listdir(args.scenes_dir) if file.endswith('.pk')]

    for scene in scenes:
        # print(scene,"\n")
        scene_content = read_pickle_file(scene)
        objects_dict = scene_content['gz_obj_poses']
        # nearest first type 0
        sorted_objects = order(objects_dict, type=0)
        scene_content["nearest_first"] = sorted_objects
        #random order type 1
        sorted_objects = order(objects_dict, type=1)
        scene_content["random"] = sorted_objects
        write_pickle_file(scene_content, scene)

