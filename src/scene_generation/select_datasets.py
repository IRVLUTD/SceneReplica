import os, sys
import pickle
import random
import argparse
import numpy as np
from datetime import datetime
from operator import itemgetter

sys.path.append("../utils")
from utils_scene import write_pickle_file, read_pickle_file, write_json, read_json


def sort_key(element):
    e1 = element.split("scene_id_")[1]
    e2 = e1.split(".pk")[0]

    return int(e2)


def select_scenes(unshuffled_scenes, objects, min_appearances=2, max_appearances=2):
    print("+--------------------------------------------+")
    print(f"objects: {objects}\n")
    print("+--------------------------------------------+")
    print(f"min {min_appearances} max {max_appearances}\n")

    selected_scenes = {}

    while True:
        object_counts = {obj: 0 for obj in objects}
        selected_combinations = []
        scene_ids = list(range(1, len(unshuffled_scenes) + 1))
        # print(scene_ids)
        # shuffle the scenes, so the bias wont be always on the earliest ones
        scenes = unshuffled_scenes.copy()
        random.shuffle(scenes)

        for combination in scenes:
            if all(object_counts[obj] < max_appearances for obj in combination):
                selected_combinations.append(combination)
                for obj in combination:
                    object_counts[obj] += 1
            if len(selected_combinations) == 20:
                break

        # check if each object appears at least 5 times
        if all(count >= min_appearances for count in object_counts.values()):
            break
        else:
            print("Failed to find a valid solution, trying again...")
    shuffled_scene_ids = [
        scene_ids[unshuffled_scenes.index(item)]
        for index, item in enumerate(selected_combinations)
    ]
    # print(shuffled_scene_ids)
    print("+--------------------------------------------+")
    print(f"20 scenes found")
    print("+--------------------------------------------+")
    print("+--------------------------------------------+")
    print(f"sceneID              Scene")
    print("+--------------------------------------------+")
    print("+--------------------------------------------+")
    for i, combination in enumerate(selected_combinations):
        print(f"  {shuffled_scene_ids[i]:<12}      {combination}")
        selected_scenes[shuffled_scene_ids[i]] = combination
        print("+--------------------------------------------+")

    selected_scenes["count"] = object_counts
    print(f"count {object_counts} \n")

    return selected_scenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "--scenes_dir",
        default="/home/benchmark/Datasets/benchmarking/scene_gen/",
        help="path to the Scenes dir",
    )
    parser.add_argument(
        "-min",
        "--min_appearances",
        type=int,
        default="5",
        help="minimum appearance of an object in all of the scenes",
    )
    parser.add_argument(
        "-max",
        "--max_appearances",
        type=int,
        default="7",
        help="minimum appearance of an object in all of the scenes",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default="20",
        help="no of unique 20 datasets need to be generated",
    )
    args = parser.parse_args()

    # scene_data_location = "./Scenes_2023-05-06/scene_info"
    scene_data_location = args.scenes_dir

    selection = []
    files = sorted(
        [file for file in os.listdir(scene_data_location) if file.endswith(".pk")],
        key=sort_key,
    )
    for file in files:
        pickle_contents = read_pickle_file(os.path.join(scene_data_location, file))
        selection.append(tuple(pickle_contents["gz_obj_poses"].keys()))

    # objectnames used in scene generation
    objects = [
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
        "052_extra_large_clamp",
    ]

    save_path = "../saved_datasets"
    # check for data saving folder - present or not
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    num_dir_present = len(os.listdir(save_path))
    # saving the run data
    today = str(num_dir_present + 1) + " #_" + str(datetime.now())[:-10]

    sub_folder = os.path.join(save_path, today)

    os.mkdir(sub_folder)

    for iteration in range(args.iterations):
        print(f"iteration {iteration}")
        final_scenes = select_scenes(
            selection, objects, args.min_appearances, args.max_appearances
        )

        write_pickle_file(
            final_scenes, os.path.join(sub_folder, f"dataset_{iteration+1}.pk")
        )

        contents = read_pickle_file(
            os.path.join(sub_folder, f"dataset_{iteration+1}.pk")
        )

        # print(f"contents {contents}")

    statistics_file = os.path.join(sub_folder, "statistics.json")
    stats = {}
    for dataset in os.listdir(sub_folder):
        stats[dataset] = {}
        scene_nums = []
        contents = read_pickle_file(os.path.join(sub_folder, dataset))
        stats[dataset]["count"] = contents["count"]
        for item in contents.keys():
            if item == "count":
                continue
            else:
                scene_nums.append(item)
        stats[dataset]["scenes"] = scene_nums

    write_json(stats, statistics_file)
