import os
import time, sys
import logging
import argparse
from shutil import copy
import cv2
from scipy.stats import entropy

sys.path.append("../utils")
from utils_scene import read_pickle_file, read_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )

    parser.add_argument(
        "--scenes_dir",
        default="/home/benchmark/Datasets/benchmarking/scene_gen/",
        help="path to the Scenes dir with pickle files",
    )
    parser.add_argument(
        "--dataset_dir",
        default="../saved_datasets/test",
        help="path to a saved datatsets folder with statistics (result after running select_datasets.py)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename=os.path.join(args.dataset_dir, "diversity.log"),
        format="%(message)s",
        filemode="w",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logging.info(f"dataset directory: {args.dataset_dir}")
    logger.info(f"++---------------------------------------++\n")

    stats_file = os.path.join(args.dataset_dir, "statistics.json")

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

    logger.info(f"objects used: {objects}")
    logger.info(f"++---------------------------------------++\n")
    stats = read_json(stats_file)
    max_score = 0
    max_dataset = ""

    for dataset, dataset_info in stats.items():
        # enter dataset having 20 scenes
        diverse_count = {obj: [] for obj in objects}
        score = 0

        for _scene_id in dataset_info["scenes"]:
            # enter scene having 5 objects

            scene_content = read_pickle_file(
                os.path.join(args.scenes_dir, f"scene_id_{_scene_id}.pk")
            )
            for object, Spose_id in scene_content["stable_pose_idx"].items():
                diverse_count[object].append(Spose_id)

        for object, count in diverse_count.items():
            if object in {
                "011_banana",
                "024_bowl",
                "040_large_marker",
                "052_extra_large_clamp",
                "037_scissors",
            }:
                continue
            unique_it = len(set(count))
            if unique_it > 1:
                score += entropy(count, base=2)
            else:
                score = -1
                break

        if score > max_score:
            max_score = score
            max_dataset = dataset

        # print(f"++---------------------------------------++")
        logger.info(f" {(dataset):>9}    score: {score:<21}")

    logger.info(f"++---------------------------------------++\n")
    logger.info(f"selected {max_dataset}, having score {max_score}")
    logger.info(f"++---------------------------------------++")

    copy(
        os.path.join(args.dataset_dir, max_dataset),
        os.path.join(args.dataset_dir, "diverse_dataset.pk"),
    )
    logger.info("copied diverse dataset into diverse_dataset.pk")
