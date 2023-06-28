import os, sys
import argparse
import matplotlib.pyplot as plt
import cv2
import time
import tkinter as tk
from tkinter import filedialog

sys.path.append("../utils/")
from utils_scene import read_pickle_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "--image_dir",
        default="/home/benchmark/Datasets/benchmarking/scene_gen/",
        help="path to the Scenes dir",
    )

    args = parser.parse_args()

    while True:
        # Create a file selection dialog
        root = tk.Tk()
        root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
        # root.deiconify()  # restore the root window
        file_paths = (
            filedialog.askopenfilenames()
        )  # show an "Open" dialog box and return the path to the selected file
        root.destroy()  # add this

        for file_path in file_paths:
            if file_path:
                contents = read_pickle_file(file_path)

                fig = plt.figure(figsize=(10, 7), num=file_path)
                rows = 4
                columns = 5
                index = 0

                for _scene_id in contents.keys():
                    if _scene_id == "count":  # skip if _scene_id is not 13
                        continue
                    else:
                        scene_img_path = os.path.join(
                            args.image_dir, "color-%06d.jpg" % _scene_id
                        )  # replace with your actual image dir
                        scene_img = cv2.imread(scene_img_path).copy()

                        fig.add_subplot(rows, columns, index + 1)
                        plt.imshow(cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB))
                        plt.axis("off")
                        plt.title("Scene_" + str(_scene_id))
                        index += 1  # increment index for subplot

                plt.show()
                plt.close()
            else:
                break  # if no file is selected, break the loop
