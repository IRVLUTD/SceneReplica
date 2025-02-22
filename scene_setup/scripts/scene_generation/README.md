# Scene Generation and Selection

## Overview
The scene generation proceeds entirely in gazebo simulation by trying to spawn objects on a table in a collision free manner. For a *scene*, 5 objects are spawned and a motion plan is checked to each of their grasps (from an offline grasp dataset). If all 5 objects have at least 1 valid motion plan to some grasp then we select (save) the scene.

The objects are spawned in a randomly chosen stable pose along with a random rotation around Z axis applied. The table's surface is discretized into a grid with each grid point denoting the possible spawn location (x,y) for an object. If a grid point is not feasible, we search for others in a breath-first-search like manner.

Lastly, from the several scenes generated, we attempt to obtain a dataset of 20 scenes with the following properties:
1. Each object must appear in 5-7 scenes
2. There should be diversity in poses of an object
3. Score a dataset by a diversity metric for stable pose used (we use entropy of count distribution)

# Setup and Usage
Source the appropirate ROS workspace with all the dependencies (gazebo, fetch_ros, moveit etc). Multiple terminals needed for gazebo, moveit, main python script and rviz.

- Setup the data directories (gazebo models, grasp data) into appropriate folders as described [here](../../README.md#data-setup)
- Gazebo and Moveit Launch: 
  - Launch gazebo with just the Fetch robot and `cafe_table_org` model. Launching with just the robot model avoids errors in launch collisions with table.
  - Launch moveit using the launch file for simulation
- In a third terminal, run `python generate_scenes.py` with appropriate command line args for data directory and number of rows and columns for the spawn grid. More rows/cols increases the number of possible object spawn locations and brings them closer (but also makes it hard to get all objects as reachable or increase possible collisions).

## Scene Generation
- `generate_scenes.py`: Main scripts to trigger the generation of scenes in gazebo. Check the command line args to give (main one is the data dir)
  - `--data_dir` : folder where a `scene_gen/iter_TIMESTAMP/` folders will be created to store selected scenes
  - `--rows` and `--cols` : rows and cols used to discretize the table grid
  - `--grasp` : whether to also try execute the motion plan to grasp (instead of just planning it, useful for viz and debugging) 

- `viz_scenes.py` : Visualize the selected scenes in gazebo. The setup steps are same as that for `generate_grasps.py` but here you just visualize the scenes.

## Saving Metadata for Scenes
This allows us to store the metadata for recreating the scene in real world

1. Follow the above steps for gazebo and moveit.
2. In a separate terminal, run `python save_pose_results.py`
   1. It will start subscribing to a topic published by `viz_scenes.py` and will
      save the appropriate data on callback
   2. Check the output dir used by this script and modify as cmd line arg if neeeded
3. Lastly run `python viz_scenes.py` which will start visualizing the scenes. You can iterate over the desired scenes to save their metadata.


## Selecting Final Scenes
1. Generate 20 Datasets, each having 20 scenes
  ```Shell
  python3 select_datasets.py --min=5 --max=7 --iterations=20 --scenes_dir=/path/to/scene_pickle_files/ 
  ```
2. Selecting Diverse dataset
  ```Shell
  python3 select_diverse.py  --scenes_dir=/path/to/scene_pickle_files/ 
  ```
3. Visualizing a dataset
  ```Shell
  python3 plot_scenes.py
  ```

