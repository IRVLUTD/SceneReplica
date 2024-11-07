import os, sys
import math
import json
import copy
import numpy as np
import trimesh
import cv2
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

import rospy
from ros_utils import ros_qt_to_rt
from tf.transformations import euler_matrix


################ Lifting, Standoff, Movement Utils #####################
def rotate_gripper(group, RT_gripper):
    """
    This action takes in the gripper's current pose and slightly rotates
    the wrist_roll_joint (last joint in the planning group "arm" for Fetch)
    joints[-1] := wrist_roll
    joints[-2] := wrist_flex

    Input:
    - group : planning group "arm" for Fetch robot
    - RT_gripper : 4x4 tf for wrist_roll_joint (gripper) w.r.t base_link
    """
    group.stop()
    joint_goal = group.get_current_joint_values()
    joint_goal[-1] += np.radians(30)
    # wrist flex joint, index 5, limit -2.1
    # joint_goal[-2] = max(joint_goal[-2] + np.radians(30), -2.1)
    group.go(joint_goal, wait=True)
    group.stop()
    rospy.sleep(1)


def move_arm_to_dropoff(group, RT_gripper, x_final=0.45, y_final=0.4):
    """
    Move arm to a final dropoff location
    Use the x_final and y_final values to plan a cartesian path
    """
    # modified fetch.srdf in fetch_moveit_config to have a new group called "wrist" just
    # containing the wrist_roll_joint
    # print(group.get_current_joint_values())
    # group_w = moveit_commander.MoveGroupCommander("wrist")
    # group_w.set_max_velocity_scaling_factor(1.0)
    # wpose = group_w.get_current_pose().pose

    # euler = mat2euler(RT_gripper[:3, :3])
    # roll = -euler[0]
    # pitch = -euler[1]
    # joint_goal = group.get_current_joint_values()
    # joint_goal[-1] = roll
    # joint_goal[-2] = pitch
    # group.go(joint_goal, wait=True)
    # group.stop()

    def set_pose_posn(wpose, pos):
        wpose.position.x = pos[0]
        wpose.position.y = pos[1]
        wpose.position.z = pos[2]

    gripper_posn = RT_gripper[:3, 3]
    x_start = gripper_posn[0]
    wps = np.linspace(x_start, x_final, 10, endpoint=True)

    # first_posn = [gripper_posn[0], y_final, gripper_posn[2]]
    # For movement in Z for object dropoff
    # final_posn = [x_final, gripper_posn[1], gripper_posn[2] - 0.1]

    waypoints = []
    wpose = group.get_current_pose().pose
    for p in wps:
        curr_pos = [p, gripper_posn[1], gripper_posn[2]]
        set_pose_posn(wpose, curr_pos)
        waypoints.append(copy.deepcopy(wpose))
    # set_pose_posn(wpose, final_posn)
    # waypoints.append(copy.deepcopy(wpose))

    (plan_standoff, fraction) = group.compute_cartesian_path(
        waypoints, 0.01, True  # waypoints to follow  # eef_step
    )  # jump_threshold
    print(f"Fraction for final dropoff movement: {fraction}")
    group.execute(plan_standoff, wait=True)
    group.stop()
    group.clear_pose_targets()

    # group.stop()
    joint_goal = group.get_current_joint_values()
    RAD_60 = np.radians(60)
    joint_goal[0] = RAD_60 if  joint_goal[0] >= 0 else -RAD_60
    group.go(joint_goal, wait=True)
    group.stop()
    rospy.sleep(0.5)


def lift_arm_joint(group, confirm=True):
    # lift the object
    offset = -0.4
    pose = group.get_current_joint_values()

    # shoulder lift joint
    limit = -1.2
    if pose[1] + offset < limit:
        pose[1] = limit
        reach_limit = True
    else:
        pose[1] += offset
        reach_limit = False

    # wrist flex joint, index 5, limit -2.1
    if reach_limit:
        pose[5] = max(pose[5] + offset, -2.1)

    group.set_joint_value_target(pose)
    plan = group.plan()

    if not plan[0]:
        print("no plan found in lifting")
        group.go(pose, wait=True)
    else:
        if confirm and user_confirmation("Lift object"):
            trajectory = plan[1]
            group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()
    return


def lift_arm_cartesian(group, RT_gripper, z_offset=0.25):
    def set_pose_posn(wpose, pos):
        wpose.position.x = pos[0]
        wpose.position.y = pos[1]
        wpose.position.z = pos[2]

    # gpos = RT_gripper[:3, 3]
    gpos = RT_gripper[:3, 3]
    z_start = gpos[2]
    z_final = gpos[2] + z_offset
    wps = np.linspace(z_start, z_final, 10, endpoint=True)
    # first_posn = [gpos[0], gpos[1], gpos[2] + 0.13]
    # For movement in Z for object dropoff
    # final_posn = [gpos[0], gpos[1], gpos[2] + 0.25]
    waypoints = []    
    wpose = group.get_current_pose().pose
    for z in wps:
        curr_pos = [gpos[0], gpos[1], z]
        set_pose_posn(wpose, curr_pos)
        waypoints.append(copy.deepcopy(wpose))
    
    (plan_standoff, fraction) = group.compute_cartesian_path(
        waypoints, 0.01, True # waypoints to follow  # eef_step
    )  # avoid_collision instead of jump_threshold
    print(f"Fraction for lifitng movement: {fraction}")
    group.execute(plan_standoff, wait=True)
    group.stop()
    group.clear_pose_targets()
    rospy.sleep(2)  #


def lift_arm_pose(group, confirm=True):
    # lift the object
    offset = 0.2
    rospy.loginfo("lift object")

    pose_goal = group.get_current_pose().pose
    pose_goal.position.z += offset
    group.set_pose_target(pose_goal)

    plan = group.plan()
    trajectory = plan[1]
    if not plan[0]:
        print("no plan found to lift position")
        return

    if confirm:
        if user_confirmation("Move to lift position"):
            pass
        else:
            sys.exit(1)

    group.execute(trajectory, wait=True)
    group.stop()
    group.clear_pose_targets()


def get_standoff_wp_poses(standoff_dist=0.1, tail_len=10, extra_off=0.01):
    """
    For any 6Dof Grasp pose (as 4x4 tf), compute the standoffpose and waypoints along
    the direction from standoff to final pose.

    Input:
    - standoff_dist (float) : how far (in meters) the standoff pose from the final grasp pose
    - tail_len (int) : how many waypoints from standoff to final
    - extra_off (float) : add an extra waypoint (in meters) from final pose to grasp object more firmly

    Returns:
    - pose_standoff (np.ndarray) : (tail_len+1, 4, 4) numpy array containing tfs for all waypoints in
                                    gripper frame. So just premultiply with RT_gripper to get in global frame.
    """
    offset = -standoff_dist * np.linspace(0, 1, tail_len, endpoint=False)[::-1]
    offset = np.append(offset, [extra_off])
    tail_len += 1
    pose_standoff = np.tile(np.eye(4), (tail_len, 1, 1))
    pose_standoff[:, 0, 3] = offset
    return pose_standoff


################ Model Free and Point Cloud Utils #####################


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def get_object_verts(model_path: str, pose):
    """
    Loads the points (vertices) for an object model after
    applying the given pose
    Input:
        model_path (str): path to the mesh file for the object
        pose (np.ndarray): 4x4 matrix specifying the transform
    """
    if not os.path.exists(model_path):
        print("[ERROR]: provided model path does not exist!")
        return
    mesh = trimesh.load(model_path)
    mesh.apply_transform(pose)
    return mesh.vertices


def model_based_top_down_grasp(points_base):
    """
    Gives a top down grasp for an object, assumes a parallel jaw gripper
    Input:
        points_base: (N,3) array of object points in robot base frame
    Returns:
        RT_grasp: 4x4 tf giving the transform for the grasp
        gripper_width: width for the parallel jaw gripper
    """
    gripper_finger_length = 0.062
    gripper_tip_to_base_offset = 0.216  # length from finger tip to gripper base
    margin_height = 0.005  # margin of error in estimating object's height
    margin_width = 0.0015

    xy = points_base[:, :2]
    center = np.mean(xy, axis=0).reshape((1, 2))
    xy_centered = xy - center
    pca = PCA(n_components=2)
    xy_new = pca.fit_transform(xy_centered)
    # we consider 3 grasps in the transformed xy space, each with two end points
    upper = np.max(xy_new[:, 1]) + margin_width
    lower = np.min(xy_new[:, 1]) - margin_width
    gripper_width = upper - lower

    half = gripper_width / 2
    grasps = np.zeros((6, 2), dtype=np.float32)
    # grasp 1
    grasps[0, :] = (0, half)
    grasps[1, :] = (0, -half)
    # grasp 2
    grasps[2, :] = (0, upper)
    grasps[3, :] = (0, upper - gripper_width)
    # grasp 3
    grasps[4, :] = (0, lower + gripper_width)
    grasps[5, :] = (0, lower)
    # transform grasps to the original space
    grasps_origin = pca.inverse_transform(grasps)
    grasps_origin += center

    # use the 1st grasp by default
    grasp_center = (grasps_origin[0, :] + grasps_origin[1, :]) / 2
    # use the 2nd component to define the grasping angle
    component = pca.components_[1, :]
    theta = math.atan2(component[0], component[1])
    # construct the RT matrix
    RT = euler_matrix(0, np.pi / 2, 0) @ euler_matrix(theta, 0, 0)
    RT[0, 3] = grasp_center[0]
    RT[1, 3] = grasp_center[1]
    # deal with noises in z values
    z_pts = points_base[:, 2]
    print("TOP-DOWN STATS-------------------------")
    z_max = np.max(z_pts)
    z_min = np.min(z_pts)
    z_mean = np.mean(z_pts)
    height = z_max - z_min
    z_c = z_min + height/2 # middle point (not always equal to mean)
    print("H, max, min, mean, center", height, z_max, z_min, z_mean, z_c)
    if gripper_finger_length >= height:
        # try to go down the entire object and grasp from bottom
        z_tip = z_min + 0.003 # 1mm offset
    else:
        # NOTE: Two cases to consider here
        # Case1: H/2 > L           ---> ztip = z_c + (H/2 - L) {go a bit more up than center}
        # Case2: H/2 < L but H > L ---> ztip = z_c - (L - H/2) {go a bit more down than center} == z_c + (H/2 - L)
        z_tip = z_c + (height/2 - gripper_finger_length)
    z_tip = max(z_tip, 0.745) # table height check
    print("Z_TIP:", z_tip)
    z_gripper_base = z_tip + gripper_tip_to_base_offset
    RT[2, 3] = z_gripper_base
    return RT, gripper_width


def model_free_top_down_grasp(camera_pose, mask_id, label, xyz_image, percent_filter=0.025):

    gripper_finger_length = 0.062
    gripper_tip_to_base_offset = 0.216
    margin_width = 0.0015
    
    # target mask
    mask = (label == mask_id).astype(np.uint8)

    # erode mask
    kernel = np.ones((3, 3), np.uint8)          
    mask2 = cv2.erode(mask, kernel)
    
    # process all the points
    depth = xyz_image[:, :, 2]    
    index = depth > 0
    points_all = xyz_image[index, :]
    labels_all = label[index]
    points_base_all = np.matmul(camera_pose[:3, :3], points_all.T) + camera_pose[:3, 3].reshape((3, 1))
    points_base_all = points_base_all.T
    
    # create a KD tree for the base points in xy plane
    tree = KDTree(points_base_all[:, :2])
    
    # process points for the target
    index = (mask2 > 0) & (depth > 0)
    points = xyz_image[index, :]
    points_base = np.matmul(camera_pose[:3, :3], points.T) + camera_pose[:3, 3].reshape((3, 1))
    points_base = points_base.T

    # perform pca
    xy = points_base[:, :2]    
    center = np.mean(xy, axis=0).reshape((1, 2))
    xy_centered = xy - center
    pca = PCA(n_components=2)
    xy_new = pca.fit_transform(xy_centered)
    
    # we consider 3 grasps in the transformed xy space, each with two end points
    upper = np.max(xy_new[:, 1]) + margin_width
    lower = np.min(xy_new[:, 1]) - margin_width
    gripper_width = upper - lower    
    
    half = gripper_width / 2
    grasps = np.zeros((6, 2), dtype=np.float32)
    # grasp 1
    grasps[0, :] = (0, half)
    grasps[1, :] = (0, -half)
    # grasp 2
    grasps[2, :] = (0, upper)
    grasps[3, :] = (0, upper - gripper_width)
    # grasp 3
    grasps[4, :] = (0, lower + gripper_width)
    grasps[5, :] = (0, lower)
    
    if gripper_width > (0.1 - 0.005):
        delta = gripper_width - (0.1 - 0.005)
        grasps[:, 1] += max(0, (delta/2.1 + 0.005))

    # transform grasps to the original space
    grasps_origin = pca.inverse_transform(grasps)
    grasps_origin += center
    
    # select one of the grasps
    end1 = grasps_origin[0, :]
    end2 = grasps_origin[1, :]
    d, index1 = tree.query(end1)
    l1 = labels_all[index1]
    d, index2 = tree.query(end2)
    l2 = labels_all[index2]
    print('end point1 label', l1, 'end point2 label', l2)
    
    # use the 1st grasp by default
    grasp_center = (grasps_origin[0, :] + grasps_origin[1, :]) / 2
    if l1 > 0 and l2 == 0:
        # use the 2nd grasp
        grasp_center = (grasps_origin[2, :] + grasps_origin[3, :]) / 2
        print('select 2nd top-down grasp')
    elif l1 == 0 and l2 > 0:
        # use the 3rd grasp
        grasp_center = (grasps_origin[4, :] + grasps_origin[5, :]) / 2
        print('select 3rd top-down grasp')
    
    # use the 2nd component to define the grasping angle
    component = pca.components_[1, :]
    theta = math.atan2(component[0], component[1])
       
    # construct the RT matrix
    RT = euler_matrix(0, np.pi / 2, 0) @ euler_matrix(theta, 0, 0)
    RT[0, 3] = grasp_center[0]
    RT[1, 3] = grasp_center[1]
    
    # deal with noises in z values
    z = np.sort(points_base[:, 2])
    num = len(z)
    # percent = 0.005
    lower = int(num * percent_filter)
    upper = int(num * (1 - percent_filter))
    z_pts = z[lower:upper]

    print("TOP-DOWN STATS-------------------------")
    z_max = np.max(z_pts)
    z_min = np.min(z_pts)
    z_mean = np.mean(z_pts)
    height = z_max - z_min
    z_c = z_min + height/2 # middle point (not always equal to mean)
    print("H, max, min, mean, center", height, z_max, z_min, z_mean, z_c)
    if gripper_finger_length >= height:
        # try to go down the entire object and grasp from bottom
        z_tip = z_min + 0.003 # 1mm offset
    else:
        # NOTE: Two cases to consider here
        # Case1: H/2 > L           ---> ztip = z_c + (H/2 - L) {go a bit more up than center}
        # Case2: H/2 < L but H > L ---> ztip = z_c - (L - H/2) {go a bit more down than center} == z_c + (H/2 - L)
        z_tip = z_c + (height/2 - gripper_finger_length)
    z_tip = max(z_tip, 0.745) # table height check
    print("Z_TIP:", z_tip)
    z_gripper_base = z_tip + gripper_tip_to_base_offset
    RT[2, 3] = z_gripper_base
    return RT, gripper_width    
 

def compute_oriented_bbox(points_base):
    """
    Computes an oriented bounding box for a given set of points
    assumes in robot base frame, z axis is upward, x is away from robot
    and y is right to left when viewed from robot.

    Input:
    - points_base (N,3) : object points in robot's base_link frame

    Returns:
    - Tuple (7, ) : 
      - [x,y,z] location for object center
      - [xlen, ylen, zlen] dimensions of bbox along each axis
      - theta : how much to rotate bbox with  Z-axis as axis of rotation
    """
    xyz_max = (0.95, 0.35, 0.95)
    # Clip point cloud based on workspace constraints
    index_x = (points_base[:, 0] < xyz_max[0])
    index_y = (points_base[:,1] < xyz_max[1])
    index_z = (points_base[:,2] < xyz_max[2])
    mask  = np.logical_and(index_x, index_y, index_z)
    # print(f"OLD shape: {points_base.shape}")
    # print(f"OLD MAX: {np.max(points_base, axis=0)} | MIN: {np.min(points_base, axis=0)}")
    points_base = points_base[mask]
    # print(f"NEW MAX: {np.max(points_base, axis=0)} | MIN: {np.min(points_base, axis=0)}")
    # print(f"NEW shape: {points_base.shape}\n")

    _pts = points_base.copy() 
    percent_filter = 0.05
    for _i in range(3):
        # deal with noises in z values
        _pts_axis = np.sort(_pts[:, _i])
        num = len(_pts_axis)
        # percent = 0.005
        lower = int(num * percent_filter)
        upper = int(num * (1 - percent_filter))
        _pts = _pts[lower:upper]
    points_base = _pts

    height = np.max(points_base[:, 2]) - np.min(points_base[:, 2])
    xy = points_base[:, :2]
    center = np.mean(xy, axis=0).reshape((1, 2))
    xy_centered = xy - center
    pca = PCA(n_components=2)
    xy_new = pca.fit_transform(xy_centered)
    max_coords = np.max(xy_new, axis=0)
    min_coords = np.min(xy_new, axis=0)
    # 4 corners: (minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)
    # these corners are in pca transformed space
    corners = np.zeros((4, 2), dtype=np.float32)
    corners[0, :] = (min_coords[0], min_coords[1])
    corners[1, :] = (min_coords[0], max_coords[1])
    corners[2, :] = (max_coords[0], max_coords[1])
    corners[3, :] = (max_coords[0], min_coords[1])
    corners_org = pca.inverse_transform(corners)
    corners_org += center # get the corners in input pts reference frame
    rect_ylen = np.linalg.norm(corners_org[3] - corners_org[2])
    rect_xlen = np.linalg.norm(corners_org[3] - corners_org[0])
    # use the 2nd component to define the grasping angle
    component = pca.components_[1, :]
    theta = np.pi - math.atan2(component[0], component[1])
    # print(f"HEIGHT: {height}")
    # height = min(height, np.max(points_base[:, 2]) - np.mean(points_base[:, 2]))
    print(f"HEIGHT: {height}\n")
    # return np.mean(points_base, axis=0), rect_xlen, rect_ylen, height, theta
    _cent = (np.max(points_base, axis=0) + np.min(points_base, axis=0))/2.0
    return _cent, rect_xlen, rect_ylen, height, theta


############### Misc and Grasp File I/O UTILS #####################


def user_confirmation(message):
    print(message)
    val = input("Proceed? Y/N: ")
    if val == "N" or val == "n":
        return False
    else:
        return True


def get_object_name(ycb_id: str):
    """Allowed Subset of YCB Objects
    "003" : "003_cracker_box",
    "004": "004_sugar_box",
    "005": "005_tomato_soup_can",
    "006": "006_mustard_bottle",
    "007": "007_tuna_fish_can",
    "008": "008_pudding_box",
    "009": "009_gelatin_box",
    "010": "010_potted_meat_can",
    "011": "011_banana",
    "021": "021_bleach_cleanser",
    "024": "024_bowl",
    "025": "025_mug",
    "035": "035_power_drill",
    "037": "037_scissors",
    "040": "040_large_marker",
    "052": "052_extra_large_clamp",
    """
    if ycb_id == "003":
        return "003_cracker_box"
    elif ycb_id == "004":
        return "004_sugar_box"
    elif ycb_id == "005":
        return "005_tomato_soup_can"
    elif ycb_id == "006":
        return "006_mustard_bottle"
    elif ycb_id == "007":
        return "007_tuna_fish_can"
    elif ycb_id == "008":
        return "008_pudding_box"
    elif ycb_id == "009":
        return "009_gelatin_box"
    elif ycb_id == "010":
        return "010_potted_meat_can"
    elif ycb_id == "011":
        return "011_banana"
    elif ycb_id == "021":
        return "021_bleach_cleanser"
    elif ycb_id == "024":
        return "024_bowl"
    elif ycb_id == "025":
        return "025_mug"
    elif ycb_id == "035":
        return "035_power_drill"
    elif ycb_id == "037":
        return "037_scissors"
    elif ycb_id == "040":
        return "040_large_marker"
    elif ycb_id == "052":
        return "052_extra_large_clamp"
    else:
        return None


def parse_grasps(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    grasps = data["grasps"]

    n = len(grasps)
    poses_grasp = np.zeros((n, 4, 4), dtype=np.float32)
    for i in range(n):
        pose = grasps[i]["pose"]
        rot = pose[3:]
        trans = pose[:3]
        RT = ros_qt_to_rt(rot, trans)
        poses_grasp[i, :, :] = RT
    return poses_grasp


def close_grasps(RT_obj, RT_grasps, close_idxs):
    # translate all RT graspits grasps using the object mean
    # transform grasps to robot base
    # n = RT_grasps.shape[0]
    n = RT_grasps.shape[0]
    RT_grasps_base = np.zeros_like(RT_grasps)
    for i in range(n):
        RT_g = RT_grasps[i]  # RT_grasps[i]
        # transform grasp to robot base
        RT = RT_obj @ RT_g
        RT_grasps_base[i] = RT
    print(close_idxs)
    RT_grasps_base = RT_grasps_base[close_idxs]
    return RT_grasps_base, close_idxs


def sort_grasps(RT_obj, RT_gripper, RT_grasps):
    # translate all RT graspits grasps using the object mean
    # transform grasps to robot base
    n = RT_grasps.shape[0]
    RT_grasps_base = np.zeros_like(RT_grasps)
    distances = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        RT_g = RT_grasps[i]
        # transform grasp to robot base
        RT = RT_obj @ RT_g
        RT_grasps_base[i] = RT
        d = np.linalg.norm(RT_gripper[:3, 3] - RT[:3, 3])
        distances[i] = d

    index = np.argsort(distances)
    RT_grasps_base = RT_grasps_base[index]
    # print("Distances to gripper:", distances)
    # print("Index:", index)
    return RT_grasps_base, index


def sort_and_filter_grasps(RT_obj, RT_gripper, RT_grasps, table_height: float):
    # translate all RT graspits grasps using the object mean
    # transform grasps to robot base
    n = RT_grasps.shape[0]
    # RT_grasps_base = np.zeros_like(RT_grasps)
    distances = np.zeros((n,), dtype=np.float32)
    RT_grasps_base = []
    distances = []
    for i in range(n):
        RT_g = RT_grasps[i]
        # transform grasp to robot base
        RT = RT_obj @ RT_g
        trans = RT[:3, 3]
        if trans[-1] > (table_height + 0.02):  # 2cm offset above table surface
            RT_grasps_base.append(RT)
            d = np.linalg.norm(RT_gripper[:3, 3] - RT[:3, 3])
            distances.append(d)
    final_grasp_len = len(RT_grasps_base)
    pruned_ratio = (n - final_grasp_len) / n
    print(f"Filter ratio: {pruned_ratio}")
    if pruned_ratio == 1.0:
        print(f"returning all none")
        return None, None, None
    RT_grasps_base = np.asarray(RT_grasps_base)
    distances = np.asarray(distances, dtype=np.float32)
    index = np.argsort(distances)
    RT_grasps_base = RT_grasps_base[index]
    return RT_grasps_base, index, pruned_ratio


def model_free_sort_and_filter_grasps(RT_grasps, table_height: float, RT_cam=None):
    """
    Model free version. Doesn't actually use the RT_gripper, just filters!
    Assumes that the RT_grasps are already in the base frame!
    """
    n = len(RT_grasps)
    if n == 0:
        return None, None, None
    RT_grasps_base = []
    if RT_cam is not None:
        RT_tf = RT_cam
    else:
        RT_tf = np.eye(4)
    for i in range(n):
        RT = RT_tf @ RT_grasps[i]
        # RT = RT_grasps[i] # RT_obj @ RT_g
        trans = RT[:3, 3]
        if trans[-1] > (table_height + 0.02):  # 2cm offset above table surface
            RT_grasps_base.append(RT)
    final_grasp_len = len(RT_grasps_base)
    pruned_ratio = (n - final_grasp_len) / n
    print(f"Filter ratio: {pruned_ratio}")
    if pruned_ratio == 1.0:
        print(f"returning all none")
        return None, None, None
    RT_grasps_base = np.asarray(RT_grasps_base)
    # We dont sort by distances, so the index is same
    return RT_grasps_base, [i for i in range(len(RT_grasps_base))], pruned_ratio


def extract_grasps(graspit_grasps, gripper_name, obj_offset):
    # counting
    n = 0
    index = []
    for i in range(len(graspit_grasps)):
        if graspit_grasps[i]["gripper"] == gripper_name:
            n += 1
            index.append(i)

    # get grasps
    poses_grasp = np.zeros((n, 4, 4), dtype=np.float32)
    for i in range(n):
        ind = index[i]
        pose = graspit_grasps[ind]["pose"]
        rot = pose[3:]
        trans = pose[:3]
        RT = ros_qt_to_rt(rot, trans)

        RT_offset = np.eye(4, dtype=np.float32)
        RT_offset[:3, 3] = -obj_offset

        poses_grasp[i, :, :] = RT_offset @ RT
    return poses_grasp
