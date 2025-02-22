import math
import numpy as np
import re
import pickle
from scipy import spatial
from transforms3d.quaternions import quat2mat

def trans_error(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error

def rotation_error(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose (3x1 vector).
    :param R_gt: Rotational element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi # [rad] -> [deg]
    return error


def ADDS(points_gt, points_est):
    """
        takes in the points transformed into ground truth and estimated poses
    """
    nn_index = spatial.cKDTree(points_est)
    nn_dists, _ = nn_index.query(points_gt, k=1)
    e=nn_dists.mean()
    return e

def ADD(points_gt, points_est):
    """
        Average Distance of Model Points for objects with no indistinguishable views
        - by Hinterstoisser et al. (ACCV 2012).

        :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
        :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
        :param model: Object model given by a dictionary where item 'pts'
        is nx3 ndarray with 3D model points.
        :return: Error of pose_est w.r.t. pose_gt.
        """
    e = np.linalg.norm(points_est - points_gt, axis=1).mean()
    return e


def VOCap(rec, prec):
    index = np.where(np.isfinite(rec))[0]
    rec = rec[index]
    prec = prec[index]
    if len(rec) == 0 or len(prec) == 0:
        ap = 0
    else:
        mrec = np.insert(rec, 0, 0)
        mrec = np.append(mrec, 0.1)
        mpre = np.insert(prec, 0, 0)
        mpre = np.append(mpre, prec[-1])
        for i in range(1, len(mpre)):
            mpre[i] = max(mpre[i], mpre[i-1])
        i = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = np.sum(np.multiply(mrec[i] - mrec[i-1], mpre[i])) * 10
    return ap


def read_pickle_file(filename):
    with open(filename, "rb") as handle:
        contents = pickle.load(handle)
    handle.close()
    return contents

def convert_standard_to_rosqt(pose_s):
    """Converts (posn, w,x,y,z) quat to ROS format (posn, x,y,z,w) quat"""
    posn = pose_s[:3]
    q_s = pose_s[3:]
    quat = [q_s[1], q_s[2], q_s[3], q_s[0]]
    # posn.extend(quat)
    return [*posn, *quat]

def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


def sort_files_by_scene_number(files):
    # Make a copy of the list to avoid modifying the original list
    files = files.copy()

    # Regular expression to match the scene number in the file names
    scene_number_regex = re.compile(r"_scene-(\d+)_")

    def scene_number(file_name):
        match = scene_number_regex.search(file_name)
        return int(match.group(1)) if match else float('inf')

    # Sort the file names based on the scene number
    files.sort(key=scene_number)

    return files