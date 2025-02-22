import os
import scipy.io as sio
import numpy as np
import argparse
import trimesh
from utils_pose_estimation import *
import scipy
import matplotlib.pyplot as plt
import copy
# algo:
    # if in realworld, no pose found, perception failure 100%
    # if in realworld, its able to grasp and lift, not a perception failure


_classes_all = [
        '__background__',
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
NUM_OBJECTS = len(_classes_all)

def get_thresholds_points(model_dir, model_names):
    """
        returns threshold of each object. Threshold = 0.1 * object diameter
    """
    model_extents = {modelname: None for modelname in model_names}
    threshold = {modelname: None for modelname in model_names}
    points_all = {modelname: None for modelname in model_names}
    for modelname in model_names:
        modelfile = os.path.join(model_dir, modelname, 'textured_simple.obj')
        mesh = trimesh.load_mesh(modelfile)
        points = mesh.vertices

        extents = 2*np.max(np.absolute(points),axis=0)
        model_extents[modelname] = extents
        threshold[modelname] = 0.25*np.max(extents)
        points_all[modelname] = points

    print(f"model extents {model_extents}")
    print(f"threshold {threshold}")

    return threshold, points_all

def read_mat_file(filename):
    contents = sio.loadmat(filename)
    return contents

def extract_experiment_metadata(filename, gt=False):
    filename_pattern = re.compile(r'.*method-(?P<method>\w+)_scene-(?P<scene>\d+)_ord-(?P<order>\w+)$')
    exp_data = filename_pattern.match(filename)
    return exp_data.group

def make_args():
    parser = argparse.ArgumentParser(
        description="pose analysis"
    )

    parser.add_argument(
        "-e",
        "--exp_dir",
        default="./gdrnpp",
        help="evaluation directory like poserbpf, posecnn etc..,",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="/home/benchmark/Datasets/benchmarking/",
        help="Path to parent of model dataset, grasp and scenes dir",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = make_args()
    exp_files = os.listdir(args.exp_dir)  # these are not sorted now
    exp_files = sort_files_by_scene_number(exp_files)
    model_names = _classes_all[1:]
    model_dir = os.path.join(args.data_dir, 'models')
    threshold, points_all = get_thresholds_points(model_dir, model_names)

    # read the results from the exp_files
    head_tail = os.path.split(args.exp_dir)
    model = head_tail[1]
    print('model to be evaluated:', model)
    output_dir = './results_' + model
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    filename = os.path.join(output_dir, f'results{model}.mat')
    if os.path.exists(filename):
        results_all = scipy.io.loadmat(filename)
        print('load results from file', filename)
        distances_sys = results_all['distances_sys']
        distances_non = results_all['distances_non']
        errors_rotation = results_all['errors_rotation']
        errors_translation = results_all['errors_translation']
        results_cls_id = results_all['results_cls_id'].flatten()
        print(len(distances_sys))
    else:
        # save results
        num_max = 100000
        num_results = 1
        distances_sys = np.zeros((num_max, num_results), dtype=np.float32)
        distances_non = np.zeros((num_max, num_results), dtype=np.float32)
        errors_rotation = np.zeros((num_max, num_results), dtype=np.float32)
        errors_translation = np.zeros((num_max, num_results), dtype=np.float32)
        results_cls_id = np.zeros((num_max,), dtype=np.float32)
        count = -1

        # loop through 2 orders
        for _order in ["nearest_first", "random"]:
            print(f'==============order: {_order}=============')
            for file in exp_files:
                # filename = 'poserbpf/23-05-28_T193243_method-poserbpf_scene-27_ord-nearest_first'
                exp_data = extract_experiment_metadata(filename=file)
                order = exp_data("order")
                if order != _order:
                    continue
                print(file)

                # read ground truth, get the order
                gt_file = os.path.join(args.data_dir, "final_scenes/metadata", "meta-%06d.mat"%int(exp_data("scene")))
                gt_data = read_mat_file(gt_file)
                gt_objects = [obj.strip() for obj in gt_data["object_names"]]
                gt_poses = gt_data["poses"]

                gt_order = str(gt_data[exp_data("order")][0])     # order in object names
                gt_order = gt_order.split(",")
                prev_objs = set()
                set_gt_order_dec = set(gt_order)
                # loop through each object
                print(f"+++++++--------------------------SCENE  {exp_data('scene')}--------------------------+++++++++")
                for seq, _object in enumerate(gt_order):
                    print(f"+++-----------ACTUAL OBJECT Grasped: {_object}--------------+++")
                    # read estimated poses
                    est_data = read_pickle_file(os.path.join(args.exp_dir, file, f"gt_exp_data_{seq}.pk"))
                    # gt_order_dec = set(gt_order) - prev_objs
                    set_gt_order_dec.difference_update(prev_objs)
                    prev_objs.add(_object)
                    for object in set_gt_order_dec:
                        print(f"object {object}")
                        # to be checked
                        count += 1
                        # get the ground truth pose
                        object_gt_pose = gt_poses[gt_objects.index(object)]
                        object_gt_pose = convert_standard_to_rosqt(object_gt_pose)
                        object_gt_pose = ros_qt_to_rt( object_gt_pose[3:],object_gt_pose[:3]) # get object pose as 4x4
                        # get the object index in _classes_all
                        cls_index = _classes_all.index(object)
                        results_cls_id[count] = cls_index
                        est_pose = est_data['estimated_poses'][object]

                        #transofrm the points through gt pose and estimated pose
                        points = points_all[object]
                        points_gt = trimesh.transform_points(points.copy(), object_gt_pose)
                        points_est = trimesh.transform_points(points.copy(), est_pose)

                        if est_pose is None:
                            print(f"pose of {object} not detected! 100% perception failure")
                            distances_sys[count, :] = np.inf
                            distances_non[count, :] = np.inf
                            errors_rotation[count, :] = np.inf
                            errors_translation[count, :] = np.inf
                            continue

                        # for symmetric objects
                        adds = ADDS(points_gt, points_est)
                        distances_sys[count, 0] = adds
                        # for non symmetric objects
                        add = ADD(points_gt, points_est)
                        distances_non[count, 0] = add
                        # from list to np array
                        RT = np.array(est_pose).astype(np.float32)
                        RT_gt = np.array(object_gt_pose).astype(np.float32)
                        errors_rotation[count, 0] = rotation_error(RT[:3, :3], RT_gt[:3, :3])
                        errors_translation[count, 0] = trans_error(RT[:3, 3], RT_gt[:3, 3])
                        # if adds > threshold[object]:
                        # if adds > 0.025:   #absolute threshold
                        #     print(f"perception failure")
                        #     print(f"object: {object} ADDS: {adds} method: {exp_data('method')} order: {exp_data('order')} scene: {exp_data('scene')}\n")
                        #     print(f"translation error {errors_translation[count,0]}")
                        #     print(f"rotation error {errors_rotation[count,0]}")
                        #     import sys
                        #     sys.exit()

        distances_sys = distances_sys[:count + 1, :]
        distances_non = distances_non[:count + 1, :]
        errors_rotation = errors_rotation[:count + 1, :]
        errors_translation = errors_translation[:count + 1, :]
        results_cls_id = results_cls_id[:count + 1]

        results_all = {'distances_sys': distances_sys,
                    'distances_non': distances_non,
                    'errors_rotation': errors_rotation,
                    'errors_translation': errors_translation,
                    'results_cls_id': results_cls_id}

        filename = os.path.join(output_dir, f'results_{model}.mat')
        scipy.io.savemat(filename, results_all)
        print(f"+++++++--------------------------END OF ORDER: {_order}--------------------------+++++++++\n")

    # plot results
    max_distance = 0.1
    index_plot = [0]
    color = ['g']
    leng = [model]
    num = len(leng)
    ADD = np.zeros((NUM_OBJECTS, num), dtype=np.float32)
    ADDS = np.zeros((NUM_OBJECTS, num), dtype=np.float32)
    TS = np.zeros((NUM_OBJECTS, num), dtype=np.float32)
    classes = list(copy.copy(_classes_all))
    classes[0] = 'all'
    for k in range(NUM_OBJECTS):
        fig = plt.figure(figsize=(16, 12))
        if k == 0:
            index = range(len(results_cls_id))
        else:
            index = np.where(results_cls_id == k)[0]

        if len(index) == 0:
            continue
        print('%s: %d objects' % (classes[k], len(index)))

        # distance symmetry
        ax = fig.add_subplot(2, 3, 1)
        lengs = []
        for i in index_plot:
            D = distances_sys[index, i]
            ind = np.where(D > max_distance)[0]
            D[ind] = np.inf
            d = np.sort(D)
            n = len(d)
            accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
            plt.plot(d, accuracy, color[i], linewidth=2)
            ADDS[k, i] = VOCap(d, accuracy)
            lengs.append('%s (%.2f)' % (leng[i], ADDS[k, i] * 100))
            print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

        ax.legend(lengs)
        plt.xlabel('Average distance threshold in meter (symmetry)')
        plt.ylabel('accuracy')
        ax.set_title(classes[k])

        # distance non-symmetry
        ax = fig.add_subplot(2, 3, 2)
        lengs = []
        for i in index_plot:
            D = distances_non[index, i]
            ind = np.where(D > max_distance)[0]
            D[ind] = np.inf
            d = np.sort(D)
            n = len(d)
            accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
            plt.plot(d, accuracy, color[i], linewidth=2)
            ADD[k, i] = VOCap(d, accuracy)
            lengs.append('%s (%.2f)' % (leng[i], ADD[k, i] * 100))
            print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

        ax.legend(lengs)
        plt.xlabel('Average distance threshold in meter (non-symmetry)')
        plt.ylabel('accuracy')
        ax.set_title(classes[k])

        # translation
        ax = fig.add_subplot(2, 3, 3)
        lengs = []
        for i in index_plot:
            D = errors_translation[index, i]
            ind = np.where(D > max_distance)[0]
            D[ind] = np.inf
            d = np.sort(D)
            n = len(d)
            accuracy = np.cumsum(np.ones((n, ), np.float32)) / n
            plt.plot(d, accuracy, color[i], linewidth=2)
            TS[k, i] = VOCap(d, accuracy)
            lengs.append('%s (%.2f)' % (leng[i], TS[k, i] * 100))
            print('%s, %s: %d objects missed' % (classes[k], leng[i], np.sum(np.isinf(D))))

        ax.legend(lengs)
        plt.xlabel('Translation threshold in meter')
        plt.ylabel('accuracy')
        ax.set_title(classes[k])

        # rotation histogram
        count = 4
        for i in index_plot:
            ax = fig.add_subplot(2, 3, count)
            D = errors_rotation[index, i]
            ind = np.where(np.isfinite(D))[0]
            D = D[ind]
            ax.hist(D, bins=range(0, 190, 10), range=(0, 180))
            plt.xlabel('Rotation angle error')
            plt.ylabel('count')
            ax.set_title(leng[i])
            count += 1

        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        filename = output_dir + '/' + classes[k] + '.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        # plt.show()

    # print ADD
    print('==================ADD======================')
    for k in range(len(classes)):
        print('%s: %f' % (classes[k], ADD[k, 0]))
    for k in range(len(classes ) -1):
        print('%f' % (ADD[k +1, 0]))
    print('%f' % (ADD[0, 0]))
    print('===========================================')

    # print ADD-S
    print('==================ADD-S====================')
    for k in range(len(classes)):
        print('%s: %f' % (classes[k], ADDS[k, 0]))
    for k in range(len(classes ) -1):
        print('%f' % (ADDS[k +1, 0]))
    print('%f' % (ADDS[0, 0]))
    print('===========================================')