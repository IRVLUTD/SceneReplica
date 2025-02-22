import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
import munkres as munkres


def visualize_segmentation(im, masks, nc=None, return_rgb=False, save_dir=None):
    """Visualize segmentations nicely. Based on code from:
    https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

    @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
    @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., K}
    @param nc: total number of colors. If None, this will be inferred by masks
    """
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    if not return_rgb:
        fig = plt.figure()
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        fig.add_axes(ax)
        ax.imshow(im)

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap("gist_rainbow")
    colors = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]

    if not return_rgb:
        # matplotlib stuff
        fig = plt.figure()
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        fig.add_axes(ax)

    # Mask
    imgMask = np.zeros(im.shape)

    # Draw color masks
    for i in np.unique(masks):
        if i == 0:  # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = 0.4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = masks == i

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)

    # Draw mask contours
    for i in np.unique(masks):
        if i == 0:  # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = 0.4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = masks == i

        # Find contours
        try:
            contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
            )
        except:
            im2, contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
            )

        # Plot the nice outline
        for c in contour:
            if save_dir is None and not return_rgb:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=False,
                    facecolor=color_mask,
                    edgecolor="w",
                    linewidth=1.2,
                    alpha=0.5,
                )
                ax.add_patch(polygon)
            else:
                cv2.drawContours(im, contour, -1, (255, 255, 255), 2)

    if save_dir is None and not return_rgb:
        ax.imshow(im)
        return fig
    elif return_rgb:
        return im
    elif save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image


def process_label_image(label_image, class_colors, cls_indexes):
    """
    Change label image to label index
    Code adapted from: https://github.com/IRVLUTD/posecnn-pytorch/ycb_toolbox/ycb_extract_objects.py
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    labels = np.ones((height, width), dtype=np.int32)
    labels += 100 # SOME DEFAULT VALUE GREATER THAN NUM_CLASSES
    # label image is in BGR order
    index = (
        label_image[:, :, 2]
        + 256 * label_image[:, :, 1]
        + 256 * 256 * label_image[:, :, 0]
    )
    for i in range(len(class_colors)):
        color = class_colors[i]
        ind = color[2] + 256 * color[1] + 256 * 256 * color[0]
        I = np.where(index == ind)
        labels[I[0], I[1]] = cls_indexes[i]
    return labels


# Code adapted from: https://github.com/davisvideochallenge/davis-2017/blob/master/python/lib/davis/measures/f_boundary.py
def seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width     : Width of desired bmap  <= seg.shape[1]
        height  :   Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray): Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    # from IPython import embed; embed()

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def compute_segments_assignment(prediction, gt):
    """
    Computes the matching between prediction and gt segmentation masks
    Adapted from here: https://github.com/IRVLUTD/UnseenObjectClustering/lib/utils/evaluation.py
    @param gt: a [H x W] numpy.ndarray with ground truth masks
    @param prediction: a [H x W] numpy.ndarray with predicted masks

    @return assignments: list of (i,j) tupeles into cost matrx for matching
    """
    BACKGROUND_LABEL = 101
    OBJECTS_LABEL = 1
    # Get unique OBJECT labels from GT and prediction
    labels_gt = np.unique(gt)
    # labels_gt = labels_gt[~np.isin(labels_gt, [BACKGROUND_LABEL])]
    num_labels_gt = labels_gt.shape[0]

    BACKGROUND_LABEL = 0
    labels_pred = np.unique(prediction)
    # labels_pred = labels_pred[~np.isin(labels_pred, [BACKGROUND_LABEL])]
    num_labels_pred = labels_pred.shape[0]
    print(f"Num: Labels: {num_labels_pred} | GT: {num_labels_gt}")

    # F-measure: used for computing the cost matrix for Munkres
    F = np.zeros((num_labels_gt, num_labels_pred))

    # For every pair of GT label vs. predicted label, calculate stuff
    for i, gt_i in enumerate(labels_gt):
        gt_i_mask = (gt == gt_i)
        for j, pred_j in enumerate(labels_pred):
            pred_j_mask = (prediction == pred_j)
            # true positive
            A = np.logical_and(pred_j_mask, gt_i_mask)
            tp = np.int64(np.count_nonzero(A))  # Cast this to numpy.int64 so 0/0 = nan
            # precision
            prec = tp / np.count_nonzero(pred_j_mask)
            # recall
            rec = tp / np.count_nonzero(gt_i_mask)
            # F-measure
            if prec + rec > 0:
                F[i, j] = (2 * prec * rec) / (prec + rec)

    ### Compute the Hungarian assignment ###
    F[np.isnan(F)] = 0

    if num_labels_gt < num_labels_pred:
        pass
    else:
        pass


    m = munkres.Munkres()
    cost = F.max() - F.copy()
    print(f"COST SHAPE: {cost.shape}")
    assignments = m.compute(
        F.max() - F.copy()
    )  # list of (y,x) indices into F (these are the matchings)
    return assignments, labels_pred, labels_gt
