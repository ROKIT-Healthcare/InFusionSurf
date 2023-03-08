import trimesh
from scipy.spatial import cKDTree
import numpy as np
import os


def compute_iou(mesh_pred, mesh_target):
    res = 0.1
    v_pred = mesh_pred.voxelized(pitch=res)
    v_target = mesh_target.voxelized(pitch=res)
    v_target_mesh = v_target.as_boxes()
    v_pred_mesh = v_pred.as_boxes()
    
    v_pred_pts = np.round(v_pred.points / res).astype(int)
    v_target_pts = np.round(v_target.points / res).astype(int)
    
    v_pred_filled = set(tuple(x) for x in v_pred_pts)
    v_target_filled = set(tuple(x) for x in v_target_pts)
    inter = v_pred_filled.intersection(v_target_filled)
    union = v_pred_filled.union(v_target_filled)
    iou = len(inter) / len(union)
    
    return iou


def compute_metrics(path_pred, path_target):
    mesh_pred = trimesh.load_mesh(path_pred)
    mesh_target = trimesh.load_mesh(path_target)
    iou = compute_iou(mesh_pred, mesh_target)

    pointcloud_pred, idx = mesh_pred.sample(int(mesh_pred.area * 1e4), return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_tgt, idx = mesh_target.sample(int(mesh_pred.area * 1e4), return_index=True)
    pointcloud_tgt = pointcloud_tgt.astype(np.float32)
    normals_tgt = mesh_target.face_normals[idx]

    thresholds = np.linspace(1. / 1000, 1, 1000)

    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud_pred, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy ** 2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]

    return [chamferL1, iou, normals_correctness, F[49], F[14]]


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """ Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    """ Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold