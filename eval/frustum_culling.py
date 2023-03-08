import numpy as np
import math


def focal_to_fov(focal, height):
    return math.degrees(2.0 * math.atan(height / (2.0 * focal)))


def fov_to_focal(fov, height):
    return height / (2.0 * math.tan(math.radians(0.5 * fov)))


def get_projection_matrix(fov, aspect, near, far):
    f = 1.0 / math.tan(0.5 * math.radians(fov))

    return np.array([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
        [0.0, 0.0, -1.0, 0.0]
    ])


def cull_from_one_pose(points, pose, projection_matrix, depth=None):
    rotation = np.transpose(pose[:3, :3])
    translation = -pose[:3, 3:4]

    camera_space = rotation[np.newaxis, :, :] @ (points[..., np.newaxis] + translation[np.newaxis, :, :])

    ones = np.ones([camera_space.shape[0], 1, 1], dtype=np.float32)
    camera_space = np.concatenate([camera_space, ones], 1)

    clip_space = projection_matrix[np.newaxis, :, :] @ camera_space
    clip_space = clip_space[..., 0]
    w = clip_space[..., 3]

    cull_pattern = np.ones(points.shape[0], dtype=np.float32)
    cull_pattern = np.where(clip_space[:, 0] > w, np.zeros_like(cull_pattern), cull_pattern)
    cull_pattern = np.where(clip_space[:, 0] < -w, np.zeros_like(cull_pattern), cull_pattern)
    cull_pattern = np.where(clip_space[:, 1] > w, np.zeros_like(cull_pattern), cull_pattern)
    cull_pattern = np.where(clip_space[:, 1] < -w, np.zeros_like(cull_pattern), cull_pattern)
    cull_pattern = np.where(clip_space[:, 2] > w, np.zeros_like(cull_pattern), cull_pattern)
    cull_pattern = np.where(clip_space[:, 2] < -w, np.zeros_like(cull_pattern), cull_pattern)

    if depth is not None:
        H, W = depth.shape
        pos_x = ((clip_space[:, 0] / w + 1.0) * 0.5 * (W - 1)).astype(np.int32)
        pos_x = np.clip(pos_x, 0, W - 1)
        pos_y = ((-clip_space[:, 1] / w + 1.0) * 0.5 * (H - 1)).astype(np.int32)
        pos_y = np.clip(pos_y, 0, H - 1)

        sampled_depth = depth[pos_y, pos_x]

        eps = 1e-2
        cull_pattern = np.where(w > (sampled_depth + eps), np.zeros_like(cull_pattern), cull_pattern)
        # cull_pattern = np.where((w > (sampled_depth + eps)) & (sampled_depth > 0), np.zeros_like(cull_pattern), cull_pattern)

    return cull_pattern


def get_grid_culling_pattern(points, poses, projection_matrix, depth_maps=None, verbose=False):

    cull_pattern = np.ones_like(points[..., 0])

    for i, pose in enumerate(poses):
        if verbose:
            print('Processing pose ' + str(i + 1) + ' out of ' + str(poses.shape[0]))

        depth = None
        if depth_maps is not None:
            depth = depth_maps[i]
        cull_pattern = cull_pattern * (1.0 - cull_from_one_pose(points, pose, projection_matrix, depth))

    return 1.0 - cull_pattern

