import os
import numpy as np
import re
import cv2
import imageio

def load_poses(posefile):
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if 'nan' in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
            poses.append(pose_floats)

    return poses, valid


def load_first_pose(datadir, pose_file):    
    with open(os.path.join(datadir, pose_file), 'r') as f:
        pose = [list(map(float, f.readline().split())) for _ in range(4)]
    
    return np.array(pose)


def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]


def resize_images(images, H, W, interpolation=cv2.INTER_LINEAR):
    resized = np.zeros((images.shape[0], H, W, images.shape[3]), dtype=images.dtype)
    for i, img in enumerate(images):
        r = cv2.resize(img, (W, H), interpolation=interpolation)
        if images.shape[3] == 1:
            r = r[..., np.newaxis]
        resized[i] = r
    return resized


def get_intrinsics(basedir, crop):
    depth = imageio.imread(os.path.join(basedir, 'depth_filtered', 'depth0.png'))
    H, W = depth.shape[:2]
    H = H - crop / 2
    W = W - crop / 2
    focal = load_focal_length(os.path.join(basedir, 'focal.txt'))

    return H, W, focal