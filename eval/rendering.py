import trimesh
import os
import pyrender
import math
import numpy as np

def render_depth_maps(mesh, poses, H, W, yfov=60.0, far=2.0):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=math.radians(yfov), aspectRatio=W/H, znear=0.01, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(H, W)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for pose in poses:
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)
        depth_maps.append(depth)

    return depth_maps


# For meshes with backward-facing faces. For some reasong the no_culling flag in pyrender doesn't work for depth maps
def render_depth_maps_doublesided(mesh, poses, H, W, yfov=60.0, far=2.0):
    depth_maps_1 = render_depth_maps(mesh, poses, H, W, yfov, far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    depth_maps_2 = render_depth_maps(mesh, poses, H, W, yfov, far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]] # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in range(len(depth_maps_1)):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where((depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map), depth_maps_2[i], depth_map)
        depth_maps.append(depth_map)

    return depth_maps

if __name__ == '__main__':
    with open('../data/grey_white_room/poses.txt', 'r') as f:
        poses = f.readlines()
    
    poses = [[float(ele) for ele in pose.strip().split()] for pose in poses]
    poses = np.array(poses)
    poses = poses.reshape(-1, 4, 4)
    mesh = trimesh.load_mesh('../../iccv_rokit-pjt-mlops-2022-03-21/gt/grey_white_room/gt_mesh.ply')
    
    # pose = np.array([[0.998135, 0.031898, -0.052051, -0.304830],
    #                   [0.000000, 0.852638, 0.522494, 1.367226],
    #                   [0.061048, -0.521527, 0.851053, 1.462762],
    #                   [0.000000, 0.000000, 0.000000, 1.000000]])
    
    depth_map = render_depth_maps(mesh, poses[:1], 480, 640, yfov=46.82644889274108, far=10)
