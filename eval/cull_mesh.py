import os
from frustum_culling import get_grid_culling_pattern
from frustum_culling import get_projection_matrix
from frustum_culling import focal_to_fov
from dataloader_util import load_poses, get_intrinsics
import rendering
import trimesh
import numpy as np


def cull_by_bounds(points, cull_pattern, scene_bounds):
    cull_pattern = np.where(np.all(points > scene_bounds[0], axis=1), cull_pattern, np.zeros_like(cull_pattern))
    cull_pattern = np.where(np.all(points < scene_bounds[1], axis=1), cull_pattern, np.zeros_like(cull_pattern))

    return cull_pattern


def cull_mesh(mesh_path, save_path, pose_file, training_poses,
              intrinsics_path='', far=2.0,
              scene_bounds=None, subdivide=False):

    poses, _ = load_poses(pose_file)
    poses = np.array(poses).astype(np.float32)

    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = mesh.vertices
    triangles = mesh.faces
    if subdivide:
        vertices, triangles = trimesh.remesh.subdivide_to_size(vertices, triangles, max_edge=0.05)

    # Cull with the bounding box first
    cull_pattern = np.ones_like(vertices[..., 0])
    if scene_bounds is not None:
        cull_pattern = cull_by_bounds(vertices, cull_pattern, scene_bounds)

    triangles_in_bounds = []
    for triangle in triangles:
        if cull_pattern[triangle[0]] == 1 or cull_pattern[triangle[1]] == 1 or cull_pattern[triangle[2]] == 1:
            triangles_in_bounds.append(triangle)

    triangles = np.array(triangles_in_bounds)

    # Use a maximum of ~100 poses for performance reasons
    step = max(poses.shape[0] // 100, 1)
    poses = poses[::step, ...]

    # # This part can be simplified if you are not running this on a remote cluster
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    H, W, focal = get_intrinsics(intrinsics_path, 0)
    fov = focal_to_fov(focal, H)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()
    print(fov)
    # depth_maps = rendering.render_depth_maps(mesh, poses, H, W, fov, 10.0)
    depth_maps = rendering.render_depth_maps_doublesided(mesh, poses, H, W, fov, far)

    # Cull faces
    points = vertices[:, :3]
    proj = get_projection_matrix(fov, W / H, near=0.01, far=far)

    print(proj)
    cull_pattern = get_grid_culling_pattern(points, poses, proj, depth_maps, True)

    triangles_in_frustum = []
    for triangle in triangles:
        if cull_pattern[triangle[0]] == 1 or cull_pattern[triangle[1]] == 1 or cull_pattern[triangle[2]] == 1:
            triangles_in_frustum.append(triangle)

    triangles_in_frustum = np.array(triangles_in_frustum)

    mesh = trimesh.Trimesh(vertices, triangles_in_frustum, process=False)
    mesh.remove_unreferenced_vertices()

    mesh.export(save_path)

    return depth_maps


if __name__ == '__main__':
    # Checkpoint path information
    experiments = [
        {
            'basedir': './3d-rgbd-iccv-exp-neural-rgbd-baseline',
            'expname': 'kitchen',
        },
    ]

    # Whiteroom
    # scene_bounds = np.array([[-2.46, -0.1, 0.36],
    #                          [3.06, 3.3, 8.2]])

    # Kitchen
    scene_bounds = np.array([[-3.12, -0.1, -3.18],
                             [3.75, 3.3, 5.45]])

    # Breakfast room
    # scene_bounds = np.array([[-2.23, -0.5, -1.7],
    #                          [1.85, 2.77, 3.0]])

    # Staircase
    # scene_bounds = np.array([[-4.14, -0.1, -5.25],
    #                          [2.52, 3.43, 1.08]])

    # Complete kitchen
    # scene_bounds = np.array([[-5.55, 0.0, -6.45],
    #                          [3.65, 3.1, 3.5]])

    # Green room
    # scene_bounds = np.array([[-2.5, -0.1, 0.4],
    #                          [5.4, 2.8, 5.0]])

    # Grey white room
    # scene_bounds = np.array([[-0.55, -0.1, -3.75],
    #                          [5.3, 3.0, 0.65]])

    # Morning apartment
    # scene_bounds = np.array([[-1.38, -0.1, -2.2],
    #                          [2.1, 2.1, 1.75]])

    pose_file = 'poses.txt'
    training_poses = 'trainval_poses.txt'
    intrinsics_path = os.path.join('../data/kitchen')

    for e in experiments:
        basedir, expname = e.values()
        print(basedir, expname)

        mesh_path = os.path.join(basedir, expname, 'output_meshes', 'mesh_199999_transformed.ply')
        save_path = os.path.join(basedir, expname, 'output_meshes', 'mesh_200000_culled.ply')

        depth_maps = cull_mesh(mesh_path, save_path, os.path.join('../data', expname, pose_file), training_poses, intrinsics_path, scene_bounds=scene_bounds, subdivide=True, far=10)