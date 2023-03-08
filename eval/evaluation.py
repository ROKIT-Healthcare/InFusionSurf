import os
import sys
sys.path.append('./')

import torch
import torch.nn as nn
import trimesh
import numpy as np

import parser_util
from cull_mesh import cull_mesh
from mesh_metrics import compute_metrics
from dataloader_util import load_poses, load_first_pose
from optimization import PoseArray

def align_and_cull_mesh(args):
    datadir = args.datadir
    basedir = args.basedir
    expname = args.expname

    root = os.path.join(basedir, expname)
    
    for extckpt in args.extckpt:
        transformed_path = os.path.join(root, 'mesh', '{:06d}_transformed.ply'.format(extckpt))

        state_dict = torch.load(os.path.join(root, '{:06d}.tar'.format(extckpt)), map_location=torch.device('cpu'))
        pose_array_state_dict = state_dict['pose_array_state_dict']

        # get first pose 
        gt_pose = load_first_pose(datadir, 'poses.txt')
        trainval_pose = load_first_pose(datadir, 'trainval_poses.txt')

        translation, sc_factor = args.translation, args.sc_factor

        # prepare matrices
        gl_to_scannet = np.array([[1, 0, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 0, 1]]).astype(np.float32).reshape([4, 4])
        
        sc_tr_mat = np.eye(4)
        sc_tr_mat[:3, :3] /= sc_factor
        sc_tr_mat[:3, 3] -= translation
        
        trainval_pose[:3, 3] += translation
        trainval_pose[:3, 3] *= sc_factor
        
        pose_array = PoseArray(1)
        pose_array.data = nn.Parameter(pose_array_state_dict['data'][:1])

        R = pose_array.get_rotation_matrices(torch.tensor([0])).detach().numpy()
        t = pose_array.get_translations(torch.tensor([0])).detach().numpy()

        pose_mat = np.concatenate([R, t[..., None]], axis=-1)[0]
        pose_mat = np.concatenate([pose_mat, [[0, 0, 0, 1]]], axis=0)
        
        sc_mat = np.eye(4) * sc_factor
        
        del state_dict
        
        # align mesh
        align_mat = gl_to_scannet @ sc_tr_mat @ pose_mat @ trainval_pose @ sc_mat
        
        mesh = trimesh.load(os.path.join(root, 'mesh', '{:06d}.ply'.format(extckpt)), force='mesh', process=False)
        mesh = mesh.apply_transform(np.linalg.inv(align_mat))

        mesh = mesh.apply_transform(gt_pose)

        mesh.export(transformed_path)

        print(f"saved in {transformed_path}")
        
        print(transformed_path)
        save_path = os.path.join(root, 'mesh', '{:06d}_culled.ply'.format(extckpt))

        depth_maps = cull_mesh(transformed_path, 
                               save_path, 
                               os.path.join(datadir, 'poses.txt'), 
                               'trainval_poses.txt', 
                               datadir, 
                               scene_bounds=None, 
                               far=args.far / sc_factor, 
                               subdivide=True)

if __name__ == '__main__':
    parser = parser_util.get_parser()
    parser.add_argument("--extckpt", action="append", default=[], type=int, help='mesh extraction checkpoints')
    parser.add_argument("--gtdir", type=str, help='ground truth mesh dir')
    
    args = parser.parse_args()
    
    align_and_cull_mesh(args)
    
    gtdir = args.gtdir
    datadir = args.datadir
    basedir = args.basedir
    expname = args.expname
    for extckpt in args.extckpt:
        result = []
        
        print(f"evaluating {expname} ...")
        pred_path = os.path.join(basedir, expname, 'mesh', '{:06d}_culled.ply'.format(extckpt))

        metrics = compute_metrics(pred_path, gtdir)

        scene_result = {'scene_name': expname, 
                        'iteration': extckpt, 
                        'chamferL1': metrics[0],
                        'IoU': metrics[1],
                        'Normal correctness': metrics[2],
                        'F-score': metrics[3]}

        print(f"result: {scene_result}")