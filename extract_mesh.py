# import load_network_model
import os
import ast
import scene_bounds

import numpy as np
import torch

import marching_cubes as mcubes
import trimesh

from torch.cuda.amp import autocast as autocast

import parser_util

from optimize import run_network
from optimization import FeatureArray, DeformationField, PoseArray, PerFrameAlignment
from nerf_helpers import *


def get_batch_query_fn(query_fn, feature_array, network_fn):

    fn = lambda f, i0, i1: query_fn(f[i0:i1, None, :], viewdirs=torch.zeros_like(f[i0:i1]),
                                    feature_array=feature_array,
                                    pose_array=None,
                                    frame_ids=torch.zeros_like(f[i0:i1, 0], dtype=torch.int32),
                                    per_frame_alignment=None,
                                    deformation_field=None,
                                    c2w_array=None,
                                    network_fn=network_fn)

    return fn


def extract_mesh(query_fn, feature_array, network_fn, args, voxel_size=0.01, isolevel=0.0, scene_name='', mesh_savepath=''):
    # Query network on dense 3d grid of points
    voxel_size *= args.sc_factor  # in "network space"
    tx, ty, tz = scene_bounds.get_scene_bounds(scene_name, voxel_size, args.sc_factor, True)

    query_pts = torch.stack(torch.meshgrid(tx, ty, tz, indexing='ij'), -1).to(torch.float32)
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])

    fn = get_batch_query_fn(query_fn, feature_array, network_fn)

    chunk = 1024 * 64
    with autocast():
        raw = [fn(flat, i, i + chunk)[0].cpu().data.numpy() for i in range(0, flat.shape[0], chunk)]
    
    raw = np.concatenate(raw, 0).astype(np.float32)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    sigma = raw[..., -1]

    print('Running Marching Cubes')
    vertices, triangles = mcubes.marching_cubes(sigma, isolevel, truncation=3.0)
    print('done', vertices.shape, triangles.shape)

    # normalize vertex positions
    vertices[:, :3] /= np.array([[tx.shape[0] - 1, ty.shape[0] - 1, tz.shape[0] - 1]])

    # Rescale and translate
    tx = tx.cpu().data.numpy()
    ty = ty.cpu().data.numpy()
    tz = tz.cpu().data.numpy()
    
    scale = np.array([tx[-1] - tx[0], ty[-1] - ty[0], tz[-1] - tz[0]])
    offset = np.array([tx[0], ty[0], tz[0]])
    vertices[:, :3] = scale[np.newaxis, :] * vertices[:, :3] + offset

    # Transform to metric units
    vertices[:, :3] = vertices[:, :3] / args.sc_factor - args.translation

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    # Transform the mesh to Scannet's coordinate system
    gl_to_scannet = np.array([[1, 0, 0, 0],
                              [0, 0, -1, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 1]]).astype(np.float32).reshape([4, 4])

    mesh.apply_transform(gl_to_scannet)

    if mesh_savepath == '':
        mesh_savepath = os.path.join(args.basedir, args.expname, f"mesh_vs{voxel_size / args.sc_factor.ply}")
    mesh.export(mesh_savepath)

    print('Mesh saved')

if __name__ == '__main__':    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = parser_util.get_parser()    
    parser.add_argument("--extckpt", action="append", default=[], type=int,
                        help='mesh extraction checkpoints')
    parser.add_argument("--ext_voxel_size", type=float,
                        default=0.01, help='mesh extraction voxel size')
    
    args = parser.parse_args()    
    
    outdir = os.path.join(args.basedir, args.expname, 'mesh')
    
    os.makedirs(outdir, exist_ok=True)
    
    scene_name = args.expname

    print(args)
    
    def create_test_model(args, load_ckpt_iter=-1):
        ckptdir = os.path.join(args.basedir, args.expname)
        
        if load_ckpt_iter == -1:
            ckpts = [os.path.join(ckptdir, f) for f in sorted(os.listdir(ckptdir)) if ".tar" in f]
            ckpts = ckpts[-1:]
        else:
            ckpts = [os.path.join(ckptdir, f) for f in sorted(os.listdir(ckptdir)) if f"{load_ckpt_iter}.tar" in f]
            
        print('Found ckpts', ckpts)
        if len(ckpts) > 0:
            ckpt_path = ckpts[0]
        else:
            raise IndexError(f"No checkpoint on iteration {load_ckpt_iter} found")
            
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        network_fn_state_dict = ckpt['network_fn_state_dict']
        
        models = {}
        model = FastSurf(network_fn_state_dict['xyz_min'], network_fn_state_dict['xyz_max'],
                         voxel_size=network_fn_state_dict['voxel_size'][0], feature_dim=args.dense_features, 
                         D=args.netdepth, W=args.netwidth, 
                         viewbase_pe=args.multires_views, i_view_embed=args.i_embed,
                         n_frame_features=args.frame_features).to(device)
                
        models['model'] = model

        print(model.eval())

        # Create feature array
        feature_array = None
        if args.frame_features > 0:
            feature_array = FeatureArray(1, args.frame_features)
            models['feature_array'] = feature_array

        # Create pose array
        pose_array = None

        # Create deformation field
        deformation_field = None
        
        per_frame_alignment = None

        def network_query_fn(inputs, viewdirs, feature_array, pose_array, frame_ids, per_frame_alignment, deformation_field, c2w_array, network_fn):
            return run_network(
                inputs, viewdirs, feature_array, pose_array, frame_ids, per_frame_alignment, deformation_field, c2w_array, network_fn,
                netchunk=args.netchunk)
        
        iteration = ckpt['global_step']
        
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

        if feature_array is not None:
            ckpt['feature_array_state_dict']['data'] = ckpt['feature_array_state_dict']['data'][:1]

            feature_array.load_state_dict(ckpt['feature_array_state_dict'])
        
        render_kwargs_test = {
            'network_query_fn': network_query_fn,
            'perturb': False,
            'N_importance': args.N_importance,
            'feature_array': feature_array,
            'pose_array': pose_array,
            'N_samples': args.N_samples,
            'network_fn': model,
            'per_frame_alignment': per_frame_alignment,
            'deformation_field': deformation_field,
            'mode': args.mode,
            'raw_noise_std': 0.,
            'truncation': args.trunc,
            'sc_factor': args.sc_factor,
        }

        # NDC only good for LLFF-style forward facing data
        if args.dataset_type != 'llff' or args.no_ndc:
            print('Not ndc!')
            render_kwargs_test['ndc'] = False
            
        return render_kwargs_test, models, iteration
    
    print(args.extckpt)
    for ckpt in args.extckpt:
        print(f"Extracting mesh from checkpoint {ckpt}")
        try:
            render_kwargs_test, models, iteration = create_test_model(args, load_ckpt_iter=ckpt)
        except IndexError as e:
            print(e)
            continue

        feature_array = render_kwargs_test.get('feature_array')        

        network_fn = render_kwargs_test['network_fn']
        feature_array_fn = feature_array

        isolevel = 0.0 if args.mode == 'sdf' else 20.0
        mesh_savepath = os.path.join(outdir, '{:06d}.ply'.format(iteration))

        extract_mesh(
            render_kwargs_test['network_query_fn'], 
            feature_array_fn, 
            network_fn, 
            args, 
            isolevel=isolevel, 
            scene_name=scene_name,
            mesh_savepath=mesh_savepath,
            voxel_size=args.ext_voxel_size
        )
        
        torch.cuda.empty_cache()