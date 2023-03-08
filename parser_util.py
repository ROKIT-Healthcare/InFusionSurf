import configargparse
import os

def get_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/scannet/scene0005_00', help='input data directory')
    
    parser.add_argument("--use_filtered_depth", action='store_true',
                        help='use filtered depth when loading depth')
    
    # training options
    parser.add_argument("--netdepth", type=int, default=3,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument("--dense_features", type=int, default=12,
                        help='dense feature number')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_iters", type=int, default=1000000,
                        help='number of iterations for which to train the network')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights tar file to reload for coarse network')
    parser.add_argument("--rgb_weight", type=float,
                        default=1.0, help='weight of the img loss')
    parser.add_argument("--depth_weight", type=float,
                        default=1.0, help='weight of the depth loss')
    parser.add_argument("--fs_weight", type=float,
                        default=1.0, help='weight of the free-space loss')
    parser.add_argument("--trunc_weight", type=float,
                        default=1.0, help='weight of the truncation loss')
    parser.add_argument("--rgb_loss_type", type=str, default='l2',
                        help='which RGB loss to use - l1/l2 are currently supported')
    parser.add_argument("--sdf_loss_type", type=str, default='l2',
                        help='which SDF loss to use - l1/l2 are currently supported')
    parser.add_argument("--frame_features", type=int, default=0,
                        help='number of channels of the learnable per-frame features')
    parser.add_argument("--optimize_poses", action='store_true',
                        help='optimize a pose refinement for the initial poses')
    parser.add_argument("--use_deformation_field", action='store_true',
                        help='use a deformation field to account for inaccuracies in intrinsic parameters')
    parser.add_argument("--use_per_frame_alignment", action='store_true',
                        help='use a per frame alignment to correct intrinsic error per frame')

    parser.add_argument("--voxel_size", type=float, default=0.1,
                        help='voxel size for dense grid')
    
    parser.add_argument("--fusion_init", type=int, default=3000,
                        help='TSDF fution prior pre-training step iterations')

    parser.add_argument("--progressive_learning", action="append", default=None, required=False, type=int,
                        help='progressive learning steps')
    
    # rendering options
    parser.add_argument("--N_batch_pt", type=int, default=1024,
                        help='number of voxel batch for pretraining')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, 1 for method of Instant Neural Graphics Primitives [Müller et al. 2022], -1 for none')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--mode", type=str, default='density',
                        help='whether the network predicts density or SDF values')
    parser.add_argument("--trunc", type=float, default=0.05,
                        help='length of the truncation region in meters')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='scannet',
                        help='options: llff / blender / deepvoxels / synthetic / scannet')
    parser.add_argument("--trainskip", type=int, default=1,
                        help='will load 1/N images from the training set, useful for large datasets like deepvoxels')
    parser.add_argument("--factor", type=int, default=1,
                        help='downsample factor for depth images')
    parser.add_argument("--sc_factor", type=float, default=1.0,
                        help='factor by which to scale the camera translation and the depth maps')
    parser.add_argument("--translation", action="append", default=None, required=False, type=float,
                        help='translation vector for the camera poses')
    parser.add_argument("--crop", type=int, default=0,
                        help='number of pixels by which to crop the image edges (e.g. due to undistortion artifacts')
    parser.add_argument("--near", type=float, default=0.0, help='distance to the near plane')
    parser.add_argument("--far", type=float, default=1.0, help='distance to the far plane')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    
    return parser