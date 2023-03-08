# Copyright (c) 2018 Andy Zeng
import os
import sys
import time

import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

import imageio
import numpy as np
import re
import cv2

import parser_util
from pycuda.compiler import SourceModule

from datasets.load_scannet import load_scannet_data

# Cuda kernel function (C++)
cuda_src_mod = SourceModule("""
    __global__ void integrate(float * tsdf_vol,
                              float * weight_vol,
                              float * color_vol,
                              float * vol_dim,
                              float * vol_origin,
                              float * cam_intr,
                              float * cam_pose,
                              float * other_params,
                              float * color_im,
                              float * depth_im) {
      // Get voxel index
      int gpu_loop_idx = (int) other_params[0];
      int max_threads_per_block = blockDim.x;
      int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
      int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
      int vol_dim_x = (int) vol_dim[0];
      int vol_dim_y = (int) vol_dim[1];
      int vol_dim_z = (int) vol_dim[2];
      if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
          return;
      // Get voxel grid coordinates (note: be careful when casting)
      float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
      float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
      float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
      // Voxel grid coordinates to world coordinates
      float voxel_size = other_params[1];
      float pt_x = vol_origin[0]+voxel_x*voxel_size;
      float pt_y = vol_origin[1]+voxel_y*voxel_size;
      float pt_z = vol_origin[2]+voxel_z*voxel_size;
      // World coordinates to camera coordinates
      float tmp_pt_x = pt_x-cam_pose[0*4+3];
      float tmp_pt_y = pt_y-cam_pose[1*4+3];
      float tmp_pt_z = pt_z-cam_pose[2*4+3];
      float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
      float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
      cam_pt_y = -cam_pt_y;
      float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
      cam_pt_z = -cam_pt_z;
      // Camera coordinates to image pixels
      int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
      int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
      // Skip if outside view frustum
      int im_h = (int) other_params[2];
      int im_w = (int) other_params[3];
      if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z < 0)
          return;
      // Skip invalid depth
      float depth_value = depth_im[pixel_y*im_w+pixel_x];
      if (depth_value == 0)
          return;
      // Integrate TSDF
      float trunc_margin = other_params[4];
      float depth_diff = depth_value - cam_pt_z;
      if (depth_diff < -trunc_margin)
          return;
      float dist = fmin(1.0f,depth_diff/trunc_margin);
      float w_old = weight_vol[voxel_idx];
      float obs_weight = other_params[5];
      float w_new = w_old + obs_weight;
      weight_vol[voxel_idx] = w_new;
      tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
      // Integrate color
      float old_color = color_vol[voxel_idx];
      float old_b = floorf(old_color/(256*256));
      float old_g = floorf((old_color-old_b*256*256)/256);
      float old_r = old_color-old_b*256*256-old_g*256;
      float new_color = color_im[pixel_y*im_w+pixel_x];
      float new_b = floorf(new_color/(256*256));
      float new_g = floorf((new_color-new_b*256*256)/256);
      float new_r = new_color-new_b*256*256-new_g*256;
      new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
      new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
      new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
      color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
    }""")


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """
    def __init__(self, vol_bnds, voxel_size, trunc_margin):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = trunc_margin  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = np.ceil((self._vol_bnds[:,1]-self._vol_bnds[:,0])/self._voxel_size).copy(order='C').astype(int)
        self._vol_bnds[:,1] = self._vol_bnds[:,0]+self._vol_dim*self._voxel_size
        self._vol_origin = self._vol_bnds[:,0].copy(order='C').astype(np.float32)

        print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
          self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
          self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
        )

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Copy voxel volumes to GPU
        self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
        cuda.memcpy_htod(self._tsdf_vol_gpu,self._tsdf_vol_cpu)
        self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
        cuda.memcpy_htod(self._weight_vol_gpu,self._weight_vol_cpu)
        self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
        cuda.memcpy_htod(self._color_vol_gpu,self._color_vol_cpu)

        self._cuda_integrate = cuda_src_mod.get_function("integrate")

        # Determine block/grid size on GPU
        gpu_dev = cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
        n_blocks = int(np.ceil(float(np.prod(self._vol_dim))/float(self._max_gpu_threads_per_block)))
        grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y,int(np.floor(np.sqrt(n_blocks/grid_dim_x))))
        grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z,int(np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
        self._max_gpu_grid_dim = np.array([grid_dim_x,grid_dim_y,grid_dim_z]).astype(int)
        self._n_gpu_loops = int(np.ceil(float(np.prod(self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))

    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
        """
        im_h, im_w = depth_im.shape[:2]

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[...,2]*self._color_const + color_im[...,1]*256 + color_im[...,0])

        for gpu_loop_idx in range(self._n_gpu_loops):
            self._cuda_integrate(self._tsdf_vol_gpu,
                                self._weight_vol_gpu,
                                self._color_vol_gpu,
                                cuda.InOut(self._vol_dim.astype(np.float32)),
                                cuda.InOut(self._vol_origin.astype(np.float32)),
                                cuda.InOut(cam_intr.reshape(-1).astype(np.float32)),
                                cuda.InOut(cam_pose.reshape(-1).astype(np.float32)),
                                cuda.InOut(np.asarray([
                                  gpu_loop_idx,
                                  self._voxel_size,
                                  im_h,
                                  im_w,
                                  self._trunc_margin,
                                  obs_weight
                                ], np.float32)),
                                cuda.InOut(color_im.reshape(-1).astype(np.float32)),
                                cuda.InOut(depth_im.reshape(-1).astype(np.float32)),
                                block=(self._max_gpu_threads_per_block,1,1),
                                grid=(
                                  int(self._max_gpu_grid_dim[0]),
                                  int(self._max_gpu_grid_dim[1]),
                                  int(self._max_gpu_grid_dim[2]),
                                )
        )

    def get_volume(self):
        cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
        cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)

        return self._tsdf_vol_cpu, self._color_vol_cpu

def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud.
    """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose, far):
    """Get corners of 3D camera view frustum of depth image
    """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = min([np.max(depth_im), far])

    view_frust_pts = np.array([
        (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
        -(np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
        -np.array([0,max_depth,max_depth,max_depth,max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T

    return view_frust_pts

def tsdf_fusion():
    parser = parser_util.get_parser()
    args = parser.parse_args()
    print(args)
    
    grid_dir = os.path.join(args.datadir, 'geo_prior')
    
    if args.dataset_type == "scannet":
        images, depth_images, poses, hwf, frame_indices = load_scannet_data(basedir=args.datadir,
                                                                            trainskip=args.trainskip,
                                                                            downsample_factor=args.factor,
                                                                            translation=args.translation,
                                                                            sc_factor=args.sc_factor,
                                                                            crop=args.crop,
                                                                            use_filtered_depth=args.use_filtered_depth)
    else:
        print('Unknown dataset type', grid_parameter.get('dataset_type'), 'exiting')
        raise Exception

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)

    vol_bnds = np.zeros((3,2))

    cam_intr = np.array([[focal, 0, W / 2],
                         [0, focal, H / 2],
                         [0, 0, 1]])

    for cam_pose, depth_im in zip(poses, depth_images):
        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose, args.far)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

    vol_bnds = vol_bnds.astype(np.float32)
    xyz_min, xyz_max = vol_bnds[:, 0], vol_bnds[:, 1]
    print(f"scene bound: {xyz_min}, {xyz_max}")

    os.makedirs(grid_dir, exist_ok=True)

    with open(os.path.join(grid_dir, 'scene_bound.txt'), 'w') as f:
        f.write('\n'.join(
            [f"[{xyz_min[0]}, {xyz_min[1]}, {xyz_min[2]}]",
             f"[{xyz_max[0]}, {xyz_max[1]}, {xyz_max[2]}]"]))

    print("Initializing voxel volume...")
    tsdf_vol = TSDFVolume(vol_bnds, 
                          voxel_size=args.voxel_size * args.sc_factor, 
                          trunc_margin=args.trunc * args.sc_factor)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    print("fusion started")
    for cam_pose, color_image, depth_im, in zip(poses, images, depth_images):
        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    fps = images.shape[0] / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    tsdf_np, color_np = tsdf_vol.get_volume()

    tsdf_file = os.path.join(grid_dir, 'init_tsdf.npy')
    np.save(tsdf_file, tsdf_np)
    
if __name__ == '__main__':
    print(os.environ)
    tsdf_fusion()