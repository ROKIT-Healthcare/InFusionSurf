#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <vector>

#include <iostream>

template <typename scalar_t>
__global__ void integrate_cuda_kernel(
    scalar_t *__restrict__ tsdf_vol,
    scalar_t *__restrict__ weight_vol,
    scalar_t *__restrict__ color_vol,
    const int *__restrict__ vol_dim,
    const scalar_t *__restrict__ vol_origin,
    const scalar_t *__restrict__ cam_intr,
    const scalar_t *__restrict__ cam_pose,
    const scalar_t *__restrict__ other_params,
    const scalar_t *__restrict__ color_im,
    const scalar_t *__restrict__ depth_im,
    const int n_gpu_loops)
{
    // Get voxel index
    for (int gpu_loop_idx = 0; gpu_loop_idx < n_gpu_loops; gpu_loop_idx++)
    {

        int max_threads_per_block = blockDim.x;
        int block_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
        int voxel_idx = gpu_loop_idx * gridDim.x * gridDim.y * gridDim.z * max_threads_per_block + block_idx * max_threads_per_block + threadIdx.x;

        int vol_dim_x = vol_dim[0];
        int vol_dim_y = vol_dim[1];
        int vol_dim_z = vol_dim[2];

        if (voxel_idx > vol_dim_x * vol_dim_y * vol_dim_z)
            continue;

        // Get voxel grid coordinates (note: be careful when casting)
        float voxel_x = floorf(((float)voxel_idx) / ((float)(vol_dim_y * vol_dim_z)));
        float voxel_y = floorf(((float)(voxel_idx - ((int)voxel_x) * vol_dim_y * vol_dim_z)) / ((float)vol_dim_z));
        float voxel_z = (float)(voxel_idx - ((int)voxel_x) * vol_dim_y * vol_dim_z - ((int)voxel_y) * vol_dim_z);

        // Voxel grid coordinates to world coordinates
        float voxel_size = other_params[0];

        float pt_x = vol_origin[0] + voxel_x * voxel_size;
        float pt_y = vol_origin[1] + voxel_y * voxel_size;
        float pt_z = vol_origin[2] + voxel_z * voxel_size;

        float tmp_pt_x = pt_x - cam_pose[0 * 4 + 3];
        float tmp_pt_y = pt_y - cam_pose[1 * 4 + 3];
        float tmp_pt_z = pt_z - cam_pose[2 * 4 + 3];

        float cam_pt_x = cam_pose[0 * 4 + 0] * tmp_pt_x + cam_pose[1 * 4 + 0] * tmp_pt_y + cam_pose[2 * 4 + 0] * tmp_pt_z;
        float cam_pt_y = cam_pose[0 * 4 + 1] * tmp_pt_x + cam_pose[1 * 4 + 1] * tmp_pt_y + cam_pose[2 * 4 + 1] * tmp_pt_z;
        cam_pt_y = -cam_pt_y;

        float cam_pt_z = cam_pose[0 * 4 + 2] * tmp_pt_x + cam_pose[1 * 4 + 2] * tmp_pt_y + cam_pose[2 * 4 + 2] * tmp_pt_z;
        cam_pt_z = -cam_pt_z;

        // Camera coordinates to image pixels
        int pixel_x = (int)roundf(cam_intr[0 * 3 + 0] * (cam_pt_x / cam_pt_z) + cam_intr[0 * 3 + 2]);
        int pixel_y = (int)roundf(cam_intr[1 * 3 + 1] * (cam_pt_y / cam_pt_z) + cam_intr[1 * 3 + 2]);

        // Skip if outside view frustum
        int im_h = (int)other_params[1];
        int im_w = (int)other_params[2];

        if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z < 0)
            continue;

        // Skip invalid depth
        float depth_value = depth_im[pixel_y * im_w + pixel_x];

        if (depth_value == 0)
            continue;

        // Integrate TSDF
        float trunc_margin = other_params[3];
        float depth_diff = depth_value - cam_pt_z;

        if (depth_diff < -trunc_margin)
            continue;

        float dist = fmin(1.0f, depth_diff / trunc_margin);
        float w_old = weight_vol[voxel_idx];
        float obs_weight = other_params[4];

        float w_new = w_old + obs_weight;

        weight_vol[voxel_idx] = w_new;

        tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx] * w_old + obs_weight * dist) / w_new;
        /*
                // Integrate color
                float old_color = color_vol[voxel_idx];
                float old_b = floorf(old_color / (256 * 256));
                float old_g = floorf((old_color - old_b * 256 * 256) / 256);
                float old_r = old_color - old_b * 256 * 256 - old_g * 256;

                float new_r = color_im[pixel_y * im_w + pixel_x];
                float new_g = color_im[im_h * im_w + pixel_y * im_w + pixel_x];
                float new_b = color_im[2 * im_h * im_w + pixel_y * im_w + pixel_x];

                new_b = fmin(roundf((old_b * w_old + obs_weight * new_b) / w_new), 255.0f);
                new_g = fmin(roundf((old_g * w_old + obs_weight * new_g) / w_new), 255.0f);
                new_r = fmin(roundf((old_r * w_old + obs_weight * new_r) / w_new), 255.0f);

                color_vol[voxel_idx] = new_b * 256 * 256 + new_g * 256 + new_r;
        */
    }
}

std::vector<torch::Tensor> tsdf_fusion_cuda(
    torch::Tensor depth_images,
    torch::Tensor color_images,
    torch::Tensor cam_poses,
    torch::Tensor cam_intr,
    torch::Tensor scene_bnds,
    float voxel_size,
    float trunc_margin,
    float obs_weight)
{
    // Calculate voxel dimension
    auto vol_dim = (scene_bnds.index({1, "..."}) - scene_bnds.index({0, "..."})) / voxel_size;
    const auto vol_origin = scene_bnds.index({0, "..."}).contiguous();

    vol_dim = vol_dim.ceil();
    vol_dim = vol_dim.to(torch::kInt).contiguous();

    const auto vol_dim_cpu = vol_dim.to(torch::kCPU, vol_dim.scalar_type());
    const auto vol_dim_a = vol_dim_cpu.accessor<int, 1>();

    // Build TSDF, color, weight voxel grid
    auto tsdf_vol = torch::ones({vol_dim_a[0], vol_dim_a[1], vol_dim_a[2]}).to(depth_images.device(), depth_images.scalar_type());
    auto weight_vol = torch::zeros_like(tsdf_vol);
    auto color_vol = torch::zeros_like(tsdf_vol);

    const int vol_dim_total = vol_dim_a[0] * vol_dim_a[1] * vol_dim_a[2];

    // Other parameters
    const std::vector<float> other_params({voxel_size, (float)depth_images.size(1), (float)depth_images.size(2), trunc_margin, obs_weight});
    const auto other_params_gpu_tensor = torch::from_blob((void *)other_params.data(), {5}, torch::TensorOptions().dtype(torch::kFloat32)).to(depth_images.device(), depth_images.scalar_type());

    // Specify cuda kernel environment
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const int threads = prop.maxThreadsPerBlock;

    float n_blocks = vol_dim_total / (float)threads;

    const int grid_dim_x = min(prop.maxGridSize[0], (int)cbrt(n_blocks));

    n_blocks = n_blocks / grid_dim_x;
    const int grid_dim_y = min(prop.maxGridSize[1], (int)sqrt(n_blocks));

    n_blocks = n_blocks / grid_dim_y;
    const int grid_dim_z = min(prop.maxGridSize[2], (int)ceil(n_blocks));

    const int max_gpu_grid_dim = grid_dim_x * grid_dim_y * grid_dim_z;
    const int n_gpu_loops = (int)ceil(vol_dim_total / (float)(max_gpu_grid_dim * threads));

    const dim3 grid(grid_dim_x, grid_dim_y, grid_dim_z);

    for (int idx = 0; idx < depth_images.size(0); idx++)
    {
        const auto color_im = color_images.index({idx, "..."}).contiguous();
        const auto depth_im = depth_images.index({idx, "..."}).contiguous();
        const auto cam_pose = cam_poses.index({idx, "..."}).contiguous();

        AT_DISPATCH_FLOATING_TYPES(depth_images.scalar_type(), "integrate_cuda_kernel", ([&]
                                                                                         { integrate_cuda_kernel<scalar_t><<<grid, threads>>>(
                                                                                               tsdf_vol.data_ptr<scalar_t>(),
                                                                                               weight_vol.data_ptr<scalar_t>(),
                                                                                               color_vol.data_ptr<scalar_t>(),
                                                                                               vol_dim.data_ptr<int>(),
                                                                                               vol_origin.data_ptr<scalar_t>(),
                                                                                               cam_intr.data_ptr<scalar_t>(),
                                                                                               cam_pose.data_ptr<scalar_t>(),
                                                                                               other_params_gpu_tensor.data_ptr<scalar_t>(),
                                                                                               color_im.data_ptr<scalar_t>(),
                                                                                               depth_im.data<scalar_t>(),
                                                                                               n_gpu_loops); }));
    }

    return {tsdf_vol, color_vol};
}
