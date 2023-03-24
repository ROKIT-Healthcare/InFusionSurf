#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> tsdf_fusion_cuda(
    torch::Tensor depth_images,
    torch::Tensor color_images,
    torch::Tensor cam_poses,
    torch::Tensor cam_intr,
    torch::Tensor scene_bnds,
    float voxel_size,
    float trunc_margin,
    float obs_weight);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> tsdf_fusion(
    torch::Tensor depth_images,
    torch::Tensor color_images,
    torch::Tensor cam_poses,
    torch::Tensor cam_intr,
    torch::Tensor scene_bnds,
    float voxel_size,
    float trunc_margin,
    float obs_weight)
{

    CHECK_CUDA(depth_images);
    CHECK_CUDA(color_images);
    CHECK_CUDA(cam_poses);
    CHECK_CUDA(cam_intr);
    CHECK_CUDA(scene_bnds);

    if (!depth_images.is_contiguous())
        depth_images = depth_images.contiguous();

    if (!color_images.is_contiguous())
        color_images = color_images.contiguous();

    if (!cam_poses.is_contiguous())
        cam_poses = cam_poses.contiguous();

    if (!cam_intr.is_contiguous())
        cam_intr = cam_intr.contiguous();

    if (!scene_bnds.is_contiguous())
        scene_bnds = scene_bnds.contiguous();

    return tsdf_fusion_cuda(
        depth_images,
        color_images,
        cam_poses,
        cam_intr,
        scene_bnds,
        voxel_size,
        trunc_margin,
        obs_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fusion", &tsdf_fusion, "tsdf_fusion (CUDA)");
}