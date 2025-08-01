/*
Batch version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query_gpu.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {

    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(idx_tensor);

    const float* new_xyz = new_xyz_tensor.data_ptr<float>();
    const float* xyz = xyz_tensor.data_ptr<float>();
    int* idx = idx_tensor.data_ptr<int>();

    ball_query_kernel_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx);
    return 1;
}
