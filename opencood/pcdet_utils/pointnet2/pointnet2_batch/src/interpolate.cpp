/*
Batch version of point interpolation, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/

#include <torch/serialize/tensor.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interpolate_gpu.h"


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void three_nn_wrapper_fast(int b, int n, int m, at::Tensor unknown_tensor,
    at::Tensor known_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor) {

    CHECK_INPUT(unknown_tensor);
    CHECK_INPUT(known_tensor);
    CHECK_INPUT(dist2_tensor);
    CHECK_INPUT(idx_tensor);

    const float* unknown = unknown_tensor.data_ptr<float>();
    const float* known = known_tensor.data_ptr<float>();
    float* dist2 = dist2_tensor.data_ptr<float>();
    int* idx = idx_tensor.data_ptr<int>();

    three_nn_kernel_launcher_fast(b, n, m, unknown, known, dist2, idx);
}


void three_interpolate_wrapper_fast(int b, int c, int m, int n,
                         at::Tensor points_tensor,
                         at::Tensor idx_tensor,
                         at::Tensor weight_tensor,
                         at::Tensor out_tensor) {

    CHECK_INPUT(points_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(weight_tensor);
    CHECK_INPUT(out_tensor);

    const float* points = points_tensor.data_ptr<float>();
    const float* weight = weight_tensor.data_ptr<float>();
    float* out = out_tensor.data_ptr<float>();
    const int* idx = idx_tensor.data_ptr<int>();

    three_interpolate_kernel_launcher_fast(b, c, m, n, points, idx, weight, out);
}


void three_interpolate_grad_wrapper_fast(int b, int c, int n, int m,
                            at::Tensor grad_out_tensor,
                            at::Tensor idx_tensor,
                            at::Tensor weight_tensor,
                            at::Tensor grad_points_tensor) {

    CHECK_INPUT(grad_out_tensor);
    CHECK_INPUT(idx_tensor);
    CHECK_INPUT(weight_tensor);
    CHECK_INPUT(grad_points_tensor);

    const float* grad_out = grad_out_tensor.data_ptr<float>();
    const float* weight = weight_tensor.data_ptr<float>();
    float* grad_points = grad_points_tensor.data_ptr<float>();
    const int* idx = idx_tensor.data_ptr<int>();

    three_interpolate_grad_kernel_launcher_fast(b, c, n, m, grad_out, idx, weight, grad_points);
}
