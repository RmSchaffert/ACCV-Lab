#ifndef LANE_HELPERS_EXT_IMPL_INPUT_CHECKS_H
#define LANE_HELPERS_EXT_IMPL_INPUT_CHECKS_H

#include <torch/torch.h>

namespace lane_helpers::ext_impl {

inline void check_cuda(const at::Tensor& tensor, const char* description) {
    AT_ASSERTM(tensor.is_cuda(), description, " must be on CUDA");
}

inline void check_device(const at::Tensor& tensor, const char* description) {
    TORCH_CHECK(tensor.is_cpu() || tensor.is_cuda(), description, " must be on CPU or CUDA");
}

inline void check_contiguous(const at::Tensor& tensor, const char* description) {
    TORCH_CHECK(tensor.is_contiguous(), description, " must be contiguous");
}

inline void check_type(const at::Tensor& tensor, const char* description) {
    if (tensor.is_cuda()) {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat64 ||
                        tensor.scalar_type() == torch::kFloat16 || tensor.scalar_type() == torch::kBFloat16,
                    description, " must have dtype float16, float32, float64, or bfloat16 on CUDA");
    } else {
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat64,
                    description, " must have dtype float32 or float64 on CPU");
    }
}

inline void check_same_device(const at::Tensor& lhs, const at::Tensor& rhs, const char* message) {
    TORCH_CHECK(lhs.device() == rhs.device(), message);
}

inline void check_same_dtype(const at::Tensor& lhs, const at::Tensor& rhs, const char* message) {
    AT_ASSERTM(lhs.scalar_type() == rhs.scalar_type(), message);
}

inline void check_sample_size_type(const at::Tensor& sample_sizes, const char* description) {
    TORCH_CHECK(sample_sizes.scalar_type() == at::kInt || sample_sizes.scalar_type() == at::kLong,
                description, " must have dtype int32 or int64");
}

inline void check_sample_sizes(const at::Tensor& sample_sizes, int max_size, const char* description) {
    if (sample_sizes.numel() == 0) {
        return;
    }
    TORCH_CHECK(!torch::any(sample_sizes < 0).item<bool>() &&
                    !torch::any(sample_sizes > max_size).item<bool>(),
                description, " values must be in [0, ", max_size, "]");
}

}  // namespace lane_helpers::ext_impl

#define CHECK_CUDA(x) ::lane_helpers::ext_impl::check_cuda((x), #x)
#define CHECK_DEVICE(x) ::lane_helpers::ext_impl::check_device((x), #x)
#define CHECK_CONTIGUOUS(x) ::lane_helpers::ext_impl::check_contiguous((x), #x)
#define CHECK_TYPE(x) ::lane_helpers::ext_impl::check_type((x), #x)

#endif  // LANE_HELPERS_EXT_IMPL_INPUT_CHECKS_H
