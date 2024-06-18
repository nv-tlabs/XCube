#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "../common/kdtree_cuda.cuh"


// Note: the distance returned here is squared!
std::pair<torch::Tensor, torch::Tensor> knn_query_fast(const torch::Tensor& queries,
                              const torch::Tensor& ref_xyz, int nb_points) {
    CHECK_CUDA(queries)
    CHECK_IS_FLOAT(queries)
    CHECK_CUDA(ref_xyz)
    CHECK_IS_FLOAT(ref_xyz)

    // If ref is less than max-leaf-size, an error will occur. Just do cdist.
    if (ref_xyz.size(0) < 64) {
        torch::Tensor cdist = torch::cdist(queries.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3)}),
                                           ref_xyz.index({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3)}));
        auto cmin = torch::topk(cdist, nb_points, 1, false);
        return std::make_pair(torch::square(std::get<0>(cmin)), std::get<1>(cmin).to(torch::kInt32));
    }

    // Index requires reference to have stride 4.
    torch::Tensor strided_ref = ref_xyz;
    if (ref_xyz.stride(0) != 4) {
        strided_ref = torch::zeros({strided_ref.size(0), 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        strided_ref.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(torch::indexing::None, 3)}, ref_xyz);
    }

    // Build KDTree based on reference point cloud
    size_t n_ref = strided_ref.size(0);
    tinyflann::KDTreeCuda3dIndex<tinyflann::CudaL2> knn_index(strided_ref.data_ptr<float>(), n_ref);
    knn_index.buildIndex();

    // Compute for each point its nearest N neighbours.
    int n_query = queries.size(0);
    torch::Tensor dist = torch::zeros({n_query, nb_points}, at::device(queries.device()).dtype(at::ScalarType::Float));
    torch::Tensor indices = torch::zeros({n_query, nb_points}, at::device(queries.device()).dtype(at::ScalarType::Int));

    knn_index.knnSearch(queries.data_ptr<float>(), n_query, queries.stride(0), indices.data_ptr<int>(),
                        dist.data_ptr<float>(), nb_points);

    return std::make_pair(dist, indices);
}
