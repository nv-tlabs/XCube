#include <torch/extension.h>
#include <optional>

/**
knn_query_fast:
    :param query (N, 3)
    :param reference (M, 3)
    :param int knn K
    :return
        dist (N, K) squared distances
        idx (N, K) index into reference.
**/
std::pair<torch::Tensor, torch::Tensor> knn_query_fast(const torch::Tensor& queries,
                              const torch::Tensor& ref_xyz, int k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_query_fast", &knn_query_fast, "Query k-NN neighbours.");
}
