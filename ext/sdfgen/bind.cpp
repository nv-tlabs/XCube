#include <torch/extension.h>
//#include <Eigen/Dense>
using namespace pybind11::literals;

// Use the classical NN search algorithm.
//      Our kd-tree is 10x faster than pytorch-3d's kNN algorithm (9ms vs. 110ms+)
std::vector<torch::Tensor> sdf_from_points(const torch::Tensor& queries,
                              const torch::Tensor& ref_xyz,
                              const torch::Tensor& ref_normal,
                              int nb_points, float stdv, bool compute_grad, bool imls, int adaptive_knn);

#ifdef BUILD_SDF_FROM_MESH
// Will build BVH of the mesh, use the raystab (OptiX) algorithm to determine in-out.
//      borrowed from [Instant-NGP, 2022]
//   ref_triangles should be contiguous with shape (T, 3, 3) of float.
// However, this is a bit slower than point-based. (~20ms)
enum class EMeshSdfMode : int {
    Watertight,
    Raystab,
    PathEscape,
};
torch::Tensor sdf_from_mesh(const torch::Tensor& queries, const torch::Tensor& ref_triangles, EMeshSdfMode mode);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sdf_from_points", &sdf_from_points,
          "Compute sdf value from reference points.",
          "queries"_a, "ref_xyz"_a, "ref_normal"_a, "nb_points"_a, "stdv"_a,
          "compute_grad"_a=false, "imls"_a=false, "adaptive_knn"_a=0
    );
#ifdef BUILD_SDF_FROM_MESH
    m.def("sdf_from_mesh", &sdf_from_mesh, "Compute sdf value from reference mesh.");
    py::enum_<EMeshSdfMode>(m, "MeshSdfMode")
            .value("Watertight", EMeshSdfMode::Watertight)
            .value("Raystab", EMeshSdfMode::Raystab)
            .value("PathEscape", EMeshSdfMode::PathEscape);
#endif
}
