#include <torch/extension.h>
#include "functions.h"


#define CHECK_DEVICE(a, b) TORCH_CHECK(a.device() == b.device(), #a " and " #b " are not on the same device!")

void pybind_kernel_eval(py::module& m) {
    m.def("kernel_evaluation", [](const fvdb::GridBatch& grid,
            const fvdb::JaggedTensor& query, const fvdb::JaggedTensor& queryKernel,
            const fvdb::JaggedTensor& gridKernel, const fvdb::JaggedTensor& gridAlpha) -> fvdb::JaggedTensor {
        TORCH_CHECK(gridAlpha.element_count() == grid.total_voxels(), "grid_alpha must have number of voxels");
        torch::Tensor dummyGrad = torch::zeros({0, 0, 3}, 
                torch::TensorOptions().device(query.device()).dtype(query.dtype()));
        auto res = KernelEvaluation::apply(
                grid.impl(), false,
                query, queryKernel.jdata(), gridKernel.jdata(), gridAlpha.jdata(), dummyGrad);
        return query.jagged_like(res[0]);
    }, py::arg("grid"), py::arg("query"), py::arg("query_kernel"), py::arg("grid_kernel"), py::arg("grid_alpha"));

    m.def("kernel_evaluation_grad", [](const fvdb::GridBatch& grid,
            const fvdb::JaggedTensor& query, const fvdb::JaggedTensor& queryKernel,
            const fvdb::JaggedTensor& gridKernel, const fvdb::JaggedTensor& gridAlpha,
            const fvdb::JaggedTensor& gradKernelQuery) -> std::vector<fvdb::JaggedTensor> {
        TORCH_CHECK(gradKernelQuery.dim() == 3);
        TORCH_CHECK(gridAlpha.element_count() == grid.total_voxels(), "grid_alpha must have number of voxels");
        auto res = KernelEvaluation::apply(
                grid.impl(), true,
                query, queryKernel.jdata(), gridKernel.jdata(), gridAlpha.jdata(), gradKernelQuery.jdata());
        return {query.jagged_like(res[0]), query.jagged_like(res[1])};
    }, py::arg("grid"), py::arg("query"), py::arg("query_kernel"), py::arg("grid_kernel"),
          py::arg("grid_alpha"), py::arg("grad_kernel_query"));

    m.def("matrix_building", [](const fvdb::GridBatch& gridI, const fvdb::GridBatch& gridJ,
                                const fvdb::JaggedTensor& ptsPos,
                                const fvdb::JaggedTensor& ptsKernelI, const fvdb::JaggedTensor& ptsKernelJ,
                                const fvdb::JaggedTensor& iKernel, const fvdb::JaggedTensor& jKernel,
                                const fvdb::JaggedTensor& gradPtsKernelPosI, const fvdb::JaggedTensor& gradPtsKernelPosJ,
                                const fvdb::JaggedTensor& indexMap,
                                bool grad, const fvdb::JaggedTensor& matRef) {
        TORCH_CHECK(iKernel.element_count() == gridI.total_voxels(), "iKernel does not have correct size!");
        TORCH_CHECK(jKernel.element_count() == gridJ.total_voxels(), "jKernel does not have correct size!");
        auto ret = MatrixBuilding::apply(
                gridI.impl(), gridJ.impl(),
                ptsPos,
                ptsKernelI.jdata(), ptsKernelJ.jdata(),
                iKernel.jdata(), jKernel.jdata(),
                gradPtsKernelPosI.jdata(), gradPtsKernelPosJ.jdata(),
                indexMap.jdata(), grad, matRef.element_count());
        return matRef.jagged_like(ret[0]);
    }, py::arg("grid_i"), py::arg("grid_j"), py::arg("pts_pos"),
          py::arg("pts_kernel_i"), py::arg("pts_kernel_j"),
          py::arg("i_kernel"), py::arg("j_kernel"),
          py::arg("grad_pts_kernel_pos_i"), py::arg("grad_pts_kernel_pos_j"),
          py::arg("index_map"), py::arg("grad"), py::arg("mat_ref"));

    m.def("k_building", [](const fvdb::GridBatch& grid,
                           const fvdb::JaggedTensor& kernel,
                           const fvdb::JaggedTensor& indexMap,
                           const fvdb::JaggedTensor& matRef) -> fvdb::JaggedTensor {
        TORCH_CHECK(kernel.element_count() == grid.total_voxels(), "Kernel does not have correct size!");
        auto ret = KBuilding::apply(
            grid.impl(), kernel.jdata(), indexMap.jdata(), matRef.element_count());
        return matRef.jagged_like(ret[0]);
    }, py::arg("grid"), py::arg("kernel"), py::arg("index_map"), py::arg("mat_ref"));

    m.def("rhs_evaluation", [](const fvdb::GridBatch& grid,
                               const fvdb::JaggedTensor& pts, const fvdb::JaggedTensor& ptsKernel,
                               const fvdb::JaggedTensor& gridKernel, const fvdb::JaggedTensor& gradKernelPts,
                               const fvdb::JaggedTensor& ptsData) -> fvdb::JaggedTensor {
        TORCH_CHECK(gradKernelPts.dim() == 3);
        TORCH_CHECK(gridKernel.element_count() == grid.total_voxels(), "grid_kernel must have number of voxels");
        auto ret = RhsEvaluation::apply(
                grid.impl(), 
                pts, ptsKernel.jdata(), gridKernel.jdata(), gradKernelPts.jdata(), ptsData.jdata());
        return gridKernel.jagged_like(ret[0]);
    }, py::arg("grid"), py::arg("pts"), py::arg("pts_kernel"), py::arg("grid_kernel"),
       py::arg("grad_kernel_pts"), py::arg("pts_data"));

    m.def("build_coo_indexer", [](const fvdb::GridBatch& grid_i, 
                                  const fvdb::GridBatch& grid_j) -> fvdb::JaggedTensor {
        return buildCOOIndexer(grid_i.impl(), grid_j.impl());
    });

    m.def("csr_matrix_multiplication", [](
            const fvdb::GridBatch& grid_i, const fvdb::GridBatch& grid_j,
            const fvdb::JaggedTensor& coordsI, const fvdb::JaggedTensor& coordsJ,
            const fvdb::JaggedTensor& iValue, const fvdb::JaggedTensor& jValue,
            const fvdb::JaggedTensor& iRowPtr, const fvdb::JaggedTensor& jRowPtr,
            const fvdb::JaggedTensor& iColInds, const fvdb::JaggedTensor& jColInds,
            const fvdb::JaggedTensor& indexMap, const fvdb::JaggedTensor& numEntries) -> fvdb::JaggedTensor {
        CHECK_DEVICE(grid_j, grid_j);
        TORCH_CHECK(grid_i.grid_count() == grid_j.grid_count(), "Batch size must be the same!");
        TORCH_CHECK(coordsI.size(0) == grid_i.total_voxels());
        TORCH_CHECK(coordsJ.size(0) == grid_j.total_voxels());
        TORCH_CHECK(iValue.size(0) == iColInds.size(0));
        TORCH_CHECK(jValue.size(0) == jColInds.size(0));
        int64_t allNumEntries = numEntries.jdata().size(0);

        auto ret = CsrMatrixMultiplication::apply(
                grid_i.impl(), grid_j.impl(),
                coordsI.jdata(), coordsJ.jdata(), iValue.jdata(), jValue.jdata(), iRowPtr.jdata(), jRowPtr.jdata(),
                iColInds.jdata(), jColInds.jdata(), indexMap.jdata(), allNumEntries);

        return numEntries.jagged_like(ret[0].view({-1, 1}));
    });

    m.def ("qg_building", [](const fvdb::GridBatch& grid,
                             const fvdb::JaggedTensor& pts, const fvdb::JaggedTensor& ptsKernel,
                             const fvdb::JaggedTensor& gridKernel, const fvdb::JaggedTensor& gradKernelPts,
                             bool grad) -> std::vector<fvdb::JaggedTensor> {
        CHECK_DEVICE(grid, pts);
        TORCH_CHECK(gradKernelPts.dim() == 3);
        TORCH_CHECK(gridKernel.size(0) == grid.total_voxels(), "grid_kernel must have number of voxels");
        auto ret = QgBuilding::apply(
                grid.impl(), pts, ptsKernel.jdata(), gridKernel.jdata(), gradKernelPts.jdata(), grad);
        return {pts.jagged_like(ret[0]), pts.jagged_like(ret[1])};
    }, py::arg("grid"), py::arg("pts"), py::arg("pts_kernel"), py::arg("grid_kernel"),
          py::arg("grad_kernel_pts"), py::arg("grad"));
}
