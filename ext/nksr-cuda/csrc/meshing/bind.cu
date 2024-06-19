#include <torch/extension.h>
#include <pybind11/stl.h>
#include "functions.h"
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include "inds.cuh"
#include "mgrid_ijk.cuh"


void pybind_meshing(py::module& m) {
    m.def("primal_cube_graph", [](const fvdb::GridBatch& primal, 
                                  const fvdb::GridBatch& dual) -> torch::Tensor {
        TORCH_CHECK(primal.grid_count() == 1, "Primal grid must have batch size 1!");
        TORCH_CHECK(dual.grid_count() == 1, "Dual grid must have batch size 1!");

        int64_t nPrimal = primal.total_voxels();
        torch::Tensor graph = torch::zeros({nPrimal, 8}, torch::TensorOptions().dtype(torch::kLong).device(primal.device()));

        FVDB_DISPATCH_KERNEL_DEVICE(primal.device(), [&]() {
            dispatchPrimalCubeGraph<DeviceTag>(
                *primal.impl(), *dual.impl(), graph);
        });

        return graph;
    });

    m.def("dual_cube_graph", [](const std::vector<torch::optional<fvdb::GridBatch>>& primalGrids, 
                                const fvdb::GridBatch& corner) -> torch::Tensor {
        TORCH_CHECK(corner.grid_count() == 1, "Corner grid must have batch size 1!");

        int64_t currentBaseIdx = 0;

        int64_t nCorner = corner.total_voxels();
        torch::Tensor graph = torch::full({nCorner, 8}, -1, torch::TensorOptions().dtype(
            torch::kLong).device(corner.device()));

        for (int l = 0; l < primalGrids.size(); ++l) {
            if (!primalGrids[l].has_value())
                continue;
                
            const auto primalSVH = primalGrids[l].value();
            if (primalSVH.total_voxels() <= 0)
                continue;

            FVDB_DISPATCH_KERNEL_DEVICE(primalSVH.device(), [&]() {
                switch (l) {
                    case 0:
                        dispatchDualCubeGraphLayer<DeviceTag, 1>(*primalSVH.impl(), 
                            *corner.impl(), currentBaseIdx, graph); break;
                    case 1:
                        dispatchDualCubeGraphLayer<DeviceTag, 2>(*primalSVH.impl(), 
                            *corner.impl(), currentBaseIdx, graph); break;
                    case 2:
                        dispatchDualCubeGraphLayer<DeviceTag, 4>(*primalSVH.impl(), 
                            *corner.impl(), currentBaseIdx, graph); break;
                    case 3:
                        dispatchDualCubeGraphLayer<DeviceTag, 8>(*primalSVH.impl(), 
                            *corner.impl(), currentBaseIdx, graph); break;
                    default:
                        throw std::runtime_error("DMC with large stride not instantiated!");
                }
            });

            currentBaseIdx += primalSVH.total_voxels();
        }

        torch::Tensor graphMask = torch::all(graph != -1, 1);
        return graph.index({graphMask});

    });

    m.def("build_flattened_grid", [](const fvdb::GridBatch& thisGrid,
                                     const torch::optional<fvdb::GridBatch>& childGrid,
                                     bool conforming) -> fvdb::GridBatch {
        fvdb::GridBatch flattenedGrid(thisGrid.device(), thisGrid.is_mutable());
        fvdb::JaggedTensor ijk;
        FVDB_DISPATCH_KERNEL_DEVICE(thisGrid.device(), [&]() {
            ijk = dispatchFlattenedGridIJK<DeviceTag>(thisGrid.impl(), 
                childGrid.has_value() ? childGrid->impl() : nullptr, conforming);
        });
        flattenedGrid.set_from_ijk(
            ijk, torch::zeros(3, torch::kInt32), torch::zeros(3, torch::kInt32),
            thisGrid.voxel_sizes(), thisGrid.origins()
        );
        return flattenedGrid;
    });

    m.def("marching_cubes", [](const torch::Tensor &cubeCornerInds,
                               const torch::Tensor &cornerPos,
                               const torch::Tensor &cornerValue) {
        if (cubeCornerInds.device().is_cuda()) {
            return MarchingCubesCUDA(cubeCornerInds, cornerPos, cornerValue);
        } else {
            return MarchingCubesCPU(cubeCornerInds, cornerPos, cornerValue);
        }
    }, "Marching cubes on a prebuilt graph.");
}
