#include "mc_data.h"
#include "../common/iter_util.h"

template <typename GridType, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void primalCubeGraphCallback(
        int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> primalAcc,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> dualAcc,
        TensorAccessor<int64_t, 2> graph) {

    const nanovdb::NanoGrid<GridType>* iGrid = primalAcc.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = \
        iGrid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t pOffset = primalAcc.voxelOffset(batchIdx);

    auto dAcc = dualAcc.grid(batchIdx)->getAccessor();
    const int64_t dOffset = dualAcc.voxelOffset(batchIdx);

    if (leaf.isActive(voxelIdx)) {
        nanovdb::Coord primalCoord = leaf.offsetToGlobalCoord(voxelIdx);
#pragma unroll
        for (int offset = 0; offset < 8; ++offset) {
            const auto& dualCoord = primalCoord + nanovdb::Coord(
                    cubeRelTable[offset][0], cubeRelTable[offset][1], cubeRelTable[offset][2]);
            graph[leaf.getValue(voxelIdx) - 1 + pOffset][offset] = dAcc.getValue(dualCoord) - 1 + dOffset;
        }
    }

}

template <typename GridType, int Stride, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void dualCubeGraphLayerCallback(
        int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> primalAcc,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> dualAcc,
        int64_t baseIdx,
        TensorAccessor<int64_t, 2> graph) {

    const nanovdb::NanoGrid<GridType>* iGrid = primalAcc.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = \
        iGrid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t pOffset = primalAcc.voxelOffset(batchIdx);

    auto dAcc = dualAcc.grid(batchIdx)->getAccessor();
    const int64_t dOffset = dualAcc.voxelOffset(batchIdx);

    if (leaf.isActive(voxelIdx)) {
        nanovdb::Coord primalCoord = leaf.offsetToGlobalCoord(voxelIdx);
#pragma unroll
        for (auto cubeIt = CubeFaceIterator<Stride>(primalCoord); cubeIt.isValid(); cubeIt++) {
            if (!dAcc.isActive(*cubeIt)) {
                continue;
            }
            int64_t vi = dAcc.getValue(*cubeIt) - 1 + dOffset;
            for (int ai = 0; ai < cubeIt.getAccCount(); ++ai) {
                graph[vi][cubeIt.getAccInds(ai)] = leaf.getValue(voxelIdx) - 1 + baseIdx + pOffset;
            }
        }
    }
}

template <c10::DeviceType DeviceTag>
void dispatchPrimalCubeGraph(const fvdb::detail::GridBatchImpl& primalHandle,
                             const fvdb::detail::GridBatchImpl& dualHandle,
                             torch::Tensor& graph) {

    auto graphAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(graph);
    FVDB_DISPATCH_GRID_TYPES(primalHandle, [&]() {
        auto dualAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(dualHandle);
        if constexpr (DeviceTag == torch::kCUDA) {
            auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                    fvdb::detail::GridBatchImpl::Accessor<GridType> primalAcc) {
                primalCubeGraphCallback<GridType, fvdb::TorchRAcc32>(
                    batchIdx, leafIdx, voxelIdx, primalAcc, dualAcc, graphAcc);
            };
            fvdb::forEachVoxelCUDA<GridType>(128, 1, primalHandle, cb);
        } else {
            auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                    fvdb::detail::GridBatchImpl::Accessor<GridType> primalAcc) {
                primalCubeGraphCallback<GridType, fvdb::TorchAcc>(
                    batchIdx, leafIdx, voxelIdx, primalAcc, dualAcc, graphAcc);
            };
            fvdb::forEachVoxelCPU<GridType>(1, primalHandle, cb);
        }
    });
}

template <c10::DeviceType DeviceTag, int Stride>
void dispatchDualCubeGraphLayer(const fvdb::detail::GridBatchImpl& primalHandle,
                                const fvdb::detail::GridBatchImpl& dualHandle,
                                int64_t baseIdx,
                                torch::Tensor& graph) {

    auto graphAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(graph);
    FVDB_DISPATCH_GRID_TYPES(primalHandle, [&]() {
        auto dualAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(dualHandle);
        if constexpr (DeviceTag == torch::kCUDA) {
            auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                    fvdb::detail::GridBatchImpl::Accessor<GridType> primalAcc) {
                dualCubeGraphLayerCallback<GridType, Stride, fvdb::TorchRAcc32>(
                    batchIdx, leafIdx, voxelIdx, primalAcc, dualAcc, baseIdx, graphAcc);
            };
            fvdb::forEachVoxelCUDA<GridType>(128, 1, primalHandle, cb);
        } else {
            auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                    fvdb::detail::GridBatchImpl::Accessor<GridType> primalAcc) {
                dualCubeGraphLayerCallback<GridType, Stride, fvdb::TorchAcc>(
                    batchIdx, leafIdx, voxelIdx, primalAcc, dualAcc, baseIdx, graphAcc);
            };
            fvdb::forEachVoxelCPU<GridType>(1, primalHandle, cb);
        }
    });
}
