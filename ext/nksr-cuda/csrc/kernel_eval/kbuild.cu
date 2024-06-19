#include "functions.h"
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>


template <typename GridType, typename ScalarT, typename IndexT, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void kBuildingCallback(
        int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor,
        const TensorAccessor<ScalarT, 2> kernel,
        const TensorAccessor<IndexT, 2> indexMap,   // long Tensor (I, 125)
        TensorAccessor<ScalarT, 1> outMatrix,
        TensorAccessor<ScalarT, 3> dummy3) {

    const nanovdb::NanoGrid<GridType>* grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];

    const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);
    auto acc = grid->getAccessor();

    fvdb::detail::VoxelCoordTransform transform = gridAccessor.primalTransform(batchIdx);

    if (!leaf.isActive(voxelIdx)) {
        return;
    }

    const int64_t offsetI = baseOffset + leaf.getValue(voxelIdx) - 1;
    nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);

#pragma unroll
    for (auto jt = NNIterator<3, ScalarT>(ijk); jt.isValid(); ++jt) {
        if (!acc.isActive(*jt)) {
            continue;
        }
        nanovdb::Coord diffIJ = (*jt) - ijk;
        const int64_t offsetJ = acc.getValue(*jt) - 1 + baseOffset;

        // Evaluate kernel K(i, j)
        ScalarT ijF = 0.0, ijBk, ijDk;
        nanovdb::math::Vec3<ScalarT> gradIjF(0.0), ijDb(0.0);
        kernel_grad_evaluation_fwd(
                offsetJ, offsetI, transform.scale<ScalarT>()[0],
                (ScalarT) diffIJ[0], (ScalarT) diffIJ[1], (ScalarT) diffIJ[2],
                kernel, kernel, dummy3,
                false, ijF, gradIjF, ijBk, ijDk, ijDb);

        int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta(diffIJ);
        IndexT outMatrixIdx = indexMap[offsetI][indexColIdx];

        outMatrix[outMatrixIdx] = ijF;
    }

}

template <typename GridType, typename ScalarT, c10::DeviceType DeviceTag, typename IndexT, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void kBuildingBackwardCallback(
        int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor,
        const TensorAccessor<ScalarT, 2> kernel,
        const TensorAccessor<IndexT, 2> indexMap,   // long Tensor (I, 125)
        const TensorAccessor<ScalarT, 1> gradOutMatrix,
        TensorAccessor<ScalarT, 2> gradKernel,
        TensorAccessor<ScalarT, 3> dummy3) {

    const nanovdb::NanoGrid<GridType>* grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];

    const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);
    auto acc = grid->getAccessor();

    fvdb::detail::VoxelCoordTransform transform = gridAccessor.primalTransform(batchIdx);

    if (!leaf.isActive(voxelIdx)) {
        return;
    }

    const int64_t offsetI = baseOffset + leaf.getValue(voxelIdx) - 1;
    nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);

#pragma unroll
    for (auto jt = NNIterator<3, ScalarT>(ijk); jt.isValid(); ++jt) {
        if (!acc.isActive(*jt)) {
            continue;
        }
        nanovdb::Coord diffIJ = (*jt) - ijk;
        const int64_t offsetJ = acc.getValue(*jt) - 1 + baseOffset;

        // Evaluate kernel K(i, j)
        ScalarT ijF = 0.0, ijBk, ijDk;
        nanovdb::math::Vec3<ScalarT> gradIjF(0.0), ijDb(0.0);
        kernel_grad_evaluation_fwd(
                offsetJ, offsetI, transform.scale<ScalarT>()[0],
                (ScalarT) diffIJ[0], (ScalarT) diffIJ[1], (ScalarT) diffIJ[2],
                kernel, kernel, dummy3,
                false, ijF, gradIjF, ijBk, ijDk, ijDb);

        int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta(diffIJ);
        IndexT outMatrixIdx = indexMap[offsetI][indexColIdx];

        auto dummy2 = gradKernel;
        auto dummy1 = gradOutMatrix;
        kernel_grad_evaluation_bwd<DeviceTag, ScalarT, true>(
                offsetJ, offsetI,
                kernel, kernel, dummy3 /* useless*/ ,
                gradOutMatrix, dummy2 /* useless*/ , false, 1.0,
                gradKernel, gradKernel, dummy1 /* useless*/ , dummy3 /* useless*/ ,
                outMatrixIdx, 1.0, gradIjF /* useless*/ ,
                ijF, gradIjF, ijBk, ijDk, ijDb);
    }

}

template <c10::DeviceType DeviceTag>
void dispatchKBuilding(
        const fvdb::detail::GridBatchImpl &batchHdl,
        const torch::Tensor &kernel,
        const torch::Tensor &indexMap,
        torch::Tensor &outMatrix) {

    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(kernel);
    batchHdl.checkDevice(indexMap);

    torch::Tensor dummy3 = torch::empty({0, 0, 0}, kernel.options());

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "KBuilding", [&]() {
            auto kernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(kernel);
            auto outMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(outMatrix);
            auto dummy3Acc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(dummy3);
            NKSR_DISPATCH_INTEGER_TYPES(indexMap.scalar_type(), [&]() {
                auto indexMapAcc = fvdb::tensorAccessor<DeviceTag, index_t, 2>(indexMap);
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                            fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
                        kBuildingCallback<GridType, scalar_t, index_t, fvdb::TorchRAcc32>(
                            batchIdx, leafIdx, voxelIdx, gridAccessor, kernelAcc, indexMapAcc, outMatrixAcc, dummy3Acc);
                    };
                    fvdb::forEachVoxelCUDA<GridType>(128, 1, batchHdl, cb);
                } else {
                    auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                            fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
                        kBuildingCallback<GridType, scalar_t, index_t, fvdb::TorchAcc>(
                            batchIdx, leafIdx, voxelIdx, gridAccessor, kernelAcc, indexMapAcc, outMatrixAcc, dummy3Acc);
                    };
                    fvdb::forEachVoxelCPU<GridType>(1, batchHdl, cb);
                }
            });
        });
    });
}

template <c10::DeviceType DeviceTag>
void dispatchKBuildingBackward(
        const fvdb::detail::GridBatchImpl &batchHdl,
        const torch::Tensor &kernel,
        const torch::Tensor &indexMap,
        const torch::Tensor &gradOutMatrix,
        torch::Tensor &gradKernel) {

    batchHdl.checkNonEmptyGrid();
    torch::Tensor dummy3 = torch::empty({0, 0, 0}, kernel.options());

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "KBuildingBackward", [&]() {
            auto kernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(kernel);
            auto gradOutMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradOutMatrix);
            auto gradKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradKernel);
            auto dummy3Acc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(dummy3);
            NKSR_DISPATCH_INTEGER_TYPES(indexMap.scalar_type(), [&]() {
                auto indexMapAcc = fvdb::tensorAccessor<DeviceTag, index_t, 2>(indexMap);
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                            fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
                        kBuildingBackwardCallback<GridType, scalar_t, DeviceTag, index_t, fvdb::TorchRAcc32>(
                            batchIdx, leafIdx, voxelIdx, gridAccessor, kernelAcc, indexMapAcc, gradOutMatrixAcc, 
                            gradKernelAcc, dummy3Acc);
                    };
                    fvdb::forEachVoxelCUDA<GridType>(512, 1, batchHdl, cb);
                } else {
                    auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                            fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
                        kBuildingBackwardCallback<GridType, scalar_t, DeviceTag, index_t, fvdb::TorchAcc>(
                            batchIdx, leafIdx, voxelIdx, gridAccessor, kernelAcc, indexMapAcc, gradOutMatrixAcc, 
                            gradKernelAcc, dummy3Acc);
                    };
                    fvdb::forEachVoxelCPU<GridType>(1, batchHdl, cb);
                }
            });
        });
    });
}

variable_list KBuilding::forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
        Variable kernel, Variable indexMap, int64_t numEntries) {
    
    ctx->saved_data["grid"] = grid;
    ctx->saved_data["kernel"] = kernel;
    ctx->saved_data["indexMap"] = indexMap;

    // Prepare output
    auto opts = torch::TensorOptions().dtype(kernel.dtype())
            .device(kernel.device());
    torch::Tensor outMatrix = torch::zeros(numEntries, opts);

    FVDB_DISPATCH_KERNEL_DEVICE(kernel.device(), [&]() {
        dispatchKBuilding<DeviceTag>(*grid, kernel, indexMap, outMatrix);
    });
    return {outMatrix};

}

variable_list KBuilding::backward(AutogradContext *ctx, variable_list grad_output) {
    // Use data saved in forward
    auto grid = ctx->saved_data["grid"].toCustomClass<fvdb::detail::GridBatchImpl>();
    Variable kernel = ctx->saved_data["kernel"].toTensor();
    Variable indexMap = ctx->saved_data["indexMap"].toTensor();

    // Prepare grad input
    Variable gradOutMatrix = grad_output.at(0);

    // Prepare output
    torch::Tensor gradKernel = torch::zeros_like(kernel);

    FVDB_DISPATCH_KERNEL_DEVICE(kernel.device(), [&]() {
        dispatchKBuildingBackward<DeviceTag>(*grid, kernel, indexMap, gradOutMatrix, gradKernel);
    });
    return {torch::Tensor(), gradKernel, torch::Tensor(), torch::Tensor()};
}
