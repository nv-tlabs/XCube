#include "functions.h"
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

// Get rid of fp64 operations by re-writing them
template <typename scalar_t>
_CPU_AND_GPU_CODE_ __forceinline__ nanovdb::Coord roundVec(const nanovdb::math::Vec3<scalar_t>& vec) {
    return vec.round();
}

template <>
_CPU_AND_GPU_CODE_ __forceinline__ nanovdb::Coord roundVec(const nanovdb::math::Vec3<float>& vec) {
    return nanovdb::Coord(
            (int32_t) lroundf(vec[0]),
            (int32_t) lroundf(vec[1]),
            (int32_t) lroundf(vec[2]));
}

// Iterates over each voxel in the grid, and for each voxel each neighbor in the 1-ring.
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void forEachJaggedElementAnd1NeighborCPU(const fvdb::JaggedTensor& jaggedTensor, Func func, Args... args) {
    const int64_t numElements = jaggedTensor.element_count();
    auto jaggedAcc = jaggedTensor.accessor<ScalarT, NDIMS>();

    for (int64_t elementIdx = 0; elementIdx < numElements; elementIdx += 1) {
        const int64_t batchIdx = jaggedAcc.batchIdx(elementIdx);

        for (int64_t neighborIdx = 0; neighborIdx < NNIterator<3, float>::total(); neighborIdx++) {
            func(batchIdx, elementIdx, neighborIdx, jaggedAcc, args...);
        }
    }
}

template <int32_t NDIMS, typename ScalarT, typename Func, typename... Args>
__global__ void forEachJaggedElementAnd1NeighborCUDAKernel(fvdb::JaggedRAcc32<ScalarT, NDIMS> jaggedAcc, Func func, Args... args) {
    int64_t elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t neighborIdx = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t numElements = jaggedAcc.elementCount();
    if (elementIdx >= numElements || neighborIdx >= NNIterator<3, ScalarT>::total()) {
        return;
    }

    const int64_t batchIdx = jaggedAcc.batchIdx(elementIdx);
    func(batchIdx, elementIdx, neighborIdx, jaggedAcc, args...);
}

template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void forEachJaggedElementAnd1NeighborCUDA(const fvdb::JaggedTensor& jaggedTensor, Func func, Args... args) {
    const unsigned nThreadsX = 16;
    const unsigned nThreadsY = 10;

    TORCH_CHECK(jaggedTensor.device().has_index(), "JaggedTensor device must have an index");
    c10::cuda::CUDAGuard deviceGuard(jaggedTensor.device());
    const int64_t numElements = jaggedTensor.element_count();

    const int64_t PCOUNT = numElements;
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);
    const int64_t ICOUNT = NNIterator<3, ScalarT>::total();
    const int64_t NBLOCKSY = fvdb::GET_BLOCKS(ICOUNT, nThreadsY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(nThreadsX, nThreadsY, 1);

    if (numElements > 0) {
        forEachJaggedElementAnd1NeighborCUDAKernel<NDIMS, ScalarT, Func, Args...><<<nblocks, nthreads>>>(
            jaggedTensor.packed_accessor32<ScalarT, NDIMS, torch::RestrictPtrTraits>(), func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <typename IndexT, typename GridType, typename ScalarT, c10::DeviceType DeviceTag, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void matrixBuildingCallback(
        int32_t bidx, int32_t eidx, int32_t nidx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> gridIAcc,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> gridJAcc,
        JaggedAccessor<ScalarT, 2> ptsPos,
        const TensorAccessor<ScalarT, 2> ptsKernelI,
        const TensorAccessor<ScalarT, 2> ptsKernelJ,
        const TensorAccessor<ScalarT, 2> iKernel,
        const TensorAccessor<ScalarT, 2> jKernel,
        const TensorAccessor<ScalarT, 3> gradPtsKernelPosI,
        const TensorAccessor<ScalarT, 3> gradPtsKernelPosJ,
        const TensorAccessor<IndexT, 2> indexMap,   // long Tensor (I, 125)
        bool grad,          // Build GTG or QTQ
        TensorAccessor<ScalarT, 1> outMatrix) {

    using NN = NNIterator<3, ScalarT>;

    const nanovdb::NanoGrid<GridType>* gridI = gridIAcc.grid(bidx);
    const int64_t baseOffsetI = gridIAcc.voxelOffset(bidx);
    auto iAcc = gridI->getAccessor();
    fvdb::detail::VoxelCoordTransform transformI = gridIAcc.primalTransform(bidx);

    const auto& ptsPosTh = ptsPos.data();
    const nanovdb::math::Vec3<ScalarT> piLocal = transformI.apply<ScalarT>(ptsPosTh[eidx]);
    auto it = NN(roundVec(piLocal), nidx);

    if (!iAcc.isActive(*it)) {
        return;
    }
    const int64_t offsetI = iAcc.getValue(*it) - 1 + baseOffsetI;

    // Evaluate kernel K(k, i)
    ScalarT kiF = 0.0, kiBk, kiDk;
    nanovdb::math::Vec3<ScalarT> gradKiF(0.0), kiDb(0.0);
    kernel_grad_evaluation_fwd(
            offsetI, eidx, transformI.scale<ScalarT>()[0],
            piLocal[0] - (ScalarT) (*it)[0],
            piLocal[1] - (ScalarT) (*it)[1],
            piLocal[2] - (ScalarT) (*it)[2],
            ptsKernelI, iKernel, gradPtsKernelPosI,
            grad, kiF, gradKiF, kiBk, kiDk, kiDb);

    const nanovdb::NanoGrid<GridType>* gridJ = gridJAcc.grid(bidx);
    const int64_t baseOffsetJ = gridJAcc.voxelOffset(bidx);
    auto jAcc = gridJ->getAccessor();
    fvdb::detail::VoxelCoordTransform transformJ = gridJAcc.primalTransform(bidx);

    const nanovdb::math::Vec3<ScalarT> pjLocal = transformJ.apply<ScalarT>(ptsPosTh[eidx]);

    // Iterate over index [j] (to be put into one kernel execution)
#pragma unroll
    for (auto jt = NNIterator<3, ScalarT>(roundVec(pjLocal)); jt.isValid(); ++jt) {
        if (!jAcc.isActive(*jt)) {
            continue;
        }
        const int64_t offsetJ = jAcc.getValue(*jt) - 1 + baseOffsetJ;

        // Evaluate kernel K(k, j)
        ScalarT kjF = 0.0, kjBk, kjDk;
        nanovdb::math::Vec3<ScalarT> gradKjF(0.0), kjDb(0.0);
        kernel_grad_evaluation_fwd(
                offsetJ, eidx, transformJ.scale<ScalarT>()[0],
                pjLocal[0] - (ScalarT) (*jt)[0],
                pjLocal[1] - (ScalarT) (*jt)[1],
                pjLocal[2] - (ScalarT) (*jt)[2],
                ptsKernelJ, jKernel, gradPtsKernelPosJ,
                grad, kjF, gradKjF, kjBk, kjDk, kjDb);

        // Put K(k,i)*K(k,j) into Mat[offsetI, offsetJ], aka. -> outMatrix[outMatrixIdx]
        ScalarT outVal;
        if (!grad) { outVal = kiF * kjF; }
        else { outVal = gradKiF.template dot(gradKjF); }

        nanovdb::Coord iC = roundVec(transformJ.apply(transformI.applyInv(it->asVec3s())));
        int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta((*jt) - iC);
        IndexT outMatrixIdx = indexMap[offsetI][indexColIdx];

        atomicAddIfGPU<DeviceTag>(&outMatrix[outMatrixIdx], outVal);
    }
}


template <typename IndexT, typename GridType, typename ScalarT, c10::DeviceType DeviceTag, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void matrixBuildingBackwardCallback(
        int32_t bidx, int32_t eidx, int32_t nidx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> gridIAcc,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> gridJAcc,
        JaggedAccessor<ScalarT, 2> ptsPos,
        const TensorAccessor<ScalarT, 2> ptsKernelI,
        const TensorAccessor<ScalarT, 2> ptsKernelJ,
        const TensorAccessor<ScalarT, 2> iKernel,
        const TensorAccessor<ScalarT, 2> jKernel,
        const TensorAccessor<ScalarT, 3> gradPtsKernelPosI,
        const TensorAccessor<ScalarT, 3> gradPtsKernelPosJ,
        const TensorAccessor<IndexT, 2> indexMap,
        bool grad,
        const TensorAccessor<ScalarT, 1> gradOutMatrix,
        TensorAccessor<ScalarT, 2> gradPtsKernelI,
        TensorAccessor<ScalarT, 2> gradPtsKernelJ,
        TensorAccessor<ScalarT, 2> gradIKernel,
        TensorAccessor<ScalarT, 2> gradJKernel,
        TensorAccessor<ScalarT, 3> gradGradPtsKernelPosI,
        TensorAccessor<ScalarT, 3> gradGradPtsKernelPosJ) {

    using NN = NNIterator<3, ScalarT>;

    const nanovdb::NanoGrid<GridType>* gridI = gridIAcc.grid(bidx);
    const int64_t baseOffsetI = gridIAcc.voxelOffset(bidx);
    auto iAcc = gridI->getAccessor();
    fvdb::detail::VoxelCoordTransform transformI = gridIAcc.primalTransform(bidx);

    const auto& ptsPosTh = ptsPos.data();
    const nanovdb::math::Vec3<ScalarT> piLocal = transformI.apply<ScalarT>(ptsPosTh[eidx]);
    auto it = NN(roundVec(piLocal), nidx);

    if (!iAcc.isActive(*it)) {
        return;
    }
    const int64_t offsetI = iAcc.getValue(*it) - 1 + baseOffsetI;

    // Evaluate kernel K(k, i)
    ScalarT kiF = 0.0, kiBk, kiDk;
    nanovdb::math::Vec3<ScalarT> gradKiF(0.0), kiDb(0.0);
    kernel_grad_evaluation_fwd(
            offsetI, eidx, transformI.scale<ScalarT>()[0],
            piLocal[0] - (ScalarT) (*it)[0],
            piLocal[1] - (ScalarT) (*it)[1],
            piLocal[2] - (ScalarT) (*it)[2],
            ptsKernelI, iKernel, gradPtsKernelPosI,
            grad, kiF, gradKiF, kiBk, kiDk, kiDb);

    const nanovdb::NanoGrid<GridType>* gridJ = gridJAcc.grid(bidx);
    const int64_t baseOffsetJ = gridJAcc.voxelOffset(bidx);
    auto jAcc = gridJ->getAccessor();
    fvdb::detail::VoxelCoordTransform transformJ = gridJAcc.primalTransform(bidx);

    const nanovdb::math::Vec3<ScalarT> pjLocal = transformJ.apply<ScalarT>(ptsPosTh[eidx]);

    // Iterate over index [j] (to be put into one kernel execution)
#pragma unroll
    for (auto jt = NNIterator<3, ScalarT>(roundVec(pjLocal)); jt.isValid(); ++jt) {
        if (!jAcc.isActive(*jt)) {
            continue;
        }
        const int64_t offsetJ = jAcc.getValue(*jt) - 1 + baseOffsetJ;

        // Evaluate kernel K(k, j)
        ScalarT kjF = 0.0, kjBk, kjDk;
        nanovdb::math::Vec3<ScalarT> gradKjF(0.0), kjDb(0.0);
        kernel_grad_evaluation_fwd(
                offsetJ, eidx, transformJ.scale<ScalarT>()[0],
                pjLocal[0] - (ScalarT) (*jt)[0],
                pjLocal[1] - (ScalarT) (*jt)[1],
                pjLocal[2] - (ScalarT) (*jt)[2],
                ptsKernelJ, jKernel, gradPtsKernelPosJ,
                grad, kjF, gradKjF, kjBk, kjDk, kjDb);

        nanovdb::Coord iC = roundVec(transformJ.apply(transformI.applyInv(it->asVec3s())));
        int indexColIdx = NNIterator<5, ScalarT>::CountFromDelta((*jt) - iC);
        IndexT outMatrixIdx = indexMap[offsetI][indexColIdx];

        auto dummyAcc2 = ptsKernelI;
        auto dummyAcc1 = gradOutMatrix;

        // Backward (from I)
        kernel_grad_evaluation_bwd<DeviceTag, ScalarT, true>(
                offsetI, eidx,
                ptsKernelI, iKernel, gradPtsKernelPosI,
                gradOutMatrix, dummyAcc2, grad, 1.0,
                gradPtsKernelI, gradIKernel, dummyAcc1, gradGradPtsKernelPosI,
                outMatrixIdx, kjF, gradKjF,
                kiF, gradKiF, kiBk, kiDk, kiDb);

        // Backward (from J)
        kernel_grad_evaluation_bwd<DeviceTag, ScalarT, true>(
                offsetJ, eidx,
                ptsKernelJ, jKernel, gradPtsKernelPosJ,
                gradOutMatrix, dummyAcc2, grad, 1.0,
                gradPtsKernelJ, gradJKernel, dummyAcc1, gradGradPtsKernelPosJ,
                outMatrixIdx, kiF, gradKiF,
                kjF, gradKjF, kjBk, kjDk, kjDb);
    }
}

template <c10::DeviceType DeviceTag>
void dispatchMatrixBuilding(const fvdb::detail::GridBatchImpl& gridIHandle,
                            const fvdb::detail::GridBatchImpl& gridJHandle,
                            const fvdb::JaggedTensor& ptsPos,
                            const torch::Tensor& ptsKernelI,
                            const torch::Tensor& ptsKernelJ,
                            const torch::Tensor& iKernel,
                            const torch::Tensor& jKernel,
                            const torch::Tensor& gradPtsKernelPosI,
                            const torch::Tensor& gradPtsKernelPosJ,
                            const torch::Tensor& indexMap,
                            bool grad,
                            torch::Tensor& outMatrix) {
    
    gridIHandle.checkDevice(ptsPos);
    gridIHandle.checkDevice(ptsKernelI);
    gridJHandle.checkDevice(ptsPos);
    gridJHandle.checkDevice(ptsKernelJ);

    FVDB_DISPATCH_GRID_TYPES(gridIHandle, [&]() {
        auto gridIAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(gridIHandle);
        auto gridJAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(gridJHandle);

        AT_DISPATCH_FLOATING_TYPES(ptsPos.scalar_type(), "matrixBuilding", [&]() {

            auto ptsKernelIAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernelI);
            auto ptsKernelJAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernelJ);
            auto iKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(iKernel);
            auto jKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(jKernel);
            auto gradPtsKernelPosIAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradPtsKernelPosI);
            auto gradPtsKernelPosJAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradPtsKernelPosJ);
            auto outMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(outMatrix);
            
            NKSR_DISPATCH_INTEGER_TYPES(indexMap.scalar_type(), [&]() {
                auto indexMapAcc = fvdb::tensorAccessor<DeviceTag, index_t, 2>(indexMap);
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t nidx, fvdb::JaggedRAcc32<scalar_t, 2> ptsPosAcc) {
                        matrixBuildingCallback<index_t, GridType, scalar_t, DeviceTag, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(
                            bidx, eidx, nidx, gridIAcc, gridJAcc, ptsPosAcc, ptsKernelIAcc, ptsKernelJAcc, iKernelAcc, jKernelAcc,
                            gradPtsKernelPosIAcc, gradPtsKernelPosJAcc, indexMapAcc, grad, outMatrixAcc);
                    };
                    forEachJaggedElementAnd1NeighborCUDA<scalar_t, 2>(ptsPos, cb);
                } else {
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t nidx, fvdb::JaggedAcc<scalar_t, 2> ptsPosAcc) {
                        matrixBuildingCallback<index_t, GridType, scalar_t, DeviceTag, fvdb::JaggedAcc, fvdb::TorchAcc>(
                            bidx, eidx, nidx, gridIAcc, gridJAcc, ptsPosAcc, ptsKernelIAcc, ptsKernelJAcc, iKernelAcc, jKernelAcc,
                            gradPtsKernelPosIAcc, gradPtsKernelPosJAcc, indexMapAcc, grad, outMatrixAcc);
                    };
                    forEachJaggedElementAnd1NeighborCPU<scalar_t, 2>(ptsPos, cb);
                }
            });

        });
    });
}

template <c10::DeviceType DeviceTag>
void dispatchMatrixBuildingBackward(
        const fvdb::detail::GridBatchImpl& gridIHandle,
        const fvdb::detail::GridBatchImpl& gridJHandle,
        const fvdb::JaggedTensor& ptsPos,
        const torch::Tensor& ptsKernelI,
        const torch::Tensor& ptsKernelJ,
        const torch::Tensor& iKernel,
        const torch::Tensor& jKernel,
        const torch::Tensor& gradPtsKernelPosI,
        const torch::Tensor& gradPtsKernelPosJ,
        const torch::Tensor& indexMap,
        bool grad,
        const torch::Tensor& gradOutMatrix,
        torch::Tensor& gradPtsKernelI,
        torch::Tensor& gradPtsKernelJ,
        torch::Tensor& gradIKernel,
        torch::Tensor& gradJKernel,
        torch::Tensor& gradGradPtsKernelPosI,
        torch::Tensor& gradGradPtsKernelPosJ) {

    gridIHandle.checkDevice(ptsPos);
    gridIHandle.checkDevice(ptsKernelI);
    gridJHandle.checkDevice(ptsPos);
    gridJHandle.checkDevice(ptsKernelJ);

    FVDB_DISPATCH_GRID_TYPES(gridIHandle, [&]() {
        auto gridIAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(gridIHandle);
        auto gridJAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(gridJHandle);

        AT_DISPATCH_FLOATING_TYPES(ptsPos.scalar_type(), "matrixBuildingBackward", [&]() {

            auto ptsKernelIAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernelI);
            auto ptsKernelJAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernelJ);
            auto iKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(iKernel);
            auto jKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(jKernel);
            auto gradPtsKernelPosIAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradPtsKernelPosI);
            auto gradPtsKernelPosJAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradPtsKernelPosJ);
            auto gradOutMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradOutMatrix);
            auto gradPtsKernelIAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradPtsKernelI);
            auto gradPtsKernelJAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradPtsKernelJ);
            auto gradIKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradIKernel);
            auto gradJKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradJKernel);
            auto gradGradPtsKernelPosIAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradGradPtsKernelPosI);
            auto gradGradPtsKernelPosJAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradGradPtsKernelPosJ);

            NKSR_DISPATCH_INTEGER_TYPES(indexMap.scalar_type(), [&]() {
                auto indexMapAcc = fvdb::tensorAccessor<DeviceTag, index_t, 2>(indexMap);
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t nidx, fvdb::JaggedRAcc32<scalar_t, 2> ptsPosAcc) {
                        matrixBuildingBackwardCallback<index_t, GridType, scalar_t, DeviceTag, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(
                            bidx, eidx, nidx, gridIAcc, gridJAcc, ptsPosAcc, ptsKernelIAcc, ptsKernelJAcc, iKernelAcc, jKernelAcc,
                            gradPtsKernelPosIAcc, gradPtsKernelPosJAcc, indexMapAcc, grad, gradOutMatrixAcc,
                            gradPtsKernelIAcc, gradPtsKernelJAcc, gradIKernelAcc, gradJKernelAcc,
                            gradGradPtsKernelPosIAcc, gradGradPtsKernelPosJAcc);
                    };
                    forEachJaggedElementAnd1NeighborCUDA<scalar_t, 2>(ptsPos, cb);
                } else {
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t nidx, fvdb::JaggedAcc<scalar_t, 2> ptsPosAcc) {
                        matrixBuildingBackwardCallback<index_t, GridType, scalar_t, DeviceTag, fvdb::JaggedAcc, fvdb::TorchAcc>(
                            bidx, eidx, nidx, gridIAcc, gridJAcc, ptsPosAcc, ptsKernelIAcc, ptsKernelJAcc, iKernelAcc, jKernelAcc,
                            gradPtsKernelPosIAcc, gradPtsKernelPosJAcc, indexMapAcc, grad, gradOutMatrixAcc,
                            gradPtsKernelIAcc, gradPtsKernelJAcc, gradIKernelAcc, gradJKernelAcc,
                            gradGradPtsKernelPosIAcc, gradGradPtsKernelPosJAcc);
                    };
                    forEachJaggedElementAnd1NeighborCPU<scalar_t, 2>(ptsPos, cb);
                }
            });
        });
    });
}

variable_list MatrixBuilding::forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> iGrid,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> jGrid,
        fvdb::JaggedTensor ptsPos,
        Variable ptsKernelI, Variable ptsKernelJ,
        Variable iKernel, Variable jKernel,
        Variable gradPtsKernelPosI, Variable gradPtsKernelPosJ,
        Variable indexMap,
        bool grad, int64_t numEntries) {
    ctx->saved_data["iGrid"] = iGrid;
    ctx->saved_data["jGrid"] = jGrid;
    ctx->saved_data["ptsPos_data"] = ptsPos.jdata();
    ctx->saved_data["ptsPos_offsets"] = ptsPos.joffsets();
    ctx->saved_data["ptsKernelI"] = ptsKernelI;
    ctx->saved_data["ptsKernelJ"] = ptsKernelJ;
    ctx->saved_data["iKernel"] = iKernel;
    ctx->saved_data["jKernel"] = jKernel;
    ctx->saved_data["gradPtsKernelPosI"] = gradPtsKernelPosI;
    ctx->saved_data["gradPtsKernelPosJ"] = gradPtsKernelPosJ;
    ctx->saved_data["indexMap"] = indexMap;
    ctx->saved_data["grad"] = grad;

    // Prepare output
    auto opts = torch::TensorOptions().dtype(ptsKernelI.dtype())
            .device(ptsKernelI.device());
    torch::Tensor outMatrix = torch::zeros(numEntries, opts);

    FVDB_DISPATCH_KERNEL_DEVICE(ptsPos.device(), [&]() {
        dispatchMatrixBuilding<DeviceTag>(
            *iGrid, *jGrid, ptsPos, ptsKernelI, ptsKernelJ, iKernel, jKernel, 
            gradPtsKernelPosI, gradPtsKernelPosJ, indexMap, grad, outMatrix);
    });

    return {outMatrix};
}

variable_list MatrixBuilding::backward(AutogradContext *ctx, variable_list grad_output) {
    
    auto iGrid = ctx->saved_data["iGrid"].toCustomClass<fvdb::detail::GridBatchImpl>();
    auto jGrid = ctx->saved_data["jGrid"].toCustomClass<fvdb::detail::GridBatchImpl>();
    
    torch::Tensor ptsPos_data = ctx->saved_data["ptsPos_data"].toTensor();
    torch::Tensor ptsPos_offsets = ctx->saved_data["ptsPos_offsets"].toTensor();
    auto ptsPos = fvdb::JaggedTensor::from_data_and_offsets(ptsPos_data, ptsPos_offsets);

    Variable ptsKernelI = ctx->saved_data["ptsKernelI"].toTensor();
    Variable ptsKernelJ = ctx->saved_data["ptsKernelJ"].toTensor();
    Variable iKernel = ctx->saved_data["iKernel"].toTensor();
    Variable jKernel = ctx->saved_data["jKernel"].toTensor();
    Variable gradPtsKernelPosI = ctx->saved_data["gradPtsKernelPosI"].toTensor();
    Variable gradPtsKernelPosJ = ctx->saved_data["gradPtsKernelPosJ"].toTensor();
    Variable indexMap = ctx->saved_data["indexMap"].toTensor();
    bool grad = ctx->saved_data["grad"].toBool();

    // Prepare grad input
    Variable gradOutMatrix = grad_output.at(0);

    // Prepare output
    torch::Tensor gradPtsKernelI = torch::zeros_like(ptsKernelI);
    torch::Tensor gradPtsKernelJ = torch::zeros_like(ptsKernelJ);
    torch::Tensor gradIKernel = torch::zeros_like(iKernel);
    torch::Tensor gradJKernel = torch::zeros_like(jKernel);
    torch::Tensor gradGradPtsKernelPosI = torch::zeros_like(gradPtsKernelPosI);
    torch::Tensor gradGradPtsKernelPosJ = torch::zeros_like(gradPtsKernelPosJ);

    FVDB_DISPATCH_KERNEL_DEVICE(ptsPos.device(), [&]() {
        dispatchMatrixBuildingBackward<DeviceTag>(
            *iGrid, *jGrid, ptsPos, ptsKernelI, ptsKernelJ, iKernel, jKernel, gradPtsKernelPosI, gradPtsKernelPosJ, indexMap,
            grad, gradOutMatrix, gradPtsKernelI, gradPtsKernelJ, gradIKernel, gradJKernel,
            gradGradPtsKernelPosI, gradGradPtsKernelPosJ);
    });

    return {torch::Tensor(), torch::Tensor(),
            torch::Tensor(),
            gradPtsKernelI, gradPtsKernelJ, gradIKernel, gradJKernel,
            gradGradPtsKernelPosI, gradGradPtsKernelPosJ,
            torch::Tensor(), torch::Tensor(), torch::Tensor()};
}
