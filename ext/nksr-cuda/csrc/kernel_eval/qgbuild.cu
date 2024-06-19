#include "functions.h"

#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include "../common/iter_util.h"
#include "../common/platform.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <torch/autograd.h>
#include <ATen/native/cuda/KernelUtils.cuh>

// Feature-VDB
#include <nanovdb/NanoVDB.h>
//#include <utils/cuda/Math.cuh>

using IndexTree = typename nanovdb::NanoTree<nanovdb::ValueIndex>;

template <c10::DeviceType DeviceTag, typename scalar_t>
_CPU_AND_GPU_CODE_ __forceinline__ void atomicIfGPU(scalar_t* tensor, scalar_t value) {
    if constexpr (DeviceTag == torch::kCUDA) {
        gpuAtomicAddNoReturn(tensor, value);
    } else {
        (*tensor) += value;
    }
}

template <c10::DeviceType DeviceTag, typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ __forceinline__ static void putValue(
        TensorAccessor<scalar_t, 2>& out,
        int i, int j, scalar_t f, const nanovdb::math::Vec3<scalar_t>& df) {
    out[i][j] = f;
}

template <c10::DeviceType DeviceTag, typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ __forceinline__ static void putValue(
        TensorAccessor<scalar_t, 3>& out,
        int i, int j, scalar_t f, const nanovdb::math::Vec3<scalar_t>& df) {
    out[i][j][0] = df[0];
    out[i][j][1] = df[1];
    out[i][j][2] = df[2];
}

template <c10::DeviceType DeviceTag, typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ __forceinline__ static void getValue(
        const TensorAccessor<scalar_t, 2>& in,
        int i, int j, scalar_t& f, nanovdb::math::Vec3<scalar_t>& df) {
    f = in[i][j];
}

template <c10::DeviceType DeviceTag, typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ __forceinline__ static void getValue(
        const TensorAccessor<scalar_t, 3>& in,
        int i, int j, scalar_t& f, nanovdb::math::Vec3<scalar_t>& df) {
    df[0] = in[i][j][0];
    df[1] = in[i][j][1];
    df[2] = in[i][j][2];
}

// Dispatch multiplication, templated based on number of tensor dimensions.
template <c10::DeviceType DeviceTag, typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ __forceinline__ scalar_t valMult(
        const TensorAccessor<scalar_t, 1> aVal,
        const TensorAccessor<scalar_t, 1> bVal,
        int32_t a, int b) {
    return aVal[a] * bVal[b];
}

template <c10::DeviceType DeviceTag, typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ __forceinline__ scalar_t valMult(
        const TensorAccessor<scalar_t, 2> aVal,
        const TensorAccessor<scalar_t, 2> bVal,
        int32_t a, int b) {
    return aVal[a][0] * bVal[b][0] + aVal[a][1] * bVal[b][1] + aVal[a][2] * bVal[b][2];
}

template <c10::DeviceType DeviceTag, typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ __forceinline__ void valMultBwd(
        const TensorAccessor<scalar_t, 1> aVal,
        const TensorAccessor<scalar_t, 1> bVal,
        int32_t a, int b,
        TensorAccessor<scalar_t, 1> gradAVal,
        TensorAccessor<scalar_t, 1> gradBVal,
        scalar_t grad_fg) {
    atomicAddIfGPU<DeviceTag, scalar_t>(&gradAVal[a], grad_fg * bVal[b]);
    atomicAddIfGPU<DeviceTag, scalar_t>(&gradBVal[b], grad_fg * aVal[a]);
}

template <c10::DeviceType DeviceTag, typename scalar_t, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ __forceinline__ void valMultBwd(
        const TensorAccessor<scalar_t, 2> aVal,
        const TensorAccessor<scalar_t, 2> bVal,
        int32_t a, int b,
        TensorAccessor<scalar_t, 2> gradAVal,
        TensorAccessor<scalar_t, 2> gradBVal,
        scalar_t grad_fg) {
    atomicAddIfGPU<DeviceTag, scalar_t>(&gradAVal[a][0], grad_fg * bVal[b][0]);
    atomicAddIfGPU<DeviceTag, scalar_t>(&gradAVal[a][1], grad_fg * bVal[b][1]);
    atomicAddIfGPU<DeviceTag, scalar_t>(&gradAVal[a][2], grad_fg * bVal[b][2]);
    atomicAddIfGPU<DeviceTag, scalar_t>(&gradBVal[b][0], grad_fg * aVal[a][0]);
    atomicAddIfGPU<DeviceTag, scalar_t>(&gradBVal[b][1], grad_fg * aVal[a][1]);
    atomicAddIfGPU<DeviceTag, scalar_t>(&gradBVal[b][2], grad_fg * aVal[a][2]);
}

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


/**
 * CSR Adjoint Matrix product. Given a CSR matrix A and a CSR matrix B, this function computes
 * A.T @ B.
 */

template <c10::DeviceType DeviceTag, typename GridType, typename scalar_t, int Dim, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void csrMatrixMultiplication(
        int32_t bidx, // The batch index
        int32_t eidx, // The element index
        int32_t nidx, // The neighbor index
        const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessorI, const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessorJ,
        const TensorAccessor<int64_t, 2> iCoords, const TensorAccessor<int64_t, 2> jCoords,
        // The descriptions of the two CSR matrices are in the next 6 arguments.
        const TensorAccessor<scalar_t, Dim> iValue, const TensorAccessor<scalar_t, Dim> jValue,
        const TensorAccessor<int, 1> iRowPtr, const TensorAccessor<int, 1> jRowPtr,
        const TensorAccessor<int, 1> iColInds, const TensorAccessor<int, 1> jColInds,
        const TensorAccessor<int, 2> indexMap, TensorAccessor<scalar_t, 1> outMatrix) {

    if (eidx >= iRowPtr.size(0) - 1) {
        return;
    }

    fvdb::detail::VoxelCoordTransform transformI = batchAccessorI.primalTransform(bidx);
    fvdb::detail::VoxelCoordTransform transformJ = batchAccessorJ.primalTransform(bidx);

    int a = nidx + iRowPtr[eidx];
    if (a >= iRowPtr[eidx + 1]) return;

    // For each element in the row of A, iterate through all elements in the row of B.
    for (int b = jRowPtr[eidx]; b < jRowPtr[eidx + 1]; ++b) {
        float fg = valMult<DeviceTag, scalar_t, TensorAccessor>(iValue, jValue, a, b);

        int offsetI = iColInds[a];
        int offsetJ = jColInds[b];

        nanovdb::Coord iC = roundVec(transformJ.apply(transformI.applyInv(
                nanovdb::math::Vec3<scalar_t>(
                        iCoords[offsetI][0], iCoords[offsetI][1], iCoords[offsetI][2]))));
        int indexColIdx = NNIterator<5, scalar_t>::CountFromDelta(nanovdb::Coord(
                jCoords[offsetJ][0] - iC[0],
                jCoords[offsetJ][1] - iC[1],
                jCoords[offsetJ][2] - iC[2]));
        int outMatrixIdx = indexMap[offsetI][indexColIdx];

        atomicAddIfGPU<DeviceTag, scalar_t>(&outMatrix[outMatrixIdx], fg);
    }
}

template <c10::DeviceType DeviceTag, typename GridType, typename scalar_t, int Dim, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void csrMatrixMultiplicationBackward(
        int32_t bidx, // The batch index
        int32_t eidx, // The element index
        int32_t nidx, // The neighbor index
        const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessorI, const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessorJ,
        const TensorAccessor<int64_t, 2> iCoords, const TensorAccessor<int64_t, 2> jCoords,
        const TensorAccessor<scalar_t, Dim> iValue, const TensorAccessor<scalar_t, Dim> jValue,
        const TensorAccessor<int, 1> iRowPtr, const TensorAccessor<int, 1> jRowPtr,
        const TensorAccessor<int, 1> iColInds, const TensorAccessor<int, 1> jColInds,
        const TensorAccessor<int, 2> indexMap, const TensorAccessor<scalar_t, 1> gradOutMatrix,
        TensorAccessor<scalar_t, Dim> gradIValue, TensorAccessor<scalar_t, Dim> gradJValue) {

    if (eidx >= iRowPtr.size(0) - 1) {
        return;
    }

    fvdb::detail::VoxelCoordTransform transformI = batchAccessorI.primalTransform(bidx);
    fvdb::detail::VoxelCoordTransform transformJ = batchAccessorJ.primalTransform(bidx);

    int a = nidx + iRowPtr[eidx];
    if (a >= iRowPtr[eidx + 1]) return;

    for (int b = jRowPtr[eidx]; b < jRowPtr[eidx + 1]; ++b) {
        int offsetI = iColInds[a];
        int offsetJ = jColInds[b];

        nanovdb::Coord iC = roundVec(transformJ.apply(transformI.applyInv(
                nanovdb::math::Vec3<scalar_t>(
                        iCoords[offsetI][0], iCoords[offsetI][1], iCoords[offsetI][2]))));
        int indexColIdx = NNIterator<5, scalar_t>::CountFromDelta(nanovdb::Coord(
                jCoords[offsetJ][0] - iC[0],
                jCoords[offsetJ][1] - iC[1],
                jCoords[offsetJ][2] - iC[2]));
        int outMatrixIdx = indexMap[offsetI][indexColIdx];

        float gradFg = gradOutMatrix[outMatrixIdx];
        valMultBwd<DeviceTag, scalar_t, TensorAccessor>(iValue, jValue, a, b, gradIValue, gradJValue, gradFg);
    }
}

// Iterates over each voxel in the grid, and for each voxel each neighbor in the 1-ring.
template <typename Func, typename... Args>
void forEachVoxelAnd1NeighborCPU(const fvdb::detail::GridBatchImpl& batchHdl, Func func, Args... args) {
    const int64_t total_voxels = batchHdl.totalVoxels();
    const auto jidx = fvdb::tensorAccessor<torch::kCPU, int16_t, 1>(batchHdl.jidx(false));

    for (int64_t elementIdx = 0; elementIdx < total_voxels; elementIdx++) {
        const int16_t batchIdx = jidx[elementIdx];

        for (int64_t neighborIdx = 0; neighborIdx < NNIterator<3, float>::total(); neighborIdx++) {
            func(batchIdx, elementIdx, neighborIdx, args...);
        }
    }
}

template <typename Func, typename... Args>
__global__ void forEachVoxelAnd1NeighborKernel(const fvdb::TorchRAcc32<int16_t, 1> jidx, Func func, Args... args) {
    int64_t elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t neighborIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (jidx.size(0) > 0) {
        int64_t batchIdx = jidx[elementIdx];
        func(batchIdx, elementIdx, neighborIdx, args...);
    } else {
        func(0, elementIdx, neighborIdx, args...);
    }
}

template <typename Func, typename... Args>
void forEachVoxelAnd1NeighborCUDA(const fvdb::detail::GridBatchImpl& batchHdl, Func func, Args... args) {
    const unsigned nThreadsX = 16;
    const unsigned nThreadsY = 10;
    const auto jidx = fvdb::tensorAccessor<torch::kCUDA, int16_t, 1>(batchHdl.jidx(false));

    const int64_t PCOUNT = batchHdl.totalVoxels();
    const int64_t NBLOCKSX = fvdb::GET_BLOCKS(PCOUNT, nThreadsX);
    const int64_t ICOUNT = NNIterator<3, float>::total();
    const int64_t NBLOCKSY = fvdb::GET_BLOCKS(ICOUNT, nThreadsY);

    dim3 nblocks(NBLOCKSX, NBLOCKSY, 1);
    dim3 nthreads(nThreadsX, nThreadsY, 1);

    forEachVoxelAnd1NeighborKernel<<<nblocks, nthreads>>>(jidx, func, args...);
}

template <c10::DeviceType DeviceTag>
void dispatchCSRMatrixMultiplication(
        const fvdb::detail::GridBatchImpl& iGrid,
        const fvdb::detail::GridBatchImpl& jGrid,
        const torch::Tensor& iCoords, const torch::Tensor& jCoords,
        const torch::Tensor& iValue, const torch::Tensor& jValue,
        const torch::Tensor& iRowPtr, const torch::Tensor& jRowPtr,
        const torch::Tensor& iColInds, const torch::Tensor& jColInds,
        const torch::Tensor& indexMap,
        torch::Tensor& outMatrix) {

    if (iValue.ndimension() == 1) {
        FVDB_DISPATCH_GRID_TYPES(iGrid, [&]() {
            AT_DISPATCH_FLOATING_TYPES(iValue.scalar_type(), "csr", [&]() {
                auto iGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(iGrid);
                auto jGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(jGrid);
                auto iCoordsAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(iCoords);
                auto jCoordsAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(jCoords);
                auto iValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(iValue);
                auto jValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(jValue);
                auto iRowPtrAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(iRowPtr);
                auto jRowPtrAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(jRowPtr);
                auto iColIndsAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(iColInds);
                auto jColIndsAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(jColInds);
                auto indexMapAcc = fvdb::tensorAccessor<DeviceTag, int, 2>(indexMap);
                auto outMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(outMatrix);

                // Dispatch according to both number of voxels and number of neighbors (3^3 = 27 points)

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t nidx) {
                        csrMatrixMultiplication<DeviceTag, GridType, scalar_t, 1, fvdb::TorchRAcc32>(bidx, eidx, nidx,
                                                iGridAcc, jGridAcc, iCoordsAcc, jCoordsAcc,
                                                iValueAcc, jValueAcc, iRowPtrAcc, jRowPtrAcc,
                                                iColIndsAcc, jColIndsAcc, indexMapAcc, outMatrixAcc);
                    };

                    // Do CUDA dispatch
                    forEachVoxelAnd1NeighborCUDA(iGrid, cb);
                } else{
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t nidx) {
                        csrMatrixMultiplication<DeviceTag, GridType, scalar_t, 1, fvdb::TorchAcc>(bidx, eidx, nidx,
                                                iGridAcc, jGridAcc, iCoordsAcc, jCoordsAcc,
                                                iValueAcc, jValueAcc, iRowPtrAcc, jRowPtrAcc,
                                                iColIndsAcc, jColIndsAcc, indexMapAcc, outMatrixAcc);
                    };

                    // Do CPU dispatch
                    forEachVoxelAnd1NeighborCPU(iGrid, cb);
                }
            });
        });
    } else if (iValue.ndimension() == 2) {
        FVDB_DISPATCH_GRID_TYPES(iGrid, [&]() {
            AT_DISPATCH_FLOATING_TYPES(iValue.scalar_type(), "csr", [&]() {
                auto iGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(iGrid);
                auto jGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(jGrid);
                auto iCoordsAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(iCoords);
                auto jCoordsAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(jCoords);
                auto iValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(iValue);
                auto jValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(jValue);
                auto iRowPtrAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(iRowPtr);
                auto jRowPtrAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(jRowPtr);
                auto iColIndsAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(iColInds);
                auto jColIndsAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(jColInds);
                auto indexMapAcc = fvdb::tensorAccessor<DeviceTag, int, 2>(indexMap);
                auto outMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(outMatrix);

                // Dispatch according to both number of voxels and number of neighbors (3^3 = 27 points)
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t nidx) {
                        csrMatrixMultiplication<DeviceTag, GridType, scalar_t, 2, fvdb::TorchRAcc32>(bidx, eidx, nidx,
                                                iGridAcc, jGridAcc, iCoordsAcc, jCoordsAcc,
                                                iValueAcc, jValueAcc, iRowPtrAcc, jRowPtrAcc,
                                                iColIndsAcc, jColIndsAcc, indexMapAcc, outMatrixAcc);
                    };

                    // Do CUDA dispatch
                    forEachVoxelAnd1NeighborCUDA(iGrid, cb);
                } else{
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t nidx) {
                        csrMatrixMultiplication<DeviceTag, GridType, scalar_t, 2, fvdb::TorchAcc>(bidx, eidx, nidx,
                                                iGridAcc, jGridAcc, iCoordsAcc, jCoordsAcc,
                                                iValueAcc, jValueAcc, iRowPtrAcc, jRowPtrAcc,
                                                iColIndsAcc, jColIndsAcc, indexMapAcc, outMatrixAcc);
                    };

                    // Do CPU dispatch
                    forEachVoxelAnd1NeighborCPU(iGrid, cb);
                }
            });
        });
    } else {
        throw std::runtime_error("Out dimension not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

}

template <c10::DeviceType DeviceTag>
void dispatchCSRMatrixMultiplicationBackward(
        const fvdb::detail::GridBatchImpl& iGrid,
        const fvdb::detail::GridBatchImpl& jGrid,
        const torch::Tensor& iCoords, const torch::Tensor& jCoords,
        const torch::Tensor& iValue, const torch::Tensor& jValue,
        const torch::Tensor& iRowPtr, const torch::Tensor& jRowPtr,
        const torch::Tensor& iColInds, const torch::Tensor& jColInds,
        const torch::Tensor& indexMap,
        const torch::Tensor& gradOutMatrix,
        torch::Tensor& gradIValue,
        torch::Tensor& gradJValue) {

    if (iValue.ndimension() == 1) {
        FVDB_DISPATCH_GRID_TYPES(iGrid, [&]() {
            AT_DISPATCH_FLOATING_TYPES(iValue.scalar_type(), "csr", [&]() {
                auto iGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(iGrid);
                auto jGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(jGrid);
                auto iCoordsAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(iCoords);
                auto jCoordsAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(jCoords);
                auto iValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(iValue);
                auto jValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(jValue);
                auto iRowPtrAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(iRowPtr);
                auto jRowPtrAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(jRowPtr);
                auto iColIndsAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(iColInds);
                auto jColIndsAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(jColInds);
                auto indexMapAcc = fvdb::tensorAccessor<DeviceTag, int, 2>(indexMap);
                auto gradOutMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradOutMatrix);
                auto gradIValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradIValue);
                auto gradJValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradJValue);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t nidx) {
                        csrMatrixMultiplicationBackward<DeviceTag, GridType, scalar_t, 1, fvdb::TorchRAcc32>(bidx, eidx, nidx,
                                                        iGridAcc, jGridAcc, iCoordsAcc, jCoordsAcc,
                                                        iValueAcc, jValueAcc, iRowPtrAcc, jRowPtrAcc,
                                                        iColIndsAcc, jColIndsAcc, indexMapAcc,
                                                        gradOutMatrixAcc, gradIValueAcc, gradJValueAcc);
                    };

                    // Do CUDA dispatch
                    forEachVoxelAnd1NeighborCUDA(iGrid, cb);
                } else{
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t nidx) {
                        csrMatrixMultiplicationBackward<DeviceTag, GridType, scalar_t, 1, fvdb::TorchAcc>(bidx, eidx, nidx,
                                                        iGridAcc, jGridAcc, iCoordsAcc, jCoordsAcc,
                                                        iValueAcc, jValueAcc, iRowPtrAcc, jRowPtrAcc,
                                                        iColIndsAcc, jColIndsAcc, indexMapAcc,
                                                        gradOutMatrixAcc, gradIValueAcc, gradJValueAcc);
                    };

                    // Do CPU dispatch
                    forEachVoxelAnd1NeighborCPU(iGrid, cb);
                }
            });
        });
    } else if (iValue.ndimension() == 2) {
        FVDB_DISPATCH_GRID_TYPES(iGrid, [&]() {
            AT_DISPATCH_FLOATING_TYPES(iValue.scalar_type(), "csr", [&]() {
                auto iGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(iGrid);
                auto jGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(jGrid);
                auto iCoordsAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(iCoords);
                auto jCoordsAcc = fvdb::tensorAccessor<DeviceTag, int64_t, 2>(jCoords);
                auto iValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(iValue);
                auto jValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(jValue);
                auto iRowPtrAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(iRowPtr);
                auto jRowPtrAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(jRowPtr);
                auto iColIndsAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(iColInds);
                auto jColIndsAcc = fvdb::tensorAccessor<DeviceTag, int, 1>(jColInds);
                auto indexMapAcc = fvdb::tensorAccessor<DeviceTag, int, 2>(indexMap);
                auto gradOutMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradOutMatrix);
                auto gradIValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradIValue);
                auto gradJValueAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradJValue);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t nidx) {
                        csrMatrixMultiplicationBackward<DeviceTag, GridType, scalar_t, 2, fvdb::TorchRAcc32>(bidx, eidx, nidx,
                                                        iGridAcc, jGridAcc, iCoordsAcc, jCoordsAcc,
                                                        iValueAcc, jValueAcc, iRowPtrAcc, jRowPtrAcc,
                                                        iColIndsAcc, jColIndsAcc, indexMapAcc,
                                                        gradOutMatrixAcc, gradIValueAcc, gradJValueAcc);
                    };

                    // Do CUDA dispatch
                    forEachVoxelAnd1NeighborCUDA(iGrid, cb);
                } else{
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t nidx) {
                        csrMatrixMultiplicationBackward<DeviceTag, GridType, scalar_t, 2, fvdb::TorchAcc>(bidx, eidx, nidx,
                                                        iGridAcc, jGridAcc, iCoordsAcc, jCoordsAcc,
                                                        iValueAcc, jValueAcc, iRowPtrAcc, jRowPtrAcc,
                                                        iColIndsAcc, jColIndsAcc, indexMapAcc,
                                                        gradOutMatrixAcc, gradIValueAcc, gradJValueAcc);
                    };

                    // Do CPU dispatch
                    forEachVoxelAnd1NeighborCPU(iGrid, cb);
                }
            });
        });
    } else {
        throw std::runtime_error("Out dimension not supported!");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

}

variable_list CsrMatrixMultiplication::forward(AutogradContext *ctx,
                                c10::intrusive_ptr<fvdb::detail::GridBatchImpl> iGrid,
                                c10::intrusive_ptr<fvdb::detail::GridBatchImpl> jGrid,
                                Variable iCoords, Variable jCoords,
                                Variable iValue, Variable jValue,
                                Variable iRowPtr, Variable jRowPtr,
                                Variable iColInds, Variable jColInds,
                                Variable indexMap, int64_t numEntries) {
    // Save for backward
    ctx->saved_data["iGrid"] = iGrid;
    ctx->saved_data["jGrid"] = jGrid;
    ctx->saved_data["iCoords"] = iCoords;
    ctx->saved_data["jCoords"] = jCoords;
    ctx->saved_data["iValue"] = iValue;
    ctx->saved_data["jValue"] = jValue;
    ctx->saved_data["iRowPtr"] = iRowPtr;
    ctx->saved_data["jRowPtr"] = jRowPtr;
    ctx->saved_data["iColInds"] = iColInds;
    ctx->saved_data["jColInds"] = jColInds;
    ctx->saved_data["indexMap"] = indexMap;

    // Prepare output
    auto opts = torch::TensorOptions().dtype(iValue.dtype()).device(iValue.device());
    torch::Tensor outMatrix = torch::zeros(numEntries, opts);

    FVDB_DISPATCH_KERNEL_DEVICE(iGrid->device(), [&]() {
        dispatchCSRMatrixMultiplication<DeviceTag>(*iGrid, *jGrid, iCoords, jCoords, iValue, jValue,
                                                    iRowPtr, jRowPtr, iColInds, jColInds, indexMap,
                                                    outMatrix);
    });

    return {outMatrix};
}

variable_list CsrMatrixMultiplication::backward(AutogradContext *ctx, variable_list grad_output) {
    // Use data saved in forward
    auto iGrid = ctx->saved_data["iGrid"].toCustomClass<fvdb::detail::GridBatchImpl>();
    auto jGrid = ctx->saved_data["jGrid"].toCustomClass<fvdb::detail::GridBatchImpl>();
    Variable iCoords = ctx->saved_data["iCoords"].toTensor();
    Variable jCoords = ctx->saved_data["jCoords"].toTensor();
    Variable iValue = ctx->saved_data["iValue"].toTensor();
    Variable jValue = ctx->saved_data["jValue"].toTensor();
    Variable iRowPtr = ctx->saved_data["iRowPtr"].toTensor();
    Variable jRowPtr = ctx->saved_data["jRowPtr"].toTensor();
    Variable iColInds = ctx->saved_data["iColInds"].toTensor();
    Variable jColInds = ctx->saved_data["jColInds"].toTensor();
    Variable indexMap = ctx->saved_data["indexMap"].toTensor();

    // Prepare grad input
    Variable gradOutMatrix = grad_output.at(0);

    // Prepare output
    torch::Tensor gradIValue = torch::zeros_like(iValue);
    torch::Tensor gradJValue = torch::zeros_like(jValue);

    FVDB_DISPATCH_KERNEL_DEVICE(iGrid->device(), [&]() {
        dispatchCSRMatrixMultiplicationBackward<DeviceTag>(*iGrid, *jGrid, iCoords, jCoords, iValue, jValue,
                                                            iRowPtr, jRowPtr, iColInds, jColInds, indexMap,
                                                            gradOutMatrix, gradIValue, gradJValue);
    });

    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            gradIValue, gradJValue,
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
}


template <c10::DeviceType DeviceTag, typename GridType, typename ScalarT, int Dim, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void qgBuilding(int32_t bidx, int32_t eidx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor,
        JaggedAccessor<ScalarT, 2> pts,
        const TensorAccessor<ScalarT, 2> ptsKernel,
        const TensorAccessor<ScalarT, 2> gridKernel,
        const TensorAccessor<ScalarT, 3> gradKernelPts,
        TensorAccessor<int, 2> outIndexer,
        TensorAccessor<ScalarT, Dim> outMatrix) {
    const int32_t pi = eidx;

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);
    auto primalAcc = gpuGrid->getAccessor();
    fvdb::detail::VoxelCoordTransform transform = batchAccessor.primalTransform(bidx);

    const auto& ptsTh = pts.data();
    const nanovdb::math::Vec3<ScalarT> p = transform.apply<ScalarT>(ptsTh[eidx][0], ptsTh[eidx][1], ptsTh[eidx][2]);

    // Iterate over all neighbors of the point
#pragma unroll
    for (auto it = NNIterator<3, ScalarT>(roundVec(p)); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1 + baseOffset;

        // Kernel gradient evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::math::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd<ScalarT>(
            offset, pi, transform.scale<ScalarT>()[0],
            p[0] - (ScalarT) (*it)[0],
            p[1] - (ScalarT) (*it)[1],
            p[2] - (ScalarT) (*it)[2],
            ptsKernel, gridKernel, gradKernelPts,
            Dim == 3, kiv, gradKiv, bk, dk, db);

        outIndexer[pi][it.getCount()] = offset;
        putValue<DeviceTag, ScalarT, TensorAccessor>(outMatrix, pi, it.getCount(), kiv, gradKiv);
    }
}

template <c10::DeviceType DeviceTag, typename GridType, typename ScalarT, int Dim, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void qgBuildingBackward(int32_t bidx, int32_t eidx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor,
        JaggedAccessor<ScalarT, 2> pts,
        const TensorAccessor<ScalarT, 2> ptsKernel,
        const TensorAccessor<ScalarT, 2> gridKernel,
        const TensorAccessor<ScalarT, 3> gradKernelPts,
        const TensorAccessor<ScalarT, Dim> gradOutQg,
        TensorAccessor<ScalarT, 2> gradPtsKernel,
        TensorAccessor<ScalarT, 2> gradGridKernel,
        TensorAccessor<ScalarT, 3> gradGradKernelPts) {
    const int32_t pi = eidx;

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);
    auto primalAcc = gpuGrid->getAccessor();
    fvdb::detail::VoxelCoordTransform transform = batchAccessor.primalTransform(bidx);

    const auto& ptsTh = pts.data();
    const nanovdb::math::Vec3<ScalarT> p = transform.apply<ScalarT>(ptsTh[eidx][0], ptsTh[eidx][1], ptsTh[eidx][2]);

    // Iterate over all neighbors of the point
#pragma unroll
    for (auto it = NNIterator<3, ScalarT>(roundVec(p)); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1 + baseOffset;

        // Kernel gradient evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::math::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd<ScalarT>(
            offset, pi, transform.scale<ScalarT>()[0],
            p[0] - (ScalarT) (*it)[0],
            p[1] - (ScalarT) (*it)[1],
            p[2] - (ScalarT) (*it)[2],
            ptsKernel, gridKernel, gradKernelPts,
            Dim == 3, kiv, gradKiv, bk, dk, db);

        // Backward
        ScalarT gData;
        nanovdb::math::Vec3<ScalarT> qData;
        getValue<DeviceTag, ScalarT, TensorAccessor>(gradOutQg, pi, it.getCount(), gData, qData);

        auto dummyAcc2 = gradPtsKernel;
        auto dummyAcc1 = gradPtsKernel[0];
        kernel_grad_evaluation_bwd<DeviceTag, ScalarT, true>(
                offset, pi,
                ptsKernel, gridKernel, gradKernelPts,
                dummyAcc1, dummyAcc2, Dim == 3, 1.0,
                gradPtsKernel, gradGridKernel, dummyAcc1, gradGradKernelPts,
                -1, gData, qData,
                kiv, gradKiv, bk, dk, db);
    }
}

template <c10::DeviceType DeviceTag>
void dispatchQgBuilding(const fvdb::detail::GridBatchImpl& grid,
                        const fvdb::JaggedTensor& ptsPos, const torch::Tensor& ptsKernel,
                        const torch::Tensor& gridKernel, const torch::Tensor& gradKernelPts,
                        torch::Tensor& outIndexer, torch::Tensor& outMatrix) {
    if (outMatrix.dim() == 2) {
        FVDB_DISPATCH_GRID_TYPES(grid, [&]() {
            AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "qgbuilding", [&]() {
                auto gridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(grid);
                auto ptsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernel);
                auto gridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gridKernel);
                auto gradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradKernelPts);
                auto outIndexerAcc = fvdb::tensorAccessor<DeviceTag, int, 2>(outIndexer);
                auto outMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(outMatrix);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedRAcc32<scalar_t, 2> ptsAcc) {
                        qgBuilding<DeviceTag, GridType, scalar_t, 2, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(bidx, eidx, gridAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, outIndexerAcc, outMatrixAcc);
                    };
                    fvdb::forEachJaggedElementChannelCUDA<scalar_t, 2>(128, 1, ptsPos, cb);
                } else {
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAcc<scalar_t, 2> ptsAcc) {
                        qgBuilding<DeviceTag, GridType, scalar_t, 2, fvdb::JaggedAcc, fvdb::TorchAcc>(bidx, eidx, gridAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, outIndexerAcc, outMatrixAcc);
                    };
                    fvdb::forEachJaggedElementChannelCPU<scalar_t, 2>(1, ptsPos, cb);
                }
            });
        });
    } else if (outMatrix.dim() == 3) {
        FVDB_DISPATCH_GRID_TYPES(grid, [&]() {
            AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "qgbuilding", [&]() {
                auto gridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(grid);
                auto ptsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernel);
                auto gridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gridKernel);
                auto gradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradKernelPts);
                auto outIndexerAcc = fvdb::tensorAccessor<DeviceTag, int, 2>(outIndexer);
                auto outMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(outMatrix);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedRAcc32<scalar_t, 2> ptsAcc) {
                        qgBuilding<DeviceTag, GridType, scalar_t, 3, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(bidx, eidx, gridAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, outIndexerAcc, outMatrixAcc);
                    };
                    fvdb::forEachJaggedElementChannelCUDA<scalar_t, 2>(128, 1, ptsPos, cb);
                } else {
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAcc<scalar_t, 2> ptsAcc) {
                        qgBuilding<DeviceTag, GridType, scalar_t, 3, fvdb::JaggedAcc, fvdb::TorchAcc>(bidx, eidx, gridAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, outIndexerAcc, outMatrixAcc);
                    };
                    fvdb::forEachJaggedElementChannelCPU<scalar_t, 2>(1, ptsPos, cb);
                }
            });
        });
    } else {
        throw std::runtime_error("Out dimension not supported!");
    }
}

template <c10::DeviceType DeviceTag>
void dispatchQgBuildingBackward(const fvdb::detail::GridBatchImpl& grid,
                                const fvdb::JaggedTensor& ptsPos, const torch::Tensor& ptsKernel,
                                const torch::Tensor& gridKernel, const torch::Tensor& gradKernelPts,
                                const torch::Tensor& gradOutMatrix,
                                torch::Tensor& gradPtsKernel, torch::Tensor& gradGridKernel,
                                torch::Tensor& gradGradKernelPts) {
    if (gradOutMatrix.dim() == 2) {
        FVDB_DISPATCH_GRID_TYPES(grid, [&]() {
            AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "qgbuilding", [&]() {
                auto gridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(grid);
                auto ptsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernel);
                auto gridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gridKernel);
                auto gradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradKernelPts);
                auto gradOutMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradOutMatrix);
                auto gradPtsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradPtsKernel);
                auto gradGridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradGridKernel);
                auto gradGradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradGradKernelPts);
                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedRAcc32<scalar_t, 2> ptsAcc) {
                        qgBuildingBackward<DeviceTag, GridType, scalar_t, 2, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(bidx, eidx, gridAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, gradOutMatrixAcc, gradPtsKernelAcc, gradGridKernelAcc, gradGradKernelPtsAcc);
                    };
                    fvdb::forEachJaggedElementChannelCUDA<scalar_t, 2>(128, 1, ptsPos, cb);
                } else {
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAcc<scalar_t, 2> ptsAcc) {
                        qgBuildingBackward<DeviceTag, GridType, scalar_t, 2, fvdb::JaggedAcc, fvdb::TorchAcc>(bidx, eidx, gridAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, gradOutMatrixAcc, gradPtsKernelAcc, gradGridKernelAcc, gradGradKernelPtsAcc);
                    };
                    fvdb::forEachJaggedElementChannelCPU<scalar_t, 2>(1, ptsPos, cb);
                }
            });
        });
    } else if (gradOutMatrix.dim() == 3) {
        FVDB_DISPATCH_GRID_TYPES(grid, [&]() {
            AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "qgbuilding", [&]() {
                auto gridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(grid);
                auto ptsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernel);
                auto gridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gridKernel);
                auto gradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradKernelPts);
                auto gradOutMatrixAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradOutMatrix);
                auto gradPtsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradPtsKernel);
                auto gradGridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradGridKernel);
                auto gradGradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradGradKernelPts);

                if constexpr (DeviceTag == torch::kCUDA) {
                    auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedRAcc32<scalar_t, 2> ptsAcc) {
                        qgBuildingBackward<DeviceTag, GridType, scalar_t, 3, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(bidx, eidx, gridAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, gradOutMatrixAcc, gradPtsKernelAcc, gradGridKernelAcc, gradGradKernelPtsAcc);
                    };
                    fvdb::forEachJaggedElementChannelCUDA<scalar_t, 2>(128, 1, ptsPos, cb);
                } else {
                    auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAcc<scalar_t, 2> ptsAcc) {
                        qgBuildingBackward<DeviceTag, GridType, scalar_t, 3, fvdb::JaggedAcc, fvdb::TorchAcc>(bidx, eidx, gridAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, gradOutMatrixAcc, gradPtsKernelAcc, gradGridKernelAcc, gradGradKernelPtsAcc);
                    };
                    fvdb::forEachJaggedElementChannelCPU<scalar_t, 2>(1, ptsPos, cb);
                }
            });
        });

    } else {
        throw std::runtime_error("Out dimension not supported!");
    }
}


variable_list QgBuilding::forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
        fvdb::JaggedTensor pts, Variable ptsKernel,
        Variable gridKernel, Variable gradKernelPts,
        bool grad) {

    ctx->saved_data["grid"] = grid;
    ctx->saved_data["pts_data"] = pts.jdata();
    ctx->saved_data["pts_offsets"] = pts.joffsets();
    ctx->saved_data["ptsKernel"] = ptsKernel;
    ctx->saved_data["gridKernel"] = gridKernel;
    ctx->saved_data["gradKernelPts"] = gradKernelPts;

    // Prepare output
    auto opts = torch::TensorOptions().device(pts.device());
    torch::Tensor indexer = torch::full({pts.size(0), 27}, -1, opts.dtype(torch::kInt32));

    torch::Tensor qg;
    if (grad) {
        qg = torch::full({pts.size(0), 27, 3}, 0.0, opts.dtype(pts.dtype()));
    } else {
        qg = torch::full({pts.size(0), 27}, 0.0, opts.dtype(pts.dtype()));
    }

    ctx->mark_non_differentiable({indexer});

    FVDB_DISPATCH_KERNEL_DEVICE(pts.device(), [&]() {
        dispatchQgBuilding<DeviceTag>(
            *grid, pts, ptsKernel, gridKernel, gradKernelPts, indexer, qg);
    });

    return {qg, indexer};
}


variable_list QgBuilding::backward(AutogradContext *ctx, variable_list grad_output) {
    auto grid = ctx->saved_data["grid"].toCustomClass<fvdb::detail::GridBatchImpl>();
    torch::Tensor pts_data = ctx->saved_data["pts_data"].toTensor();
    torch::Tensor pts_joffsets = ctx->saved_data["pts_offsets"].toTensor();
    auto pts = fvdb::JaggedTensor::from_data_and_offsets(pts_data, pts_joffsets);

    Variable ptsKernel = ctx->saved_data["ptsKernel"].toTensor();
    Variable gridKernel = ctx->saved_data["gridKernel"].toTensor();
    Variable gradKernelPts = ctx->saved_data["gradKernelPts"].toTensor();

    // Prepare grad input
    Variable gradOutQg = grad_output.at(0);

    bool grad = gradOutQg.dim() == 3;

    // Prepare output
    torch::Tensor gradPtsKernel = torch::zeros_like(ptsKernel);
    torch::Tensor gradGridKernel = torch::zeros_like(gridKernel);
    torch::Tensor gradGradKernelPts = torch::zeros_like(gradKernelPts);

    FVDB_DISPATCH_KERNEL_DEVICE(pts.device(), [&]() {
        dispatchQgBuildingBackward<DeviceTag>(
            *grid, pts, ptsKernel, gridKernel, gradKernelPts, gradOutQg, gradPtsKernel, gradGridKernel, gradGradKernelPts);
    });

    return {torch::Tensor(), torch::Tensor(), gradPtsKernel, gradGridKernel, gradGradKernelPts, torch::Tensor()};
}
