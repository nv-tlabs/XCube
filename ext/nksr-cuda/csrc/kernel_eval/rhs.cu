#include "functions.h"
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

template <typename GridType, typename ScalarT, c10::DeviceType DeviceTag, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ inline void rhsEvaluationCallback(
        int32_t bidx, 
        int32_t eidx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor,
        JaggedAccessor<ScalarT, 2> pts,
        const TensorAccessor<ScalarT, 2> ptsKernel,
        const TensorAccessor<ScalarT, 2> gridKernel,
        const TensorAccessor<ScalarT, 3> gradKernelPts,
        const TensorAccessor<ScalarT, 2> ptsData,
        TensorAccessor<ScalarT, 1> outRhs
    ) {

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);
    auto primalAcc = gpuGrid->getAccessor();
    fvdb::detail::VoxelCoordTransform transform = batchAccessor.primalTransform(bidx);

    const auto& ptsTh = pts.data();
    const nanovdb::math::Vec3<ScalarT> p = transform.apply<ScalarT>(ptsTh[eidx][0], ptsTh[eidx][1], ptsTh[eidx][2]);
    nanovdb::math::Vec3<ScalarT> pData(ptsData[eidx][0], ptsData[eidx][1], ptsData[eidx][2]);

    // For each point, iterate through all its neighbours.
    #pragma unroll
    for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1 + baseOffset;

        // Kernel (and gradient) evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::math::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd<ScalarT>(
                offset, eidx, transform.scale<ScalarT>()[0],
                p[0] - (ScalarT) (*it)[0],
                p[1] - (ScalarT) (*it)[1],
                p[2] - (ScalarT) (*it)[2],
                ptsKernel, gridKernel, gradKernelPts,
                true, kiv, gradKiv, bk, dk, db);

        ScalarT res = pData.dot(gradKiv);

        atomicAddIfGPU<DeviceTag>(&outRhs[offset], res);
    }
}

template <typename GridType, typename ScalarT, c10::DeviceType DeviceTag, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ inline void rhsEvaluationBackwardCallback(
        int32_t bidx, int32_t eidx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor,
        JaggedAccessor<ScalarT, 2> pts,
        const TensorAccessor<ScalarT, 2> ptsKernel,
        const TensorAccessor<ScalarT, 2> gridKernel,
        const TensorAccessor<ScalarT, 3> gradKernelPts,
        const TensorAccessor<ScalarT, 2> ptsData,
        const TensorAccessor<ScalarT, 1> gradOutRhs,
        TensorAccessor<ScalarT, 2> gradPtsKernel,
        TensorAccessor<ScalarT, 2> gradGridKernel,
        TensorAccessor<ScalarT, 3> gradGradKernelPts,
        TensorAccessor<ScalarT, 2> gradPtsData) {

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);
    auto primalAcc = gpuGrid->getAccessor();
    fvdb::detail::VoxelCoordTransform transform = batchAccessor.primalTransform(bidx);

    const auto& ptsTh = pts.data();
    const nanovdb::math::Vec3<ScalarT> p = transform.apply<ScalarT>(ptsTh[eidx][0], ptsTh[eidx][1], ptsTh[eidx][2]);
    nanovdb::math::Vec3<ScalarT> pData(ptsData[eidx][0], ptsData[eidx][1], ptsData[eidx][2]);

    // For each point, iterate through all its neighbours.
    #pragma unroll
    for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1 + baseOffset;

        // Kernel (and gradient) evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::math::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd(
                offset, eidx, transform.scale<ScalarT>()[0],
                p[0] - (ScalarT) (*it)[0],
                p[1] - (ScalarT) (*it)[1],
                p[2] - (ScalarT) (*it)[2],
                ptsKernel, gridKernel, gradKernelPts,
                true, kiv, gradKiv, bk, dk, db);

        auto dummyAcc2 = ptsData;
        auto dummyAcc1 = gradOutRhs;

        // Backward
        kernel_grad_evaluation_bwd<DeviceTag, ScalarT, true>(
                offset, eidx,
                ptsKernel, gridKernel, gradKernelPts,
                gradOutRhs, dummyAcc2, true, 1.0,
                gradPtsKernel, gradGridKernel, dummyAcc1, gradGradKernelPts,
                offset, 0.0, pData,
                kiv, gradKiv, bk, dk, db);
        for (int dim = 0; dim < 3; ++dim) {
            gradPtsData[eidx][dim] += gradOutRhs[offset] * gradKiv[dim];
        }
    }
}

template <c10::DeviceType DeviceTag>
void dispatchRhsEvaluation(
        const fvdb::detail::GridBatchImpl& batchHdl, 
        const fvdb::JaggedTensor& pts,
        const torch::Tensor& ptsKernel,
        const torch::Tensor& gridKernel,
        const torch::Tensor& gradKernelPts,
        const torch::Tensor& ptsData,
        torch::Tensor& outRhs) {

    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(pts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "RhsEvaluation", [&]() {

            auto batchAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto ptsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernel);
            auto gridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gridKernel);
            auto gradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradKernelPts);
            auto ptsDataAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsData);
            auto outRhsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(outRhs);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedRAcc32<scalar_t, 2> ptsAcc) {
                    rhsEvaluationCallback<GridType, scalar_t, DeviceTag, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(bidx, eidx, batchAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, ptsDataAcc, outRhsAcc);
                };
                fvdb::forEachJaggedElementChannelCUDA<scalar_t, 2>(128, 1, pts, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAcc<scalar_t, 2> ptsAcc) {
                    rhsEvaluationCallback<GridType, scalar_t, DeviceTag, fvdb::JaggedAcc, fvdb::TorchAcc>(bidx, eidx, batchAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, ptsDataAcc, outRhsAcc);
                };
                fvdb::forEachJaggedElementChannelCPU<scalar_t, 2>(1, pts, cb);
            }
        });
    });
}

template <c10::DeviceType DeviceTag>
void dispatchRhsEvaluationBackward(
        const fvdb::detail::GridBatchImpl& batchHdl, 
        const fvdb::JaggedTensor& pts,
        const torch::Tensor& ptsKernel,
        const torch::Tensor& gridKernel,
        const torch::Tensor& gradKernelPts,
        const torch::Tensor& ptsData,
        const torch::Tensor& gradOutRhs,
        torch::Tensor& gradPtsKernel,
        torch::Tensor& gradGridKernel,
        torch::Tensor& gradGradKernelPts,
        torch::Tensor& gradPtsData) {

    batchHdl.checkNonEmptyGrid();
    batchHdl.checkDevice(pts);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES(ptsKernel.scalar_type(), "RhsEvaluation", [&]() {

            auto batchAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto ptsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsKernel);
            auto gridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gridKernel);
            auto gradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradKernelPts);
            auto ptsDataAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(ptsData);
            auto gradOutRhsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradOutRhs);
            auto gradPtsKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradPtsKernel);
            auto gradGridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradGridKernel);
            auto gradGradKernelPtsAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradGradKernelPts);
            auto gradPtsDataAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradPtsData);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedRAcc32<scalar_t, 2> ptsAcc) {
                    rhsEvaluationBackwardCallback<GridType, scalar_t, DeviceTag, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(bidx, eidx, batchAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, ptsDataAcc, gradOutRhsAcc, gradPtsKernelAcc, gradGridKernelAcc, gradGradKernelPtsAcc, gradPtsDataAcc);
                };
                fvdb::forEachJaggedElementChannelCUDA<scalar_t, 2>(128, 1, pts, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAcc<scalar_t, 2> ptsAcc) {
                    rhsEvaluationBackwardCallback<GridType, scalar_t, DeviceTag, fvdb::JaggedAcc, fvdb::TorchAcc>(bidx, eidx, batchAcc, ptsAcc, ptsKernelAcc, gridKernelAcc, gradKernelPtsAcc, ptsDataAcc, gradOutRhsAcc, gradPtsKernelAcc, gradGridKernelAcc, gradGradKernelPtsAcc, gradPtsDataAcc);
                };
                fvdb::forEachJaggedElementChannelCPU<scalar_t, 2>(1, pts, cb);
            }
        });
    });
}

variable_list RhsEvaluation::forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
        fvdb::JaggedTensor pts, Variable ptsKernel,
        Variable gridKernel, Variable gradKernelPts,
        Variable ptsData) {
    ctx->saved_data["grid"] = grid;
    ctx->saved_data["pts_data"] = pts.jdata();
    ctx->saved_data["pts_offsets"] = pts.joffsets();
    ctx->saved_data["ptsKernel"] = ptsKernel;
    ctx->saved_data["gridKernel"] = gridKernel;
    ctx->saved_data["gradKernelPts"] = gradKernelPts;
    ctx->saved_data["ptsData"] = ptsData;

    // Prepare output
    auto opts = torch::TensorOptions().dtype(pts.dtype())
            .device(pts.device());
    torch::Tensor outRhs = torch::zeros(gridKernel.size(0), opts);

    FVDB_DISPATCH_KERNEL_DEVICE(pts.device(), [&]() {
        dispatchRhsEvaluation<DeviceTag>(
            *grid, pts, ptsKernel, gridKernel, gradKernelPts, ptsData, outRhs);
    });

    return {outRhs};
}


variable_list RhsEvaluation::backward(
        AutogradContext *ctx,
        variable_list grad_output) {

    auto grid = ctx->saved_data["grid"].toCustomClass<fvdb::detail::GridBatchImpl>();
    torch::Tensor pts_data = ctx->saved_data["pts_data"].toTensor();
    torch::Tensor pts_offsets = ctx->saved_data["pts_offsets"].toTensor();
    auto pts = fvdb::JaggedTensor::from_data_and_offsets(pts_data, pts_offsets);

    Variable ptsKernel = ctx->saved_data["ptsKernel"].toTensor();
    Variable gridKernel = ctx->saved_data["gridKernel"].toTensor();
    Variable gradKernelPts = ctx->saved_data["gradKernelPts"].toTensor();
    Variable ptsData = ctx->saved_data["ptsData"].toTensor();

    // Prepare grad input
    Variable gradOutRhs = grad_output.at(0);

    // Prepare output
    torch::Tensor gradPtsKernel = torch::zeros_like(ptsKernel);
    torch::Tensor gradGridKernel = torch::zeros_like(gridKernel);
    torch::Tensor gradGradKernelPts = torch::zeros_like(gradKernelPts);
    torch::Tensor gradPtsData = torch::zeros_like(ptsData);

    FVDB_DISPATCH_KERNEL_DEVICE(pts.device(), [&]() {
        dispatchRhsEvaluationBackward<DeviceTag>(
                *grid, pts, ptsKernel, gridKernel, gradKernelPts, ptsData, gradOutRhs,
                gradPtsKernel, gradGridKernel, gradGradKernelPts, gradPtsData);
    });

    return {torch::Tensor(), torch::Tensor(),
            gradPtsKernel, gradGridKernel, gradGradKernelPts, gradPtsData};
}
