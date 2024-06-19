#include "functions.h"
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

template <typename GridType, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void cooIndexerCallback(
        int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> iAcc,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> jAcc,
        TensorAccessor<int32_t, 2> indexer) {

    const nanovdb::NanoGrid<GridType>* iGrid = iAcc.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = \
        iGrid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t iBaseOffset = iAcc.voxelOffset(batchIdx);
    fvdb::detail::VoxelCoordTransform iTransform = iAcc.primalTransform(batchIdx);

    using NNIt5 = NNIterator<5, float>;
    auto jPrimalAcc = jAcc.grid(batchIdx)->getAccessor();
    const int64_t jBaseOffset = jAcc.voxelOffset(batchIdx);
    const auto& primalRange = nanovdb::math::Vec3<float>(2.5);
    fvdb::detail::VoxelCoordTransform jTransform = jAcc.primalTransform(batchIdx);

    if (leaf.isActive(voxelIdx)) {
        nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);

        const auto& iPrimal = ijk.asVec3s();
        const auto& ijWorld = iTransform.applyInv(iPrimal);
        const auto& jcPrimal = jTransform.apply(ijWorld);
        for (auto jt = NNIt5(jcPrimal); jt.isValid(); ++jt) {
            if (!jPrimalAcc.isActive(*jt)) {
                continue;
            }
            const auto& jPrimal = jt->asVec3s();
            if (!has_overlap(
                    iTransform.applyInv(iPrimal - primalRange),
                    iTransform.applyInv(iPrimal + primalRange),
                    jTransform.applyInv(jPrimal - primalRange),
                    jTransform.applyInv(jPrimal + primalRange))) {
                continue;
            }
            indexer[leaf.getValue(voxelIdx) - 1 + iBaseOffset][jt.getCount()] = \
                jPrimalAcc.getValue(*jt) - 1 + jBaseOffset;
        }
    }

}

template <c10::DeviceType DeviceTag>
void dispatchBuildCOOIndexer(
        const fvdb::detail::GridBatchImpl& iHandle,
        const fvdb::detail::GridBatchImpl& jHandle,
        torch::Tensor& indexer) {

    iHandle.checkDevice(indexer);
    jHandle.checkDevice(indexer);

    auto indexerAcc = fvdb::tensorAccessor<DeviceTag, int32_t, 2>(indexer);

    FVDB_DISPATCH_GRID_TYPES(iHandle, [&]() {
        auto jAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(jHandle);

        if constexpr (DeviceTag == torch::kCUDA) {
            auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                    fvdb::detail::GridBatchImpl::Accessor<GridType> iAcc) {
                cooIndexerCallback<GridType, fvdb::TorchRAcc32>(
                    batchIdx, leafIdx, voxelIdx, iAcc, jAcc, indexerAcc);
            };
            fvdb::forEachVoxelCUDA<GridType>(128, 1, iHandle, cb);
        } else {
            auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                    fvdb::detail::GridBatchImpl::Accessor<GridType> iAcc) {
                cooIndexerCallback<GridType, fvdb::TorchAcc>(
                    batchIdx, leafIdx, voxelIdx, iAcc, jAcc, indexerAcc);
            };
            fvdb::forEachVoxelCPU<GridType>(1, iHandle, cb);
        }
    });

}


fvdb::JaggedTensor buildCOOIndexer(
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> iSVH,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> jSVH) {

    unsigned iSize = iSVH->totalVoxels();
    torch::Tensor indexer = torch::full(
            {iSize, 125}, -1, torch::TensorOptions().dtype(torch::kInt32).device(iSVH->device()));

    FVDB_DISPATCH_KERNEL_DEVICE(iSVH->device(), [&]() {
        dispatchBuildCOOIndexer<DeviceTag>(*iSVH, *jSVH, indexer);
    });

    return iSVH->jaggedTensor(indexer, true);
}


template <typename GridType, typename ScalarT, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ inline void kernelEvaluationCallback(
        int32_t bidx, 
        int32_t eidx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor,
        JaggedAccessor<ScalarT, 2> query,
        const TensorAccessor<ScalarT, 2> queryKernel,
        const TensorAccessor<ScalarT, 2> gridKernel,
        const TensorAccessor<ScalarT, 1> gridAlpha,
        const TensorAccessor<ScalarT, 3> gradKernelQuery,
        TensorAccessor<ScalarT, 1> outFunc,
        TensorAccessor<ScalarT, 2> outGradFunc
    ) {
    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);
    auto primalAcc = gpuGrid->getAccessor();
    fvdb::detail::VoxelCoordTransform transform = batchAccessor.primalTransform(bidx);

    const auto& queryTh = query.data();

    const nanovdb::math::Vec3<ScalarT> p = transform.apply<ScalarT>(queryTh[eidx]);
    const bool grad = outGradFunc.size(0) > 0;

    auto func = static_cast<ScalarT>(0.0);
    nanovdb::math::Vec3<ScalarT> dfunc(0.0);

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
                queryKernel, gridKernel, gradKernelQuery,
                grad, kiv, gradKiv, bk, dk, db);

        func += gridAlpha[offset] * kiv;
        dfunc += gridAlpha[offset] * gradKiv;
    }

    // Write result for this point.
    outFunc[eidx] = func;
    if (grad) {
#pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            outGradFunc[eidx][dim] = dfunc[dim];
        }
    }
}


template <typename GridType, typename ScalarT, c10::DeviceType DeviceTag, template <typename T, int32_t D> typename JaggedAccessor, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ inline void kernelEvaluationBackwardCallback(
        int32_t bidx, 
        int32_t eidx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor,
        JaggedAccessor<ScalarT, 2> query,
        const TensorAccessor<ScalarT, 2> queryKernel,
        const TensorAccessor<ScalarT, 2> gridKernel,
        const TensorAccessor<ScalarT, 1> gridAlpha,
        const TensorAccessor<ScalarT, 3> gradKernelQuery,
        const TensorAccessor<ScalarT, 1> gradOutFunc,
        const TensorAccessor<ScalarT, 2> gradOutGradFunc,
        TensorAccessor<ScalarT, 2> gradQueryKernel,
        TensorAccessor<ScalarT, 2> gradGridKernel,
        TensorAccessor<ScalarT, 1> gradGridAlpha,
        TensorAccessor<ScalarT, 3> gradGradKernelQuery) {

    const nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);
    const int64_t baseOffset = batchAccessor.voxelOffset(bidx);
    auto primalAcc = gpuGrid->getAccessor();
    fvdb::detail::VoxelCoordTransform transform = batchAccessor.primalTransform(bidx);
    const auto& queryTh = query.data();

    const nanovdb::math::Vec3<ScalarT> p = transform.apply<ScalarT>(queryTh[eidx]);
    const bool grad = gradOutGradFunc.size(0) > 0;

    // For each point, iterate through all its neighbours.
    for (auto it = NNIterator<3, ScalarT>(p); it.isValid(); ++it) {
        if (!primalAcc.isActive(*it)) {
            continue;
        }
        const int64_t offset = primalAcc.getValue(*it) - 1 + baseOffset;

        ScalarT alpha = gridAlpha[offset];

        // Kernel (and gradient) evaluation
        ScalarT kiv = 0.0, bk, dk;
        nanovdb::math::Vec3<ScalarT> gradKiv(0.0), db(0.0);
        kernel_grad_evaluation_fwd(
                offset, eidx, transform.scale<ScalarT>()[0],
                p[0] - (ScalarT) (*it)[0],
                p[1] - (ScalarT) (*it)[1],
                p[2] - (ScalarT) (*it)[2],
                queryKernel, gridKernel, gradKernelQuery,
                grad, kiv, gradKiv, bk, dk, db);

        // Backprop (through the function part)
        kernel_grad_evaluation_bwd<DeviceTag, ScalarT, false>(
                offset, eidx,
                queryKernel, gridKernel, gradKernelQuery,
                gradOutFunc, gradOutGradFunc,
                false, alpha,
                gradQueryKernel, gradGridKernel, gradGridAlpha, gradGradKernelQuery,
                -1, (ScalarT) 0.0, nanovdb::math::Vec3<ScalarT>(0.0),
                kiv, gradKiv, bk, dk, db);

        if (grad) {
            // Backprop (through the grad function part)
            kernel_grad_evaluation_bwd<DeviceTag, ScalarT, false>(
                    offset, eidx,
                    queryKernel, gridKernel, gradKernelQuery,
                    gradOutFunc, gradOutGradFunc,
                    true, alpha,
                    gradQueryKernel, gradGridKernel, gradGridAlpha, gradGradKernelQuery,
                    -1, (ScalarT) 0.0, nanovdb::math::Vec3<ScalarT>(0.0),
                    kiv, gradKiv, bk, dk, db);
        }
    }
}



template <c10::DeviceType DeviceTag>
void dispatchKernelEvaluation(const fvdb::detail::GridBatchImpl& batchHdl,
                              const fvdb::JaggedTensor& query,
                              const torch::Tensor& queryKernel,
                              const torch::Tensor& gridKernel,
                              const torch::Tensor& gridAlpha,
                              const torch::Tensor& gradKernelQuery,
                              torch::Tensor& outFunc,
                              torch::Tensor& outGradFunc) {
    
    batchHdl.checkDevice(query);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES(queryKernel.scalar_type(), "kernelEvaluation", [&]() {

            auto batchAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto queryKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(queryKernel);
            auto gridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gridKernel);
            auto gridAlphaAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gridAlpha);
            auto gradKernelQueryAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradKernelQuery);
            auto outFuncAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(outFunc);
            auto outGradFuncAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(outGradFunc);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedRAcc32<scalar_t, 2> queryAcc) {
                    kernelEvaluationCallback<GridType, scalar_t, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(
                        bidx, eidx, batchAcc, queryAcc, queryKernelAcc, gridKernelAcc, gridAlphaAcc, gradKernelQueryAcc,
                        outFuncAcc, outGradFuncAcc);
                };
                fvdb::forEachJaggedElementChannelCUDA<scalar_t, 2>(128, 1, query, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAcc<scalar_t, 2> queryAcc) {
                    kernelEvaluationCallback<GridType, scalar_t, fvdb::JaggedAcc, fvdb::TorchAcc>(
                        bidx, eidx, batchAcc, queryAcc, queryKernelAcc, gridKernelAcc, gridAlphaAcc, gradKernelQueryAcc,
                        outFuncAcc, outGradFuncAcc);
                };
                fvdb::forEachJaggedElementChannelCPU<scalar_t, 2>(1, query, cb);
            }
        });
    });
}


template <c10::DeviceType DeviceTag>
void dispatchKernelEvaluationBackward(const fvdb::detail::GridBatchImpl& batchHdl,
                                      const fvdb::JaggedTensor& query,
                                      const torch::Tensor& queryKernel,
                                      const torch::Tensor& gridKernel,
                                      const torch::Tensor& gridAlpha,
                                      const torch::Tensor& gradKernelQuery,
                                      const torch::Tensor& gradOutFunc,
                                      const torch::Tensor& gradOutGradFunc,
                                      torch::Tensor& gradQueryKernel,
                                      torch::Tensor& gradGridKernel,
                                      torch::Tensor& gradGridAlpha,
                                      torch::Tensor& gradGradKernelQuery) {

    batchHdl.checkDevice(query);

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        AT_DISPATCH_FLOATING_TYPES(queryKernel.scalar_type(), "kernelEvaluationBackward", [&]() {

            auto batchAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(batchHdl);
            auto queryKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(queryKernel);
            auto gridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gridKernel);
            auto gridAlphaAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gridAlpha);
            auto gradKernelQueryAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradKernelQuery);
            auto gradOutFuncAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradOutFunc);
            auto gradOutGradFuncAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradOutGradFunc);
            auto gradQueryKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradQueryKernel);
            auto gradGridKernelAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 2>(gradGridKernel);
            auto gradGridAlphaAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 1>(gradGridAlpha);
            auto gradGradKernelQueryAcc = fvdb::tensorAccessor<DeviceTag, scalar_t, 3>(gradGradKernelQuery);

            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedRAcc32<scalar_t, 2> queryAcc) {
                    kernelEvaluationBackwardCallback<GridType, scalar_t, DeviceTag, fvdb::JaggedRAcc32, fvdb::TorchRAcc32>(
                        bidx, eidx, batchAcc, queryAcc, queryKernelAcc, gridKernelAcc, gridAlphaAcc, gradKernelQueryAcc,
                        gradOutFuncAcc, gradOutGradFuncAcc,
                        gradQueryKernelAcc, gradGridKernelAcc, gradGridAlphaAcc, gradGradKernelQueryAcc);
                };
                fvdb::forEachJaggedElementChannelCUDA<scalar_t, 2>(128, 1, query, cb);
            } else {
                auto cb = [=] (int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAcc<scalar_t, 2> queryAcc) {
                    kernelEvaluationBackwardCallback<GridType, scalar_t, DeviceTag, fvdb::JaggedAcc, fvdb::TorchAcc>(
                        bidx, eidx, batchAcc, queryAcc, queryKernelAcc, gridKernelAcc, gridAlphaAcc, gradKernelQueryAcc,
                        gradOutFuncAcc, gradOutGradFuncAcc,
                        gradQueryKernelAcc, gradGridKernelAcc, gradGridAlphaAcc, gradGradKernelQueryAcc);
                };
                fvdb::forEachJaggedElementChannelCPU<scalar_t, 2>(1, query, cb);
            }
        });
    });
}


variable_list KernelEvaluation::forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
        bool grad,
        fvdb::JaggedTensor query, Variable queryKernel,
        Variable gridKernel, Variable gridAlpha,
        Variable gradKernelQuery) {
    // Save for backward
    ctx->saved_data["grid"] = grid;
    ctx->saved_data["query_data"] = query.jdata();
    ctx->saved_data["query_offsets"] = query.joffsets();
    ctx->saved_data["queryKernel"] = queryKernel;
    ctx->saved_data["gridKernel"] = gridKernel;
    ctx->saved_data["gridAlpha"] = gridAlpha;
    ctx->saved_data["gradKernelQuery"] = gradKernelQuery;

    // Prepare output
    auto opts = torch::TensorOptions().dtype(queryKernel.dtype())
            .device(queryKernel.device());
    torch::Tensor outFunc = torch::zeros(query.size(0), opts);
    torch::Tensor outGradFunc = torch::zeros({0, 3}, opts);
    if (grad) {
        outGradFunc = torch::zeros({query.size(0), 3}, opts);
    }

    FVDB_DISPATCH_KERNEL_DEVICE(query.device(), [&]() {
        dispatchKernelEvaluation<DeviceTag>(*grid, query, queryKernel, gridKernel, 
            gridAlpha, gradKernelQuery, outFunc, outGradFunc);
    });

    return {outFunc, outGradFunc};
}

variable_list KernelEvaluation::backward(AutogradContext *ctx, variable_list grad_output) {
    // Use data saved in forward
    auto grid = ctx->saved_data["grid"].toCustomClass<fvdb::detail::GridBatchImpl>();

    torch::Tensor query_data = ctx->saved_data["query_data"].toTensor();
    torch::Tensor query_offsets = ctx->saved_data["query_offsets"].toTensor();
    auto query = fvdb::JaggedTensor::from_data_and_offsets(query_data, query_offsets);

    Variable queryKernel = ctx->saved_data["queryKernel"].toTensor();
    Variable gridKernel = ctx->saved_data["gridKernel"].toTensor();
    Variable gridAlpha = ctx->saved_data["gridAlpha"].toTensor();
    Variable gradKernelQuery = ctx->saved_data["gradKernelQuery"].toTensor();

    // Prepare grad input
    Variable gradOutFunc = grad_output.at(0);
    Variable gradOutGradFunc = grad_output.at(1);

    // Prepare output
    torch::Tensor gradQueryKernel = torch::zeros_like(queryKernel);
    torch::Tensor gradGridKernel = torch::zeros_like(gridKernel);
    torch::Tensor gradGridAlpha = torch::zeros_like(gridAlpha);
    torch::Tensor gradGradKernelQuery = torch::zeros_like(gradKernelQuery);

    FVDB_DISPATCH_KERNEL_DEVICE(query.device(), [&]() {
        dispatchKernelEvaluationBackward<DeviceTag>(*grid, query, queryKernel, gridKernel, gridAlpha,
            gradKernelQuery, gradOutFunc, gradOutGradFunc,
            gradQueryKernel, gradGridKernel, gradGridAlpha, gradGradKernelQuery);
    });

    return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
            gradQueryKernel, gradGridKernel, gradGridAlpha, gradGradKernelQuery};
}
