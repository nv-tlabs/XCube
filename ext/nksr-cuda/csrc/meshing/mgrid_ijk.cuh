#include "../common/iter_util.h"

template <typename GridType, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void childMaskCallback(
        int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> thisGridAcc,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> childGridAcc,
        TensorAccessor<bool, 1> childMask) {

    const nanovdb::NanoGrid<GridType>* grid = thisGridAcc.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];

    const int64_t baseOffset = thisGridAcc.voxelOffset(batchIdx);
    if (!leaf.isActive(voxelIdx)) {
        return;
    }

    const int64_t offsetI = baseOffset + leaf.getValue(voxelIdx) - 1;
    nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);

    auto cGridAcc = childGridAcc.grid(batchIdx)->getAccessor();

    for (auto jt = OctChildrenIterator(ijk << 1); jt.isValid(); ++jt) {
        if (cGridAcc.isActive(*jt)) {
            childMask[offsetI] = false;
            break;
        }
    }
}

template <typename GridType, template <typename T, int32_t D> typename TensorAccessor>
_CPU_AND_GPU_CODE_ void conformingIJKCallback(
        int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx,
        const fvdb::detail::GridBatchImpl::Accessor<GridType> thisGridAcc,
        const TensorAccessor<bool, 1> childMask,
        TensorAccessor<int32_t, 2> outIJK, TensorAccessor<bool, 1> outMask) {

    const nanovdb::NanoGrid<GridType>* grid = thisGridAcc.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    auto gAcc = grid->getAccessor();

    if (!leaf.isActive(voxelIdx)) {
        return;
    }

    const int64_t offsetI = leaf.getValue(voxelIdx) - 1;
    nanovdb::Coord ijk = leaf.offsetToGlobalCoord(voxelIdx);

    for (auto ct = OctChildrenIterator(ijk); ct.isValid(); ++ct) {
        int64_t vidx = gAcc.getValue(*ct) - 1;
        int64_t oidx = offsetI * 8 + ct.getCount();

        if (vidx == -1 || childMask[vidx]) {
            outIJK[oidx][0] = ct->x();
            outIJK[oidx][1] = ct->y();
            outIJK[oidx][2] = ct->z();
            outMask[oidx] = true;
        } else {
            outMask[oidx] = false;
        }
    }
}

template <c10::DeviceType DeviceTag>
fvdb::JaggedTensor dispatchFlattenedGridIJK(c10::intrusive_ptr<fvdb::detail::GridBatchImpl> thisGridPtr,
                                            c10::intrusive_ptr<fvdb::detail::GridBatchImpl> childGridPtr,
                                            bool conforming) {

    torch::Tensor childMask = torch::ones(
        thisGridPtr->totalVoxels(), torch::TensorOptions().device(thisGridPtr->device()).dtype(torch::kBool));

    // Check if any voxel contains child.
    if (childGridPtr) {
        auto childMaskAcc = fvdb::tensorAccessor<DeviceTag, bool, 1>(childMask);

        FVDB_DISPATCH_GRID_TYPES(*thisGridPtr, [&] {
            auto childGridAcc = fvdb::gridBatchAccessor<DeviceTag, GridType>(*childGridPtr);
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                        fvdb::detail::GridBatchImpl::Accessor<GridType> thisGridAcc) {
                    childMaskCallback<GridType, fvdb::TorchRAcc32>(
                        batchIdx, leafIdx, voxelIdx, thisGridAcc, childGridAcc, childMaskAcc);
                };
                fvdb::forEachVoxelCUDA<GridType>(512, 1, *thisGridPtr, cb);
            } else {
                auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                        fvdb::detail::GridBatchImpl::Accessor<GridType> thisGridAcc) {
                    childMaskCallback<GridType, fvdb::TorchAcc>(
                        batchIdx, leafIdx, voxelIdx, thisGridAcc, childGridAcc, childMaskAcc);
                };
                fvdb::forEachVoxelCPU<GridType>(1, *thisGridPtr, cb);
            }
        });
    }

    if (!conforming) {
        // For non-conforming grid, simply remove those with kids
        torch::Tensor ijk = fvdb::detail::ops::dispatchActiveGridCoords<DeviceTag>(*thisGridPtr, true).jdata();
        return ijk.index({childMask});
    } else {
        // For conforming grid, should add all 8 siblings
        const torch::TensorOptions optsData = torch::TensorOptions().dtype(torch::kInt32).device(thisGridPtr->device());
        const torch::TensorOptions optsMask = torch::TensorOptions().dtype(torch::kBool).device(thisGridPtr->device());
        torch::Tensor outIJK = torch::empty({thisGridPtr->totalVoxels() * 8, 3}, optsData);
        torch::Tensor outMask = torch::zeros({thisGridPtr->totalVoxels() * 8}, optsMask);

        auto outIJKAcc = fvdb::tensorAccessor<DeviceTag, int32_t, 2>(outIJK);
        auto outMaskAcc = fvdb::tensorAccessor<DeviceTag, bool, 1>(outMask);
        auto childMaskAcc = fvdb::tensorAccessor<DeviceTag, bool, 1>(childMask);

        FVDB_DISPATCH_GRID_TYPES(*thisGridPtr, [&] {
            if constexpr (DeviceTag == torch::kCUDA) {
                auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                        fvdb::detail::GridBatchImpl::Accessor<GridType> thisGridAcc) {
                    conformingIJKCallback<GridType, fvdb::TorchRAcc32>(
                        batchIdx, leafIdx, voxelIdx, thisGridAcc, childMaskAcc, outIJKAcc, outMaskAcc);
                };
                fvdb::forEachVoxelCUDA<GridType>(512, 1, *thisGridPtr, cb);
            } else {
                auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t voxelIdx, int32_t, 
                        fvdb::detail::GridBatchImpl::Accessor<GridType> thisGridAcc) {
                    conformingIJKCallback<GridType, fvdb::TorchAcc>(
                        batchIdx, leafIdx, voxelIdx, thisGridAcc, childMaskAcc, outIJKAcc, outMaskAcc);
                };
                fvdb::forEachVoxelCPU<GridType>(1, *thisGridPtr, cb);
            }
        });

        outIJK = outIJK.index({outMask});
        return outIJK;
    }

}
