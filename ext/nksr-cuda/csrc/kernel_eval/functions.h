#include <torch/extension.h>
#include <torch/autograd.h>

#include <fvdb/GridBatch.h>
#include "../common/iter_util.h"

using variable_list = torch::autograd::variable_list;
using AutogradContext = torch::autograd::AutogradContext;
using Variable = torch::autograd::Variable;

template <c10::DeviceType DeviceTag, typename scalar_t>
_CPU_AND_GPU_CODE_ __forceinline__ void atomicAddIfGPU(scalar_t* tensor, scalar_t value) {
    if constexpr (DeviceTag == torch::kCUDA) {
        gpuAtomicAddNoReturn(tensor, value);
    } else {
        (*tensor) += value;
    }
}

#define NKSR_DISPATCH_INTEGER_TYPES(SCALAR_TYPE, ...)                   \
    [&]() {                                                             \
        if ((SCALAR_TYPE) == torch::kInt32) {                           \
            using index_t = int32_t;                                    \
            return __VA_ARGS__();                                       \
        } else if ((SCALAR_TYPE) == torch::kLong) {                     \
            using index_t = int64_t;                                    \
            return __VA_ARGS__();                                       \
        }  else {                                                       \
            TORCH_CHECK(false, "Only int32 and int64 are supported");   \
        }                                                               \
    }()

template <typename ScalarT>
_CPU_AND_GPU_CODE_ inline ScalarT bezier_1dim(const ScalarT x) {
    static const ScalarT OPF = 1.5;
    static const ScalarT PF = 0.5;

    bool r1 = x < -OPF, r2 = x < -PF, r3 = x < PF, r4 = x < OPF;
    if (!r1 && r2) {
        return (x + OPF) * (x + OPF);
    } else if (!r2 && r3) {
        return -(ScalarT) 2 * x * x + OPF;
    } else if (!r3 && r4) {
        return (x - OPF) * (x - OPF);
    }
    return 0.0;
}

template <typename ScalarT>
_CPU_AND_GPU_CODE_ inline ScalarT bezier_grad_1dim(const ScalarT x) {
    bool r1 = x < -1.5, r2 = x < -0.5, r3 = x < 0.5, r4 = x < 1.5;
    if (!r1 && r2) {
        return 2 * x + 3;
    } else if (!r2 && r3) {
        return -4 * x;
    } else if (!r3 && r4) {
        return 2 * x - 3;
    }
    return 0.0;
}

template <typename ScalarT>
_CPU_AND_GPU_CODE_ inline bool has_overlap(
        const nanovdb::math::Vec3<ScalarT>& a_min, const nanovdb::math::Vec3<ScalarT>& a_max,
        const nanovdb::math::Vec3<ScalarT>& b_min, const nanovdb::math::Vec3<ScalarT>& b_max) {
    return (a_max[0] >= b_min[0] && b_max[0] >= a_min[0]) &&
           (a_max[1] >= b_min[1] && b_max[1] >= a_min[1]) &&
           (a_max[2] >= b_min[2] && b_max[2] >= a_min[2]);
}

template <typename ScalarT, typename Acc2, typename Acc3>
_CPU_AND_GPU_CODE_ inline void kernel_grad_evaluation_fwd(
        const int64_t offset, const int64_t pi, ScalarT scale,
        const ScalarT& pcx, const ScalarT& pcy, const ScalarT& pcz,
        const Acc2& pKernel,
        const Acc2& cKernel,
        const Acc3& gradKernelQuery,
        bool grad,
        // Output (necessary)
        ScalarT& f, nanovdb::math::Vec3<ScalarT>& gradF,
        // Output (for backward)
        ScalarT& bezierKernel, ScalarT& dpKernel, nanovdb::math::Vec3<ScalarT>& db) {

    // Kernel evaluation
    ScalarT bx = bezier_1dim(pcx), by = bezier_1dim(pcy), bz = bezier_1dim(pcz);
    bezierKernel = bx * by * bz;

    dpKernel = static_cast<ScalarT>(1.0);
    // For the Dot-Product part it could be a trivial one.
    if (cKernel.size(1) > 0) {
        dpKernel = static_cast<ScalarT>(0.0);
        for (int64_t k = 0; k < cKernel.size(1); ++k) {
//            dpKernel = fmaf(cKernel[offset][k], pKernel[pi][k], dpKernel);
            dpKernel += cKernel[offset][k] * pKernel[pi][k];
        }
    }
    f = bezierKernel * dpKernel;

    // Gradient Kernel evaluation
    if (!grad) return;

    ScalarT dbx = bezier_grad_1dim(pcx) * scale;
    ScalarT dby = bezier_grad_1dim(pcy) * scale;
    ScalarT dbz = bezier_grad_1dim(pcz) * scale;
    db[0] = dbx * by * bz;
    db[1] = bx * dby * bz;
    db[2] = bx * by * dbz;

#pragma unroll
    for (int dim = 0; dim < 3; ++ dim) {
        auto gradDot = static_cast<ScalarT>(0.0);
        if (cKernel.size(1) > 0 && gradKernelQuery.size(0) > 0) {
            for (int64_t k = 0; k < cKernel.size(1); ++k) {
                gradDot += cKernel[offset][k] * gradKernelQuery[pi][k][dim];
            }
        }
        gradF[dim] = bezierKernel * gradDot + dpKernel * db[dim];
    }
}

template <c10::DeviceType DeviceTag, typename ScalarT, bool isCompound, typename Acc1, typename Acc2, typename Acc3>
_CPU_AND_GPU_CODE_ inline void kernel_grad_evaluation_bwd(
        const int64_t offset, const int64_t pi,
        const Acc2& pKernel, const Acc2& cKernel,
        const Acc3& gradKernelQuery,
        // Upstream gradient input
        const Acc1& gradOutFunc,
        const Acc2& gradOutGradFunc,    // not used if chaining
        bool grad, const ScalarT alpha,
        // Downstream gradient output
        Acc2& gradQueryKernel, Acc2& gradGridKernel,
        Acc1& gradGridAlpha, Acc3& gradGradKernelQuery,
        // Compound input
        int64_t outIdx,
        const ScalarT of, const nanovdb::math::Vec3<ScalarT>& gradOF,
        // My forward output
        const ScalarT f, const nanovdb::math::Vec3<ScalarT>& gradF,
        const ScalarT& bezierKernel, const ScalarT& dpKernel, const nanovdb::math::Vec3<ScalarT>& db
) {

    if (!grad) {
        ScalarT gradUpper;
        if (!isCompound) { gradUpper = gradOutFunc[pi] * alpha; }
        else {
            if (outIdx == -1) {
                gradUpper = of;
            } else {
                gradUpper = gradOutFunc[outIdx] * of;
            }
        }

        if (cKernel.size(1) > 0) {
            for (int64_t k = 0; k < cKernel.size(1); ++k) {
                ScalarT dfunc = gradUpper * bezierKernel;
                atomicAddIfGPU<DeviceTag>(&gradQueryKernel[pi][k], dfunc * cKernel[offset][k]);
                atomicAddIfGPU<DeviceTag>(&gradGridKernel[offset][k], dfunc * pKernel[pi][k]);
            }
        }
        if (!isCompound) {
            atomicAddIfGPU<DeviceTag>(&gradGridAlpha[offset], f * gradOutFunc[pi]);
        }
    } else {

#pragma unroll
        for (int dim = 0; dim < 3; ++ dim) {
            ScalarT dalpha;
            if (!isCompound) { dalpha = gradOutGradFunc[pi][dim] * alpha; }
            else {
                if (outIdx == -1) {
                    dalpha = gradOF[dim];
                } else {
                    dalpha = gradOutFunc[outIdx] * gradOF[dim];
                }
            }

            if (cKernel.size(1) > 0) {
                for (int64_t k = 0; k < cKernel.size(1); ++k) {
                    atomicAddIfGPU<DeviceTag>(&gradQueryKernel[pi][k], dalpha * cKernel[offset][k] * db[dim]);
                    atomicAddIfGPU<DeviceTag>(&gradGridKernel[offset][k], dalpha * pKernel[pi][k] * db[dim]);

                    if (gradKernelQuery.size(0) > 0) {
                        atomicAddIfGPU<DeviceTag>(&gradGridKernel[offset][k], dalpha * bezierKernel * gradKernelQuery[pi][k][dim]);
                        atomicAddIfGPU<DeviceTag>(&gradGradKernelQuery[pi][k][dim], dalpha * bezierKernel * cKernel[offset][k]);
                    }
                }
            }

            if (!isCompound) {
                atomicAddIfGPU<DeviceTag>(&gradGridAlpha[offset], gradF[dim] * gradOutGradFunc[pi][dim]);
            }
        }
    }

}

fvdb::JaggedTensor buildCOOIndexer(
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> iSVH,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> jSVH);

struct RhsEvaluation : public torch::autograd::Function<RhsEvaluation> {

    static variable_list forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
        fvdb::JaggedTensor pts, Variable ptsKernel,
        Variable gridKernel, Variable gradKernelPts,
        Variable ptsData);

    static variable_list backward(
        AutogradContext *ctx,
        variable_list grad_output);
};


struct KBuilding : public torch::autograd::Function<KBuilding> {
    static variable_list forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
        Variable kernel, Variable indexMap, int64_t numEntries);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};


struct CsrMatrixMultiplication : public torch::autograd::Function<CsrMatrixMultiplication> {
    static variable_list forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> iGrid,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> jGrid,
        Variable iCoords, Variable jCoords,
        Variable iValue, Variable jValue,
        Variable iRowPtr, Variable jRowPtr,
        Variable iColInds, Variable jColInds,
        Variable indexMap, int64_t numEntries);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};


struct QgBuilding : public torch::autograd::Function<QgBuilding> {
    static variable_list forward(
        AutogradContext *ctx,
        c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
        fvdb::JaggedTensor pts, Variable ptsKernel,
        Variable gridKernel, Variable gradKernelPts,
        bool grad);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};


struct KernelEvaluation : public torch::autograd::Function<KernelEvaluation> {
    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<fvdb::detail::GridBatchImpl> grid,
                                 bool grad,
                                 fvdb::JaggedTensor query, Variable queryKernel,
                                 Variable gridKernel, Variable gridAlpha,
                                 Variable gradKernelQuery);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};


struct MatrixBuilding : public torch::autograd::Function<MatrixBuilding> {
    static variable_list forward(AutogradContext *ctx,
                                 c10::intrusive_ptr<fvdb::detail::GridBatchImpl> iGrid,
                                 c10::intrusive_ptr<fvdb::detail::GridBatchImpl> jGrid,
                                 fvdb::JaggedTensor ptsPos,
                                 Variable ptsKernelI, Variable ptsKernelJ,
                                 Variable iKernel, Variable jKernel,
                                 Variable gradPtsKernelPosI, Variable gradPtsKernelPosJ,
                                 Variable indexMap,
                                 bool grad, int64_t numEntries);

    static variable_list backward(AutogradContext *ctx, variable_list grad_output);
};
