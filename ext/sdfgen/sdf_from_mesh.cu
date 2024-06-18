#include <torch/torch.h>
#include "triangle_bvh.cuh"
#include <ATen/cuda/CUDAContext.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor")

torch::Tensor sdf_from_mesh(const torch::Tensor& queries, const torch::Tensor& ref_triangles, EMeshSdfMode mode) {
    CHECK_INPUT(queries); CHECK_IS_FLOAT(queries)
    CHECK_CONTIGUOUS(ref_triangles); CHECK_IS_FLOAT(ref_triangles)

    std::vector<Triangle> triangles_cpu(ref_triangles.size(0));
    CUDA_CHECK_THROW(cudaMemcpy((float*) triangles_cpu.data(), (float*) ref_triangles.data_ptr(),
                                ref_triangles.nbytes(),
                                ref_triangles.is_cuda() ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost));

    std::shared_ptr<TriangleBvh> triangle_bvh = TriangleBvh::make();
    triangle_bvh->build(triangles_cpu, 8);

    // Triangles are altered during tree building. So we make another copy.
    GPUMemory<Triangle> triangles_gpu;
    triangles_gpu.resize_and_copy_from_host(triangles_cpu);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    triangle_bvh->build_optix(triangles_gpu, stream);

    // Setting initial sdf to its maximum possible value will accelerate look up by directly ignoring some early branches.
    torch::Tensor sdf = torch::full(queries.size(0), 1.0e6f, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    triangle_bvh->signed_distance_gpu(
            queries.size(0),
            mode,
            (Eigen::Vector3f*) queries.data_ptr(),
            (float*) sdf.data_ptr(),
            triangles_gpu.data(),
            true,
            stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return sdf;
}
