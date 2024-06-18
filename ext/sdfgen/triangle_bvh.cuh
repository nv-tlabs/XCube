/*
* Modified from Thomas Müller & Alex Evans, NVIDIA
*/

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "gpu_memory.cuh"

/*
 * Definition of the cuda Triangle class.
 */

inline __host__ __device__ float sign(float x) {
    return copysignf(1.0, x);
}

template <typename T>
__host__ __device__ T clamp(T val, T lower, T upper) {
    return val < lower ? lower : (upper < val ? upper : val);
}

template <typename T>
__host__ __device__ void host_device_swap(T& a, T& b) {
    T c(a); a=b; b=c;
}

struct Triangle {
    __host__ __device__ Eigen::Vector3f sample_uniform_position(const Eigen::Vector2f& sample) const {
        float sqrt_x = std::sqrt(sample.x());
        float factor0 = 1.0f - sqrt_x;
        float factor1 = sqrt_x * (1.0f - sample.y());
        float factor2 = sqrt_x * sample.y();

        return factor0 * a + factor1 * b + factor2 * c;
    }

    __host__ __device__ float surface_area() const {
        return 0.5f * Eigen::Vector3f((b - a).cross(c - a)).norm();
    }

    __host__ __device__ Eigen::Vector3f normal() const {
        return (b - a).cross(c - a).normalized();
    }

    // based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd, Eigen::Vector3f& n) const {
        Eigen::Vector3f v1v0 = b - a;
        Eigen::Vector3f v2v0 = c - a;
        Eigen::Vector3f rov0 = ro - a;
        n = v1v0.cross(v2v0);
        Eigen::Vector3f q = rov0.cross(rd);
        float d = 1.0f / rd.dot(n);
        float u = d * -q.dot(v2v0);
        float v = d *  q.dot(v1v0);
        float t = d * -n.dot(rov0);
        if (u < 0.0f || u > 1.0f || v < 0.0f || (u+v) > 1.0f || t < 0.0f) {
            t = std::numeric_limits<float>::max(); // No intersection
        }
        return t;
    }

    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd) const {
        Eigen::Vector3f n;
        return ray_intersect(ro, rd, n);
    }

    // based on https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
    __host__ __device__ float distance_sq(const Eigen::Vector3f& pos) const {
        Eigen::Vector3f v21 = b - a; Eigen::Vector3f p1 = pos - a;
        Eigen::Vector3f v32 = c - b; Eigen::Vector3f p2 = pos - b;
        Eigen::Vector3f v13 = a - c; Eigen::Vector3f p3 = pos - c;
        Eigen::Vector3f nor = v21.cross(v13);

        return
            // inside/outside test
                (sign(v21.cross(nor).dot(p1)) + sign(v32.cross(nor).dot(p2)) + sign(v13.cross(nor).dot(p3)) < 2.0f)
                ?
                // 3 edges
                std::min({
                     (v21 * clamp(v21.dot(p1) / v21.squaredNorm(), 0.0f, 1.0f)-p1).squaredNorm(),
                     (v32 * clamp(v32.dot(p2) / v32.squaredNorm(), 0.0f, 1.0f)-p2).squaredNorm(),
                     (v13 * clamp(v13.dot(p3) / v13.squaredNorm(), 0.0f, 1.0f)-p3).squaredNorm(),
                 })
                :
                // 1 face
                nor.dot(p1)*nor.dot(p1)/nor.squaredNorm();
    }

    __host__ __device__ float distance(const Eigen::Vector3f& pos) const {
        return std::sqrt(distance_sq(pos));
    }

    __host__ __device__ bool point_in_triangle(const Eigen::Vector3f& p) const {
        // Move the triangle so that the point becomes the
        // triangles origin
        Eigen::Vector3f local_a = a - p;
        Eigen::Vector3f local_b = b - p;
        Eigen::Vector3f local_c = c - p;

        // The point should be moved too, so they are both
        // relative, but because we don't use p in the
        // equation anymore, we don't need it!
        // p -= p;

        // Compute the normal vectors for triangles:
        // u = normal of PBC
        // v = normal of PCA
        // w = normal of PAB

        Eigen::Vector3f u = local_b.cross(local_c);
        Eigen::Vector3f v = local_c.cross(local_a);
        Eigen::Vector3f w = local_a.cross(local_b);

        // Test to see if the normals are facing the same direction.
        // If yes, the point is inside, otherwise it isn't.
        return u.dot(v) >= 0.0f && u.dot(w) >= 0.0f;
    }

    __host__ __device__ Eigen::Vector3f closest_point_to_line(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) const {
        float t = (c - a).dot(b-a) / (b-a).dot(b-a);
        t = std::max(std::min(t, 1.0f), 0.0f);
        return a + t * (b - a);
    }

    __host__ __device__ Eigen::Vector3f closest_point(Eigen::Vector3f point) const {
        point -= normal().dot(point - a) * normal();

        if (point_in_triangle(point)) {
            return point;
        }

        Eigen::Vector3f c1 = closest_point_to_line(a, b, point);
        Eigen::Vector3f c2 = closest_point_to_line(b, c, point);
        Eigen::Vector3f c3 = closest_point_to_line(c, a, point);

        float mag1 = (point - c1).squaredNorm();
        float mag2 = (point - c2).squaredNorm();
        float mag3 = (point - c3).squaredNorm();

        float min = std::min({mag1, mag2, mag3});

        if (min == mag1) {
            return c1;
        } else if (min == mag2) {
            return c2;
        } else {
            return c3;
        }
    }

    __host__ __device__ Eigen::Vector3f centroid() const {
        return (a + b + c) / 3.0f;
    }

    __host__ __device__ float centroid(int axis) const {
        return (a[axis] + b[axis] + c[axis]) / 3;
    }

    __host__ __device__ void get_vertices(Eigen::Vector3f v[3]) const {
        v[0] = a;
        v[1] = b;
        v[2] = c;
    }

    Eigen::Vector3f a, b, c;
};

inline std::ostream& operator<<(std::ostream& os, const Triangle& triangle) {
    os << "[";
    os << "a=[" << triangle.a.x() << "," << triangle.a.y() << "," << triangle.a.z() << "], ";
    os << "b=[" << triangle.b.x() << "," << triangle.b.y() << "," << triangle.b.z() << "], ";
    os << "c=[" << triangle.c.x() << "," << triangle.c.y() << "," << triangle.c.z() << "]";
    os << "]";
    return os;
}

/*
 * Definition of the cuda BoundingBox class.
 */

template <int N_POINTS>
__host__ __device__ inline void project(Eigen::Vector3f points[N_POINTS], const Eigen::Vector3f& axis, float& min, float& max) {
    min = std::numeric_limits<float>::infinity();
    max = -std::numeric_limits<float>::infinity();

#pragma unroll
    for (uint32_t i = 0; i < N_POINTS; ++i) {
        float val = axis.dot(points[i]);

        if (val < min) {
            min = val;
        }

        if (val > max) {
            max = val;
        }
    }
}

struct BoundingBox {
    __host__ __device__ BoundingBox() {}

    __host__ __device__ BoundingBox(const Eigen::Vector3f& a, const Eigen::Vector3f& b) : min{a}, max{b} {}

    __host__ __device__ explicit BoundingBox(const Triangle& tri) {
        min = max = tri.a;
        enlarge(tri.b);
        enlarge(tri.c);
    }

    BoundingBox(std::vector<Triangle>::const_iterator begin, std::vector<Triangle>::const_iterator end) {
        min = max = begin->a;
        for (auto it = begin; it != end; ++it) {
            enlarge(*it);
        }
    }

    __host__ __device__ void enlarge(const BoundingBox& other) {
        min = min.cwiseMin(other.min);
        max = max.cwiseMax(other.max);
    }

    __host__ __device__ void enlarge(const Triangle& tri) {
        enlarge(tri.a);
        enlarge(tri.b);
        enlarge(tri.c);
    }

    __host__ __device__ void enlarge(const Eigen::Vector3f& point) {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }

    __host__ __device__ void inflate(float amount) {
        min -= Eigen::Vector3f::Constant(amount);
        max += Eigen::Vector3f::Constant(amount);
    }

    __host__ __device__ Eigen::Vector3f diag() const {
        return max - min;
    }

    __host__ __device__ Eigen::Vector3f relative_pos(const Eigen::Vector3f& pos) const {
        return (pos - min).cwiseQuotient(diag());
    }

    __host__ __device__ Eigen::Vector3f center() const {
        return 0.5f * (max + min);
    }

    __host__ __device__ BoundingBox intersection(const BoundingBox& other) const {
        BoundingBox result = *this;
        result.min = result.min.cwiseMax(other.min);
        result.max = result.max.cwiseMin(other.max);
        return result;
    }

    __host__ __device__ bool intersects(const BoundingBox& other) const {
        return !intersection(other).is_empty();
    }

    // Based on the separating axis theorem
    // (https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf)
    // Code adapted from a C# implementation at stack overflow
    // https://stackoverflow.com/a/17503268
    __host__ __device__ bool intersects(const Triangle& triangle) const {
        float triangle_min, triangle_max;
        float box_min, box_max;

        // Test the box normals (x-, y- and z-axes)
        Eigen::Vector3f box_normals[3] = {
                Eigen::Vector3f{1.0f, 0.0f, 0.0f},
                Eigen::Vector3f{0.0f, 1.0f, 0.0f},
                Eigen::Vector3f{0.0f, 0.0f, 1.0f},
        };

        Eigen::Vector3f triangle_normal = triangle.normal();
        Eigen::Vector3f triangle_verts[3];
        triangle.get_vertices(triangle_verts);

        for (int i = 0; i < 3; i++) {
            project<3>(triangle_verts, box_normals[i], triangle_min, triangle_max);
            if (triangle_max < min[i] || triangle_min > max[i]) {
                return false; // No intersection possible.
            }
        }

        Eigen::Vector3f verts[8];
        get_vertices(verts);

        // Test the triangle normal
        float triangle_offset = triangle_normal.dot(triangle.a);
        project<8>(verts, triangle_normal, box_min, box_max);
        if (box_max < triangle_offset || box_min > triangle_offset) {
            return false; // No intersection possible.
        }

        // Test the nine edge cross-products
        Eigen::Vector3f edges[3] = {
                triangle.a - triangle.b,
                triangle.a - triangle.c,
                triangle.b - triangle.c,
        };

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // The box normals are the same as it's edge tangents
                Eigen::Vector3f axis = edges[i].cross(box_normals[j]);
                project<8>(verts, axis, box_min, box_max);
                project<3>(triangle_verts, axis, triangle_min, triangle_max);
                if (box_max < triangle_min || box_min > triangle_max)
                    return false; // No intersection possible
            }
        }

        // No separating axis found.
        return true;
    }

    __host__ __device__ Eigen::Vector2f ray_intersect(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir) const {
        float tmin = (min.x() - pos.x()) / dir.x();
        float tmax = (max.x() - pos.x()) / dir.x();

        if (tmin > tmax) {
            host_device_swap(tmin, tmax);
        }

        float tymin = (min.y() - pos.y()) / dir.y();
        float tymax = (max.y() - pos.y()) / dir.y();

        if (tymin > tymax) {
            host_device_swap(tymin, tymax);
        }

        if (tmin > tymax || tymin > tmax) {
            return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        }

        if (tymin > tmin)
            tmin = tymin;

        if (tymax < tmax)
            tmax = tymax;

        float tzmin = (min.z() - pos.z()) / dir.z();
        float tzmax = (max.z() - pos.z()) / dir.z();

        if (tzmin > tzmax) {
            host_device_swap(tzmin, tzmax);
        }

        if (tmin > tzmax || tzmin > tmax) {
            return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        }

        if (tzmin > tmin) {
            tmin = tzmin;
        }

        if (tzmax < tmax) {
            tmax = tzmax;
        }

        return { tmin, tmax };
    }

    __host__ __device__ bool is_empty() const {
        return (max.array() < min.array()).any();
    }

    __host__ __device__ bool contains(const Eigen::Vector3f& p) const {
        return
                p.x() >= min.x() && p.x() <= max.x() &&
                p.y() >= min.y() && p.y() <= max.y() &&
                p.z() >= min.z() && p.z() <= max.z();
    }

    /// Calculate the squared point-AABB distance
    __host__ __device__ float distance(const Eigen::Vector3f& p) const {
        return sqrt(distance_sq(p));
    }

    __host__ __device__ float distance_sq(const Eigen::Vector3f& p) const {
        return (min - p).cwiseMax(p - max).cwiseMax(0.0f).squaredNorm();
    }

    __host__ __device__ float signed_distance(const Eigen::Vector3f& p) const {
        Eigen::Vector3f q = (p - min).cwiseAbs() - diag();
        return q.cwiseMax(0.0f).norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0f);
    }

    __host__ __device__ void get_vertices(Eigen::Vector3f v[8]) const {
        v[0] = {min.x(), min.y(), min.z()};
        v[1] = {min.x(), min.y(), max.z()};
        v[2] = {min.x(), max.y(), min.z()};
        v[3] = {min.x(), max.y(), max.z()};
        v[4] = {max.x(), min.y(), min.z()};
        v[5] = {max.x(), min.y(), max.z()};
        v[6] = {max.x(), max.y(), min.z()};
        v[7] = {max.x(), max.y(), max.z()};
    }

    Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
};

inline std::ostream& operator<<(std::ostream& os, const BoundingBox& bb) {
    os << "[";
    os << "min=[" << bb.min.x() << "," << bb.min.y() << "," << bb.min.z() << "], ";
    os << "max=[" << bb.max.x() << "," << bb.max.y() << "," << bb.max.z() << "]";
    os << "]";
    return os;
}

struct TriangleBvhNode {
    BoundingBox bb;
    int left_idx; // negative values indicate leaves
    int right_idx;
};

template <typename T, int MAX_SIZE=32>
class FixedStack {
public:
    __host__ __device__ void push(T val) {
        if (m_count >= MAX_SIZE-1) {
            printf("WARNING TOO BIG\n");
        }
        m_elems[m_count++] = val;
    }

    __host__ __device__ T pop() {
        return m_elems[--m_count];
    }

    __host__ __device__ bool empty() const {
        return m_count <= 0;
    }

private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
};

using FixedIntStack = FixedStack<int>;

/*
 * Definition of the TriangleBvh class.
 */

enum class EMeshSdfMode : int {
    Watertight,
    Raystab,
    PathEscape,
};
static constexpr const char* MeshSdfModeStr = "Watertight\0Raystab\0PathEscape\0\0";

constexpr uint32_t n_threads_linear = 128;

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

template <typename T>
constexpr uint32_t n_blocks_linear(T n_elements) {
    return (uint32_t)div_round_up(n_elements, (T)n_threads_linear);
}

template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
    if (n_elements <= 0) {
        return;
    }
    kernel<<<n_blocks_linear(n_elements), n_threads_linear, shmem_size, stream>>>((uint32_t)n_elements, args...);
}

class TriangleBvh {
public:
    virtual void signed_distance_gpu(uint32_t n_elements, EMeshSdfMode mode, const Eigen::Vector3f* gpu_positions,
                                     float* gpu_distances, const Triangle* gpu_triangles, bool use_existing_distances_as_upper_bounds,
                                     cudaStream_t stream) = 0;
    virtual void ray_trace_gpu(uint32_t n_elements, Eigen::Vector3f* gpu_positions, Eigen::Vector3f* gpu_directions, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual bool touches_triangle(const BoundingBox& bb, const Triangle* __restrict__ triangles) const = 0;
    virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;
    virtual void build_optix(const GPUMemory<Triangle>& triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvh> make();

    TriangleBvhNode* nodes_gpu() const {
        return m_nodes_gpu.data();
    }

protected:
    std::vector<TriangleBvhNode> m_nodes;
    GPUMemory<TriangleBvhNode> m_nodes_gpu;
    TriangleBvh() {};
};

#ifdef NGP_OPTIX

#include <optix.h>
#include <optix_host.h>
#include <optix_stack_size.h>

/*
 * Definition of Optix related things.
 */

#define OPTIX_CHECK_THROW(x)                                                                                 \
	do {                                                                                                     \
		OptixResult res = x;                                                                                 \
		if (res != OPTIX_SUCCESS) {                                                                          \
			throw std::runtime_error(std::string("Optix call '" #x "' failed."));                            \
		}                                                                                                    \
	} while(0)

#define OPTIX_CHECK_THROW_LOG(x)                                                                                                                          \
	do {                                                                                                                                                  \
		OptixResult res = x;                                                                                                                              \
		const size_t sizeof_log_returned = sizeof_log;                                                                                                    \
		sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */                                                                               \
		if (res != OPTIX_SUCCESS) {                                                                                                                       \
			throw std::runtime_error(std::string("Optix call '" #x "' failed. Log:\n") + log + (sizeof_log_returned == sizeof_log ? "" : "<truncated>")); \
		}                                                                                                                                                 \
	} while(0)


namespace optix {
    template <typename T>
    struct SbtRecord {
        __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    template <typename T>
    class Program {
    public:
        Program(const char* data, size_t size, OptixDeviceContext optix) {
            char log[2048]; // For error reporting from OptiX creation functions
            size_t sizeof_log = sizeof(log);

            // Module from PTX
            OptixModule optix_module = nullptr;
            OptixPipelineCompileOptions pipeline_compile_options = {};
            {
                // Default options for our module.
                OptixModuleCompileOptions module_compile_options = {};

                // Pipeline options must be consistent for all modules used in a
                // single pipeline
                pipeline_compile_options.usesMotionBlur = false;

                // This option is important to ensure we compile code which is optimal
                // for our scene hierarchy. We use a single GAS � no instancing or
                // multi-level hierarchies
                pipeline_compile_options.traversableGraphFlags =
                        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

                // Our device code uses 3 payload registers (r,g,b output value)
                pipeline_compile_options.numPayloadValues = 3;

                // This is the name of the param struct variable in our device code
                pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

                OPTIX_CHECK_THROW_LOG(optixModuleCreateFromPTX(
                        optix,
                        &module_compile_options,
                        &pipeline_compile_options,
                        data,
                        size,
                        log,
                        &sizeof_log,
                        &optix_module
                ));
            }

            // Program groups
            OptixProgramGroup raygen_prog_group   = nullptr;
            OptixProgramGroup miss_prog_group     = nullptr;
            OptixProgramGroup hitgroup_prog_group = nullptr;
            {
                OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

                OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
                raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
                raygen_prog_group_desc.raygen.module            = optix_module;
                raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
                OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
                        optix,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &raygen_prog_group
                ));

                OptixProgramGroupDesc miss_prog_group_desc  = {};
                miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
                miss_prog_group_desc.miss.module            = optix_module;
                miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
                OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
                        optix,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &miss_prog_group
                ));

                OptixProgramGroupDesc hitgroup_prog_group_desc = {};
                hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                hitgroup_prog_group_desc.hitgroup.moduleCH            = optix_module;
                hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
                OPTIX_CHECK_THROW_LOG(optixProgramGroupCreate(
                        optix,
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &hitgroup_prog_group
                ));
            }

            // Linking
            {
                const uint32_t max_trace_depth = 1;
                OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

                OptixPipelineLinkOptions pipeline_link_options = {};
                pipeline_link_options.maxTraceDepth = max_trace_depth;
                pipeline_link_options.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

                OPTIX_CHECK_THROW_LOG(optixPipelineCreate(
                        optix,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof(program_groups) / sizeof(program_groups[0]),
                        log,
                        &sizeof_log,
                        &m_pipeline
                ));

                OptixStackSizes stack_sizes = {};
                for (auto& prog_group : program_groups) {
                    OPTIX_CHECK_THROW(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
                }

                uint32_t direct_callable_stack_size_from_traversal;
                uint32_t direct_callable_stack_size_from_state;
                uint32_t continuation_stack_size;
                OPTIX_CHECK_THROW(optixUtilComputeStackSizes(
                        &stack_sizes, max_trace_depth,
                        0,  // maxCCDepth
                        0,  // maxDCDEpth
                        &direct_callable_stack_size_from_traversal,
                        &direct_callable_stack_size_from_state, &continuation_stack_size
                ));
                OPTIX_CHECK_THROW(optixPipelineSetStackSize(
                        m_pipeline, direct_callable_stack_size_from_traversal,
                        direct_callable_stack_size_from_state, continuation_stack_size,
                        1  // maxTraversableDepth
                ));
            }

            // Shader binding table
            {
                CUdeviceptr raygen_record;
                const size_t raygen_record_size = sizeof(SbtRecord<typename T::RayGenData>);
                CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
                SbtRecord<typename T::RayGenData> rg_sbt;
                OPTIX_CHECK_THROW(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
                CUDA_CHECK_THROW(cudaMemcpy(
                        reinterpret_cast<void*>(raygen_record),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                ));

                CUdeviceptr miss_record;
                size_t miss_record_size = sizeof(SbtRecord<typename T::MissData>);
                CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
                SbtRecord<typename T::MissData> ms_sbt;
                OPTIX_CHECK_THROW(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
                CUDA_CHECK_THROW(cudaMemcpy(
                        reinterpret_cast<void*>(miss_record),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                ));

                CUdeviceptr hitgroup_record;
                size_t hitgroup_record_size = sizeof(SbtRecord<typename T::HitGroupData>);
                CUDA_CHECK_THROW(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
                SbtRecord<typename T::HitGroupData> hg_sbt;
                OPTIX_CHECK_THROW(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
                CUDA_CHECK_THROW(cudaMemcpy(
                        reinterpret_cast<void*>(hitgroup_record),
                        &hg_sbt,
                        hitgroup_record_size,
                        cudaMemcpyHostToDevice
                ));

                m_sbt.raygenRecord                = raygen_record;
                m_sbt.missRecordBase              = miss_record;
                m_sbt.missRecordStrideInBytes     = sizeof(SbtRecord<typename T::MissData>);
                m_sbt.missRecordCount             = 1;
                m_sbt.hitgroupRecordBase          = hitgroup_record;
                m_sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<typename T::HitGroupData>);
                m_sbt.hitgroupRecordCount         = 1;
            }
        }

        void invoke(const typename T::Params& params, const uint3& dim, cudaStream_t stream) {
            CUDA_CHECK_THROW(cudaMemcpyAsync(m_params_gpu.data(), &params, sizeof(typename T::Params), cudaMemcpyHostToDevice, stream));
            OPTIX_CHECK_THROW(optixLaunch(m_pipeline, stream, (CUdeviceptr)(uintptr_t)m_params_gpu.data(), sizeof(typename T::Params), &m_sbt, dim.x, dim.y, dim.z));
        }

    private:
        OptixShaderBindingTable m_sbt = {};
        OptixPipeline m_pipeline = nullptr;
        GPUMemory<typename T::Params> m_params_gpu = GPUMemory<typename T::Params>(1);
    };
}


struct Raytrace {
    struct Params
    {
        Eigen::Vector3f* ray_origins;
        Eigen::Vector3f* ray_directions;
        const Triangle* triangles;
        OptixTraversableHandle handle;
    };

    struct RayGenData {};
    struct MissData {};
    struct HitGroupData {};
};

struct Raystab {
    struct Params
    {
        const Eigen::Vector3f* ray_origins;
        float* distances;
        OptixTraversableHandle handle;
    };

    struct RayGenData {};
    struct MissData {};
    struct HitGroupData {};
};

struct PathEscape {
    struct Params
    {
        const Eigen::Vector3f* ray_origins;
        const Triangle* triangles;
        float* distances;
        OptixTraversableHandle handle;
    };

    struct RayGenData {};
    struct MissData {};
    struct HitGroupData {};
};

#endif
