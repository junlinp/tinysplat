#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

struct RasterParams {
    uint32_t num_gaussians;
    uint32_t num_channels;
    uint32_t height;
    uint32_t width;
};

constexpr const char* kGaussianSplat2DShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct RasterParams {
    uint num_gaussians;
    uint num_channels;
    uint height;
    uint width;
};

kernel void gaussian_splat_2d_forward(
    constant float2* means [[buffer(0)]],
    constant float4* covariances [[buffer(1)]],
    constant float* colors [[buffer(2)]],
    constant float* opacities [[buffer(3)]],
    constant RasterParams& params [[buffer(4)]],
    device float* output [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const float x = static_cast<float>(gid.x);
    const float y = static_cast<float>(gid.y);
    float numerator0 = 0.0f;
    float numerator1 = 0.0f;
    float numerator2 = 0.0f;
    float numerator3 = 0.0f;
    float total_weight = 0.0f;

    for (uint i = 0; i < params.num_gaussians; ++i) {
        const float2 mean = means[i];
        const float4 cov = covariances[i];
        const float cov_xx = cov.x;
        const float cov_xy = cov.y;
        const float cov_yx = cov.z;
        const float cov_yy = cov.w;

        const float det = cov_xx * cov_yy - cov_xy * cov_yx;
        if (det <= 1e-8f) {
            continue;
        }

        const float inv_det = 1.0f / det;
        const float inv_xx = cov_yy * inv_det;
        const float inv_xy = -cov_xy * inv_det;
        const float inv_yx = -cov_yx * inv_det;
        const float inv_yy = cov_xx * inv_det;

        const float dx = x - mean.x;
        const float dy = y - mean.y;
        const float quad =
            dx * (inv_xx * dx + inv_xy * dy) +
            dy * (inv_yx * dx + inv_yy * dy);

        const float gaussian = exp(-0.5f * quad) / (2.0f * 3.14159265359f * sqrt(det + 1e-8f));
        const float weight = gaussian * opacities[i];

        total_weight += weight;
        const uint color_offset = i * params.num_channels;
        if (params.num_channels > 0) numerator0 += weight * colors[color_offset];
        if (params.num_channels > 1) numerator1 += weight * colors[color_offset + 1];
        if (params.num_channels > 2) numerator2 += weight * colors[color_offset + 2];
        if (params.num_channels > 3) numerator3 += weight * colors[color_offset + 3];
    }

    const float denom = max(total_weight, 1e-8f);
    const uint out_offset = (gid.y * params.width + gid.x) * params.num_channels;

    if (params.num_channels > 0) output[out_offset] = numerator0 / denom;
    if (params.num_channels > 1) output[out_offset + 1] = numerator1 / denom;
    if (params.num_channels > 2) output[out_offset + 2] = numerator2 / denom;
    if (params.num_channels > 3) output[out_offset + 3] = numerator3 / denom;

    if (params.num_channels == 4) {
        output[out_offset] *= output[out_offset + 3];
        output[out_offset + 1] *= output[out_offset + 3];
        output[out_offset + 2] *= output[out_offset + 3];
    }
}
)METAL";

class MetalRasterizer2D {
public:
    MetalRasterizer2D() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (device_ == nil) {
                throw std::runtime_error("Metal device is unavailable.");
            }

            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:kGaussianSplat2DShaderSource];
            id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&error];
            if (library == nil) {
                throw std::runtime_error(std::string("Failed to compile Metal shader: ") + [[error localizedDescription] UTF8String]);
            }

            id<MTLFunction> function = [library newFunctionWithName:@"gaussian_splat_2d_forward"];
            if (function == nil) {
                throw std::runtime_error("Metal function gaussian_splat_2d_forward was not found.");
            }

            pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
            if (pipeline_ == nil) {
                throw std::runtime_error(std::string("Failed to create Metal compute pipeline: ") + [[error localizedDescription] UTF8String]);
            }

            command_queue_ = [device_ newCommandQueue];
            if (command_queue_ == nil) {
                throw std::runtime_error("Failed to create Metal command queue.");
            }
        }
    }

    torch::Tensor forward(
        torch::Tensor means,
        torch::Tensor covariances,
        torch::Tensor colors,
        torch::Tensor opacities,
        int64_t height,
        int64_t width
    ) {
        auto means_cpu = means.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
        auto covariances_cpu = covariances.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
        auto colors_cpu = colors.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
        auto opacities_cpu = opacities.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();

        const auto num_gaussians = static_cast<uint32_t>(means_cpu.size(0));
        const auto num_channels = static_cast<uint32_t>(colors_cpu.size(1));
        TORCH_CHECK(num_channels >= 1 && num_channels <= 4, "Metal kernel supports 1 to 4 channels.");

        auto flat_covariances = torch::stack(
            {
                covariances_cpu.index({torch::indexing::Slice(), 0, 0}),
                covariances_cpu.index({torch::indexing::Slice(), 0, 1}),
                covariances_cpu.index({torch::indexing::Slice(), 1, 0}),
                covariances_cpu.index({torch::indexing::Slice(), 1, 1}),
            },
            1
        ).contiguous();

        auto output_cpu = torch::zeros(
            {height, width, static_cast<int64_t>(num_channels)},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)
        );

        @autoreleasepool {
            const NSUInteger means_bytes = means_cpu.numel() * sizeof(float);
            const NSUInteger cov_bytes = flat_covariances.numel() * sizeof(float);
            const NSUInteger colors_bytes = colors_cpu.numel() * sizeof(float);
            const NSUInteger opacities_bytes = opacities_cpu.numel() * sizeof(float);
            const NSUInteger output_bytes = output_cpu.numel() * sizeof(float);

            id<MTLBuffer> means_buffer = [device_ newBufferWithBytes:means_cpu.data_ptr<float>() length:means_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> cov_buffer = [device_ newBufferWithBytes:flat_covariances.data_ptr<float>() length:cov_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> colors_buffer = [device_ newBufferWithBytes:colors_cpu.data_ptr<float>() length:colors_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> opacities_buffer = [device_ newBufferWithBytes:opacities_cpu.data_ptr<float>() length:opacities_bytes options:MTLResourceStorageModeShared];

            RasterParams params {
                num_gaussians,
                num_channels,
                static_cast<uint32_t>(height),
                static_cast<uint32_t>(width),
            };
            id<MTLBuffer> params_buffer = [device_ newBufferWithBytes:&params length:sizeof(RasterParams) options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [device_ newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline_];
            [encoder setBuffer:means_buffer offset:0 atIndex:0];
            [encoder setBuffer:cov_buffer offset:0 atIndex:1];
            [encoder setBuffer:colors_buffer offset:0 atIndex:2];
            [encoder setBuffer:opacities_buffer offset:0 atIndex:3];
            [encoder setBuffer:params_buffer offset:0 atIndex:4];
            [encoder setBuffer:output_buffer offset:0 atIndex:5];

            MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1);
            const NSUInteger thread_width = std::min<NSUInteger>(pipeline_.threadExecutionWidth, 16);
            const NSUInteger thread_height = std::max<NSUInteger>(1, 16 / thread_width);
            MTLSize threadgroup_size = MTLSizeMake(thread_width, thread_height, 1);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
            [encoder endEncoding];

            [command_buffer commit];
            [command_buffer waitUntilCompleted];

            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                std::string message = "Metal command buffer failed.";
                if (command_buffer.error != nil) {
                    message += " ";
                    message += [[command_buffer.error localizedDescription] UTF8String];
                }
                throw std::runtime_error(message);
            }

            std::memcpy(output_cpu.data_ptr<float>(), [output_buffer contents], output_bytes);
        }

        auto output = output_cpu.to(means.device());
        return output.to(colors.scalar_type());
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLComputePipelineState> pipeline_;
};

MetalRasterizer2D& get_rasterizer() {
    static MetalRasterizer2D rasterizer;
    return rasterizer;
}

}  // namespace

torch::Tensor gaussian_splat_2d_forward_mps(
    torch::Tensor means,
    torch::Tensor covariances,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t height,
    int64_t width
) {
    TORCH_CHECK(means.device().is_mps(), "means must be on an MPS device");
    TORCH_CHECK(covariances.device().is_mps(), "covariances must be on an MPS device");
    TORCH_CHECK(colors.device().is_mps(), "colors must be on an MPS device");
    TORCH_CHECK(opacities.device().is_mps(), "opacities must be on an MPS device");
    TORCH_CHECK(means.scalar_type() == torch::kFloat32, "means must be float32");
    TORCH_CHECK(covariances.scalar_type() == torch::kFloat32, "covariances must be float32");
    TORCH_CHECK(colors.scalar_type() == torch::kFloat32, "colors must be float32");
    TORCH_CHECK(opacities.scalar_type() == torch::kFloat32, "opacities must be float32");

    return get_rasterizer().forward(means, covariances, colors, opacities, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_splat_2d_forward_mps", &gaussian_splat_2d_forward_mps, "TinySplat Metal forward kernel");
}
