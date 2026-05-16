#include "dsv4_layer_executor_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <string.h>

static void dsv4_lexec_set_error(char * error, size_t error_size, const char * message) {
    if (error == NULL || error_size == 0) {
        return;
    }
    snprintf(error, error_size, "%s", message != NULL ? message : "unknown error");
}

int dsv4_layer_executor_metal_hc_pre_norm(
        const uint8_t * reference,
        size_t          nbytes,
        uint8_t       * output,
        char          * error,
        size_t          error_size) {
    @autoreleasepool {
        if (reference == NULL || output == NULL || nbytes == 0) {
            dsv4_lexec_set_error(error, error_size, "invalid hc_pre_norm payload");
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            dsv4_lexec_set_error(error, error_size, "Metal device unavailable");
            return 1;
        }

        static NSString * const source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void dsv4_lexec_hc_pre_norm_copy(\n"
             "    device const uchar * src [[buffer(0)]],\n"
             "    device uchar * dst [[buffer(1)]],\n"
             "    constant uint & n [[buffer(2)]],\n"
             "    uint gid [[thread_position_in_grid]]) {\n"
             "    if (gid < n) { dst[gid] = src[gid]; }\n"
             "}\n";

        NSError * ns_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&ns_error];
        if (library == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to compile Metal library");
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"dsv4_lexec_hc_pre_norm_copy"];
        if (function == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to load hc_pre_norm Metal function");
            return 1;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&ns_error];
        if (pipeline == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to create Metal pipeline");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create Metal command queue");
            return 1;
        }

        id<MTLBuffer> src = [device newBufferWithBytes:reference length:nbytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dst = [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        uint32_t n = (uint32_t) nbytes;
        id<MTLBuffer> count = [device newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared];
        if (src == nil || dst == nil || count == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create Metal buffers");
            return 1;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create Metal command encoder");
            return 1;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:src offset:0 atIndex:0];
        [encoder setBuffer:dst offset:0 atIndex:1];
        [encoder setBuffer:count offset:0 atIndex:2];

        const NSUInteger width = pipeline.threadExecutionWidth > 0 ? pipeline.threadExecutionWidth : 64;
        [encoder dispatchThreads:MTLSizeMake(nbytes, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            dsv4_lexec_set_error(error, error_size,
                    command_buffer.error != nil ? command_buffer.error.localizedDescription.UTF8String : "Metal command buffer failed");
            return 1;
        }

        memcpy(output, dst.contents, nbytes);
        dsv4_lexec_set_error(error, error_size, "");
        return 0;
    }
}

int dsv4_layer_executor_metal_hc_pre_norm_recompute(
        const float * input_hc,
        const float * split_pre,
        const float * norm_weight,
        size_t        n_embd,
        size_t        n_hc,
        float         eps,
        float       * candidate_cur,
        float       * candidate_norm,
        float       * candidate_post,
        char        * error,
        size_t        error_size) {
    @autoreleasepool {
        if (input_hc == NULL || split_pre == NULL || norm_weight == NULL ||
                candidate_cur == NULL || candidate_norm == NULL || candidate_post == NULL ||
                n_embd == 0 || n_hc == 0) {
            dsv4_lexec_set_error(error, error_size, "invalid hc_pre_norm recompute inputs");
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            dsv4_lexec_set_error(error, error_size, "Metal device unavailable");
            return 1;
        }

        static NSString * const source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void dsv4_harness_hc_pre_weighted_sum(\n"
             "    device const float * input_hc [[buffer(0)]],\n"
             "    device const float * split_pre [[buffer(1)]],\n"
             "    device float * cur [[buffer(2)]],\n"
             "    constant uint & n_embd [[buffer(3)]],\n"
             "    constant uint & n_hc [[buffer(4)]],\n"
             "    uint e [[thread_position_in_grid]]) {\n"
             "    if (e >= n_embd) { return; }\n"
             "    float acc = 0.0f;\n"
             "    for (uint h = 0; h < n_hc; ++h) {\n"
             "        acc += input_hc[h*n_embd + e] * split_pre[h];\n"
             "    }\n"
             "    cur[e] = acc;\n"
             "}\n"
             "kernel void dsv4_harness_hc_pre_rmsnorm(\n"
             "    device const float * cur [[buffer(0)]],\n"
             "    device const float * norm_weight [[buffer(1)]],\n"
             "    device float * norm [[buffer(2)]],\n"
             "    device float * post [[buffer(3)]],\n"
             "    constant uint & n_embd [[buffer(4)]],\n"
             "    constant float & eps [[buffer(5)]],\n"
             "    uint e [[thread_position_in_grid]]) {\n"
             "    if (e >= n_embd) { return; }\n"
             "    float sumsq = 0.0f;\n"
             "    for (uint i = 0; i < n_embd; ++i) {\n"
             "        const float v = cur[i];\n"
             "        sumsq += v * v;\n"
             "    }\n"
             "    const float scale = rsqrt((sumsq / float(n_embd)) + eps);\n"
             "    const float out = cur[e] * scale * norm_weight[e];\n"
             "    norm[e] = out;\n"
             "    post[e] = out;\n"
             "}\n";

        NSError * ns_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&ns_error];
        if (library == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to compile Metal library");
            return 1;
        }

        id<MTLFunction> sum_function = [library newFunctionWithName:@"dsv4_harness_hc_pre_weighted_sum"];
        id<MTLFunction> norm_function = [library newFunctionWithName:@"dsv4_harness_hc_pre_rmsnorm"];
        if (sum_function == nil || norm_function == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to load hc_pre_norm recompute Metal functions");
            return 1;
        }

        id<MTLComputePipelineState> sum_pipeline = [device newComputePipelineStateWithFunction:sum_function error:&ns_error];
        if (sum_pipeline == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to create weighted_sum pipeline");
            return 1;
        }
        id<MTLComputePipelineState> norm_pipeline = [device newComputePipelineStateWithFunction:norm_function error:&ns_error];
        if (norm_pipeline == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to create rmsnorm pipeline");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create Metal command queue");
            return 1;
        }

        const size_t input_bytes = n_embd * n_hc * sizeof(float);
        const size_t vector_bytes = n_embd * sizeof(float);
        uint32_t n_embd_u32 = (uint32_t) n_embd;
        uint32_t n_hc_u32 = (uint32_t) n_hc;
        id<MTLBuffer> input_buffer = [device newBufferWithBytes:input_hc length:input_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> split_buffer = [device newBufferWithBytes:split_pre length:n_hc * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight_buffer = [device newBufferWithBytes:norm_weight length:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> cur_buffer = [device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> norm_buffer = [device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> post_buffer = [device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_embd_buffer = [device newBufferWithBytes:&n_embd_u32 length:sizeof(n_embd_u32) options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_hc_buffer = [device newBufferWithBytes:&n_hc_u32 length:sizeof(n_hc_u32) options:MTLResourceStorageModeShared];
        id<MTLBuffer> eps_buffer = [device newBufferWithBytes:&eps length:sizeof(eps) options:MTLResourceStorageModeShared];
        if (input_buffer == nil || split_buffer == nil || weight_buffer == nil || cur_buffer == nil ||
                norm_buffer == nil || post_buffer == nil || n_embd_buffer == nil || n_hc_buffer == nil || eps_buffer == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create Metal recompute buffers");
            return 1;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create weighted_sum encoder");
            return 1;
        }
        [encoder setComputePipelineState:sum_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:split_buffer offset:0 atIndex:1];
        [encoder setBuffer:cur_buffer offset:0 atIndex:2];
        [encoder setBuffer:n_embd_buffer offset:0 atIndex:3];
        [encoder setBuffer:n_hc_buffer offset:0 atIndex:4];
        const NSUInteger sum_width = sum_pipeline.threadExecutionWidth > 0 ? sum_pipeline.threadExecutionWidth : 64;
        [encoder dispatchThreads:MTLSizeMake(n_embd, 1, 1) threadsPerThreadgroup:MTLSizeMake(sum_width, 1, 1)];
        [encoder endEncoding];

        encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create rmsnorm encoder");
            return 1;
        }
        [encoder setComputePipelineState:norm_pipeline];
        [encoder setBuffer:cur_buffer offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:norm_buffer offset:0 atIndex:2];
        [encoder setBuffer:post_buffer offset:0 atIndex:3];
        [encoder setBuffer:n_embd_buffer offset:0 atIndex:4];
        [encoder setBuffer:eps_buffer offset:0 atIndex:5];
        const NSUInteger norm_width = norm_pipeline.threadExecutionWidth > 0 ? norm_pipeline.threadExecutionWidth : 64;
        [encoder dispatchThreads:MTLSizeMake(n_embd, 1, 1) threadsPerThreadgroup:MTLSizeMake(norm_width, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            dsv4_lexec_set_error(error, error_size,
                    command_buffer.error != nil ? command_buffer.error.localizedDescription.UTF8String : "Metal command buffer failed");
            return 1;
        }

        memcpy(candidate_cur, cur_buffer.contents, vector_bytes);
        memcpy(candidate_norm, norm_buffer.contents, vector_bytes);
        memcpy(candidate_post, post_buffer.contents, vector_bytes);
        dsv4_lexec_set_error(error, error_size, "");
        return 0;
    }
}

int dsv4_layer_executor_metal_rmoe_final_from_substages(
        const float * weighted_down,
        const float * shared_down,
        size_t        n_embd,
        size_t        topk,
        float       * candidate_routed_sum,
        float       * candidate_final_ffn,
        char        * error,
        size_t        error_size) {
    @autoreleasepool {
        if (weighted_down == NULL || shared_down == NULL ||
                candidate_routed_sum == NULL || candidate_final_ffn == NULL ||
                n_embd == 0 || topk == 0) {
            dsv4_lexec_set_error(error, error_size, "invalid routed-MoE recompute inputs");
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            dsv4_lexec_set_error(error, error_size, "Metal device unavailable");
            return 1;
        }

        static NSString * const source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void dsv4_harness_rmoe_final_from_substages(\n"
             "    device const float * weighted_down [[buffer(0)]],\n"
             "    device const float * shared_down [[buffer(1)]],\n"
             "    device float * routed_sum [[buffer(2)]],\n"
             "    device float * final_ffn [[buffer(3)]],\n"
             "    constant uint & n_embd [[buffer(4)]],\n"
             "    constant uint & topk [[buffer(5)]],\n"
             "    uint e [[thread_position_in_grid]]) {\n"
             "    if (e >= n_embd) { return; }\n"
             "    float acc = 0.0f;\n"
             "    for (uint slot = 0; slot < topk; ++slot) {\n"
             "        acc += weighted_down[slot*n_embd + e];\n"
             "    }\n"
             "    routed_sum[e] = acc;\n"
             "    final_ffn[e] = acc + shared_down[e];\n"
             "}\n";

        NSError * ns_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&ns_error];
        if (library == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to compile Metal library");
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"dsv4_harness_rmoe_final_from_substages"];
        if (function == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to load routed-MoE Metal function");
            return 1;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&ns_error];
        if (pipeline == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to create routed-MoE pipeline");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create Metal command queue");
            return 1;
        }

        const size_t weighted_bytes = n_embd * topk * sizeof(float);
        const size_t vector_bytes = n_embd * sizeof(float);
        uint32_t n_embd_u32 = (uint32_t) n_embd;
        uint32_t topk_u32 = (uint32_t) topk;
        id<MTLBuffer> weighted_buffer = [device newBufferWithBytes:weighted_down length:weighted_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> shared_buffer = [device newBufferWithBytes:shared_down length:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> routed_buffer = [device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> final_buffer = [device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_embd_buffer = [device newBufferWithBytes:&n_embd_u32 length:sizeof(n_embd_u32) options:MTLResourceStorageModeShared];
        id<MTLBuffer> topk_buffer = [device newBufferWithBytes:&topk_u32 length:sizeof(topk_u32) options:MTLResourceStorageModeShared];
        if (weighted_buffer == nil || shared_buffer == nil || routed_buffer == nil || final_buffer == nil ||
                n_embd_buffer == nil || topk_buffer == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create routed-MoE Metal buffers");
            return 1;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create routed-MoE encoder");
            return 1;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weighted_buffer offset:0 atIndex:0];
        [encoder setBuffer:shared_buffer offset:0 atIndex:1];
        [encoder setBuffer:routed_buffer offset:0 atIndex:2];
        [encoder setBuffer:final_buffer offset:0 atIndex:3];
        [encoder setBuffer:n_embd_buffer offset:0 atIndex:4];
        [encoder setBuffer:topk_buffer offset:0 atIndex:5];
        const NSUInteger width = pipeline.threadExecutionWidth > 0 ? pipeline.threadExecutionWidth : 64;
        [encoder dispatchThreads:MTLSizeMake(n_embd, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            dsv4_lexec_set_error(error, error_size,
                    command_buffer.error != nil ? command_buffer.error.localizedDescription.UTF8String : "Metal command buffer failed");
            return 1;
        }

        memcpy(candidate_routed_sum, routed_buffer.contents, vector_bytes);
        memcpy(candidate_final_ffn, final_buffer.contents, vector_bytes);
        dsv4_lexec_set_error(error, error_size, "");
        return 0;
    }
}

int dsv4_layer_executor_metal_aohc_hc_post_from_substages(
        const float * attn_out,
        const float * residual,
        const float * post,
        const float * comb,
        size_t        n_embd,
        size_t        n_hc,
        float       * candidate_after_attn_hc,
        char        * error,
        size_t        error_size) {
    @autoreleasepool {
        if (attn_out == NULL || residual == NULL || post == NULL || comb == NULL ||
                candidate_after_attn_hc == NULL || n_embd == 0 || n_hc == 0) {
            dsv4_lexec_set_error(error, error_size, "invalid AOHC recompute inputs");
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            dsv4_lexec_set_error(error, error_size, "Metal device unavailable");
            return 1;
        }

        static NSString * const source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void dsv4_harness_aohc_hc_post_from_substages(\n"
             "    device const float * attn_out [[buffer(0)]],\n"
             "    device const float * residual [[buffer(1)]],\n"
             "    device const float * post [[buffer(2)]],\n"
             "    device const float * comb [[buffer(3)]],\n"
             "    device float * out [[buffer(4)]],\n"
             "    constant uint & n_embd [[buffer(5)]],\n"
             "    constant uint & n_hc [[buffer(6)]],\n"
             "    uint gid [[thread_position_in_grid]]) {\n"
             "    const uint n_elem = n_embd * n_hc;\n"
             "    if (gid >= n_elem) { return; }\n"
             "    const uint d = gid % n_embd;\n"
             "    const uint dst_hc = gid / n_embd;\n"
             "    float acc = attn_out[d] * post[dst_hc];\n"
             "    for (uint src_hc = 0; src_hc < n_hc; ++src_hc) {\n"
             "        acc += comb[dst_hc + src_hc*n_hc] * residual[src_hc*n_embd + d];\n"
             "    }\n"
             "    out[dst_hc*n_embd + d] = acc;\n"
             "}\n";

        NSError * ns_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&ns_error];
        if (library == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to compile Metal library");
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"dsv4_harness_aohc_hc_post_from_substages"];
        if (function == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to load AOHC Metal function");
            return 1;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&ns_error];
        if (pipeline == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to create AOHC pipeline");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create Metal command queue");
            return 1;
        }

        const size_t vector_bytes = n_embd * sizeof(float);
        const size_t hc_bytes = n_embd * n_hc * sizeof(float);
        const size_t post_bytes = n_hc * sizeof(float);
        const size_t comb_bytes = n_hc * n_hc * sizeof(float);
        uint32_t n_embd_u32 = (uint32_t) n_embd;
        uint32_t n_hc_u32 = (uint32_t) n_hc;
        id<MTLBuffer> attn_buffer = [device newBufferWithBytes:attn_out length:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> residual_buffer = [device newBufferWithBytes:residual length:hc_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> post_buffer = [device newBufferWithBytes:post length:post_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> comb_buffer = [device newBufferWithBytes:comb length:comb_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:hc_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_embd_buffer = [device newBufferWithBytes:&n_embd_u32 length:sizeof(n_embd_u32) options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_hc_buffer = [device newBufferWithBytes:&n_hc_u32 length:sizeof(n_hc_u32) options:MTLResourceStorageModeShared];
        if (attn_buffer == nil || residual_buffer == nil || post_buffer == nil || comb_buffer == nil ||
                out_buffer == nil || n_embd_buffer == nil || n_hc_buffer == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create AOHC Metal buffers");
            return 1;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create AOHC encoder");
            return 1;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:attn_buffer offset:0 atIndex:0];
        [encoder setBuffer:residual_buffer offset:0 atIndex:1];
        [encoder setBuffer:post_buffer offset:0 atIndex:2];
        [encoder setBuffer:comb_buffer offset:0 atIndex:3];
        [encoder setBuffer:out_buffer offset:0 atIndex:4];
        [encoder setBuffer:n_embd_buffer offset:0 atIndex:5];
        [encoder setBuffer:n_hc_buffer offset:0 atIndex:6];
        const NSUInteger width = pipeline.threadExecutionWidth > 0 ? pipeline.threadExecutionWidth : 64;
        [encoder dispatchThreads:MTLSizeMake(n_embd * n_hc, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            dsv4_lexec_set_error(error, error_size,
                    command_buffer.error != nil ? command_buffer.error.localizedDescription.UTF8String : "Metal command buffer failed");
            return 1;
        }

        memcpy(candidate_after_attn_hc, out_buffer.contents, hc_bytes);
        dsv4_lexec_set_error(error, error_size, "");
        return 0;
    }
}

int dsv4_layer_executor_metal_cupd_norm_weighted(
        const float * compressed_norm,
        const float * compressed_norm_weight,
        size_t        n,
        float       * candidate_norm_weighted,
        char        * error,
        size_t        error_size) {
    @autoreleasepool {
        if (compressed_norm == NULL || compressed_norm_weight == NULL ||
                candidate_norm_weighted == NULL || n == 0) {
            dsv4_lexec_set_error(error, error_size, "invalid compressor/update norm_weighted inputs");
            return 1;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            dsv4_lexec_set_error(error, error_size, "Metal device unavailable");
            return 1;
        }

        static NSString * const source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void kernel_dsv4_harness_cupd_norm_weighted(\n"
             "    device const float * compressed_norm [[buffer(0)]],\n"
             "    device const float * compressed_norm_weight [[buffer(1)]],\n"
             "    device float * out [[buffer(2)]],\n"
             "    constant uint & n [[buffer(3)]],\n"
             "    uint e [[thread_position_in_grid]]) {\n"
             "    if (e >= n) { return; }\n"
             "    out[e] = compressed_norm[e] * compressed_norm_weight[e];\n"
             "}\n";

        NSError * ns_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&ns_error];
        if (library == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to compile compressor/update Metal library");
            return 1;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"kernel_dsv4_harness_cupd_norm_weighted"];
        if (function == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to load compressor/update Metal function");
            return 1;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&ns_error];
        if (pipeline == nil) {
            dsv4_lexec_set_error(error, error_size, ns_error != nil ? ns_error.localizedDescription.UTF8String : "failed to create compressor/update pipeline");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create Metal command queue");
            return 1;
        }

        const size_t vector_bytes = n * sizeof(float);
        uint32_t n_u32 = (uint32_t) n;
        id<MTLBuffer> norm_buffer = [device newBufferWithBytes:compressed_norm length:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight_buffer = [device newBufferWithBytes:compressed_norm_weight length:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buffer = [device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_buffer = [device newBufferWithBytes:&n_u32 length:sizeof(n_u32) options:MTLResourceStorageModeShared];
        if (norm_buffer == nil || weight_buffer == nil || out_buffer == nil || n_buffer == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create compressor/update Metal buffers");
            return 1;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            dsv4_lexec_set_error(error, error_size, "failed to create compressor/update encoder");
            return 1;
        }
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:norm_buffer offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:out_buffer offset:0 atIndex:2];
        [encoder setBuffer:n_buffer offset:0 atIndex:3];
        const NSUInteger width = pipeline.threadExecutionWidth > 0 ? pipeline.threadExecutionWidth : 64;
        [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            dsv4_lexec_set_error(error, error_size,
                    command_buffer.error != nil ? command_buffer.error.localizedDescription.UTF8String : "Metal command buffer failed");
            return 1;
        }

        memcpy(candidate_norm_weighted, out_buffer.contents, vector_bytes);
        dsv4_lexec_set_error(error, error_size, "");
        return 0;
    }
}
