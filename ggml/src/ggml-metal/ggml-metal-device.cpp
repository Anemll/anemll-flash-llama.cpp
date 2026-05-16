#include "ggml-metal-device.h"

#include "ggml-metal-impl.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>

struct ggml_metal_device_deleter {
    void operator()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

static int16_t ggml_metal_mul_mm_env_nk() {
    static int16_t nk = -1;
    if (nk != -1) {
        return nk;
    }

    nk = 32;
    const char * value = std::getenv("GGML_METAL_MUL_MM_NK");
    if (value == nullptr || value[0] == '\0') {
        return nk;
    }

    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return nk;
    }

    if (parsed >= 32 && parsed <= 128 && parsed % 16 == 0) {
        nk = int16_t(parsed);
    }

    return nk;
}

static int16_t ggml_metal_mul_mm_env_walk_mode() {
    static int16_t walk = -1;
    if (walk != -1) {
        return walk;
    }

    walk = GGML_METAL_MUL_MM_WALK_LEGACY;
    const char * value = std::getenv("GGML_METAL_MUL_MM_WALK");
    if (value == nullptr || value[0] == '\0') {
        return walk;
    }

    if (strcmp(value, "legacy") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_LEGACY;
    } else if (strcmp(value, "regular") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_REGULAR;
    } else if (strcmp(value, "morton") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_MORTON;
    }

    return walk;
}

static bool ggml_metal_in_memory_fp16_activations_enabled() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_IN_MEMORY_FP16_ACTIVATIONS");
    return value != nullptr &&
            value[0] != '\0' &&
            std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0 &&
            std::strcmp(value, "off") != 0 &&
            std::strcmp(value, "no") != 0;
}

ggml_metal_device_t ggml_metal_device_get(int device) {
    static std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device));

    return devs.back().get();
}

struct ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> data;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(void) {
    ggml_metal_pipelines_t res = new ggml_metal_pipelines();

    return res;
}

void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    if (!ppls) {
        return;
    }

    for (auto it = ppls->data.begin(); it != ppls->data.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    delete ppls;
}

void ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline) {
    ppls->data[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name) {
    if (ppls->data.find(name) == ppls->data.end()) {
        return nullptr;
    }

    return ppls->data[name];
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op) {
        case GGML_OP_ADD_ID: op_str = "add_id"; break;
        case GGML_OP_CONCAT: op_str = "concat"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, sizeof(base), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, sizeof(name), "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows(ggml_metal_library_t lib, ggml_type tidx, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_set_rows_%s_%s", ggml_type_name(tdst), ggml_type_name(tidx));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int n = op->src[0]->ne[0];

    snprintf(base, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", base, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nsg  = 1;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SCALE:      op_num = OP_UNARY_NUM_SCALE;      break;
        case GGML_OP_FILL:       op_num = OP_UNARY_NUM_FILL;       break;
        case GGML_OP_CLAMP:      op_num = OP_UNARY_NUM_CLAMP;      break;
        case GGML_OP_SQR:        op_num = OP_UNARY_NUM_SQR;        break;
        case GGML_OP_SQRT:       op_num = OP_UNARY_NUM_SQRT;       break;
        case GGML_OP_SIN:        op_num = OP_UNARY_NUM_SIN;        break;
        case GGML_OP_COS:        op_num = OP_UNARY_NUM_COS;        break;
        case GGML_OP_LOG:        op_num = OP_UNARY_NUM_LOG;        break;
        case GGML_OP_LEAKY_RELU: op_num = OP_UNARY_NUM_LEAKY_RELU; break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:        op_num = OP_UNARY_NUM_TANH;        break;
                case GGML_UNARY_OP_RELU:        op_num = OP_UNARY_NUM_RELU;        break;
                case GGML_UNARY_OP_SIGMOID:     op_num = OP_UNARY_NUM_SIGMOID;     break;
                case GGML_UNARY_OP_GELU:        op_num = OP_UNARY_NUM_GELU;        break;
                case GGML_UNARY_OP_GELU_ERF:    op_num = OP_UNARY_NUM_GELU_ERF;    break;
                case GGML_UNARY_OP_GELU_QUICK:  op_num = OP_UNARY_NUM_GELU_QUICK;  break;
                case GGML_UNARY_OP_SILU:        op_num = OP_UNARY_NUM_SILU;        break;
                case GGML_UNARY_OP_ELU:         op_num = OP_UNARY_NUM_ELU;         break;
                case GGML_UNARY_OP_NEG:         op_num = OP_UNARY_NUM_NEG;         break;
                case GGML_UNARY_OP_ABS:         op_num = OP_UNARY_NUM_ABS;         break;
                case GGML_UNARY_OP_SGN:         op_num = OP_UNARY_NUM_SGN;         break;
                case GGML_UNARY_OP_STEP:        op_num = OP_UNARY_NUM_STEP;        break;
                case GGML_UNARY_OP_HARDSWISH:   op_num = OP_UNARY_NUM_HARDSWISH;   break;
                case GGML_UNARY_OP_HARDSIGMOID: op_num = OP_UNARY_NUM_HARDSIGMOID; break;
                case GGML_UNARY_OP_EXP:         op_num = OP_UNARY_NUM_EXP;         break;
                case GGML_UNARY_OP_SOFTPLUS:    op_num = OP_UNARY_NUM_SOFTPLUS;    break;
                case GGML_UNARY_OP_EXPM1:       op_num = OP_UNARY_NUM_EXPM1;       break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;
    const bool is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(base, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", base, op_num, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_UNARY + 0);
        ggml_metal_cv_set_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:        op_str = "reglu";        break;
                case GGML_GLU_OP_GEGLU:        op_str = "geglu";        break;
                case GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       break;
                case GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   break;
                case GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    break;
                case GGML_GLU_OP_GEGLU_QUICK:  op_str = "geglu_quick";  break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu_scaled(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_GLU);
    GGML_ASSERT(ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU);

    const char * base = "kernel_swiglu_scaled_f32";
    const char * name = "kernel_swiglu_scaled_f32";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SUM_ROWS: op_num = OP_SUM_ROWS_NUM_SUM_ROWS; break;
        case GGML_OP_MEAN:     op_num = OP_SUM_ROWS_NUM_MEAN;     break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(base, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", base, op_num);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_SUM_ROWS + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*sizeof(float);

    if (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    const char * op_str = "tri";
    const int ttype = op->op_params[0];

    snprintf(base, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    const ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(base, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(ggml_metal_library_t lib, const ggml_tensor * op, int ssm_conv_bs) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d", base, ssm_conv_bs);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan(ggml_metal_library_t lib, const ggml_tensor * op)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    char base[256];
    char name[256];

    const int nsg = (ne00 + 31)/32;

    snprintf(base, 256, "kernel_ssm_scan_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = (32 + 2)*sizeof(float)*nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    switch (op->op) {
        case GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    const int ne20 = op->src[2]->ne[0]; // S_v
    const int ne21 = op->src[2]->ne[1]; // H
    const int ne30 = op->src[3]->ne[0]; // G

    const int nsg = op->src[2]->ne[0]/32;

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);
    GGML_ASSERT(ne20 % 32 == 0);

    snprintf(base, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    snprintf(name, 256, "%s_ne20=%d_ne30=%d", base, ne20, ne30);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_set_int16(cv, ne30, FC_GATED_DELTA_NET + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int nsg = 8;
    const int n   = op->src[1]->ne[1];
    const int k   = op->src[1]->ne[0];

    snprintf(base, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d", base, nsg, n, k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_set_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_set_int16(cv, k,   FC_SOLVE_TRI + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, 32)*nsg*sizeof(float), 16);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, ggml_type tsrc0, ggml_type tsrc1, int nsg, int nxpsg, int r1ptg) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d", base, nsg, nxpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,   FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, nxpsg, FC_MUL_MV + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_types(
        ggml_metal_library_t lib,
        ggml_type tsrc0,
        ggml_type tsrc1,
        int64_t src0_ne0,
        int64_t out_ne0,
        int64_t out_ne1,
        bool f16_output,
        bool use_m5_expert,
        bool use_m5_sgmatrix) {
    char base[256];
    char name[256];

    const bool bc_inp = src0_ne0 % 32 != 0;
    const bool bc_out = out_ne0 % 64 != 0 || out_ne1 % 32 != 0;
    const bool allow_f16_src1 = ggml_metal_in_memory_fp16_activations_enabled();
    const bool use_flashmoe_metal4_kernel =
            use_m5_expert &&
            ((f16_output &&
              (tsrc1 == GGML_TYPE_F32 || (allow_f16_src1 && tsrc1 == GGML_TYPE_F16)) &&
              (tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K || tsrc0 == GGML_TYPE_Q6_K)) ||
             (tsrc0 == GGML_TYPE_F16 && tsrc1 == GGML_TYPE_F16));
    // Keep the first custom Metal4 expert kernel numerically aligned with the proven
    // baseline shape; we can retune it later without touching the generic mul_mm path.
    const char * kernel_prefix = nullptr;
    if (f16_output) {
        kernel_prefix = use_flashmoe_metal4_kernel ?
                "kernel_mul_mm_flashmoe_metal4_out_f16" :
                "kernel_mul_mm_out_f16";
    } else {
        kernel_prefix = use_flashmoe_metal4_kernel ?
                "kernel_mul_mm_flashmoe_metal4" :
                "kernel_mul_mm";
    }
    const int16_t env_walk = ggml_metal_mul_mm_env_walk_mode();
    const int16_t env_nk = ggml_metal_mul_mm_env_nk();
    const bool supports_tuned_nk =
            !f16_output &&
            !use_flashmoe_metal4_kernel &&
            tsrc1 == GGML_TYPE_F32 &&
            (tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K || tsrc0 == GGML_TYPE_Q6_K);
    const int16_t nk = supports_tuned_nk ? env_nk : int16_t(32);
    const size_t tile_smem = size_t(96) * nk * size_t(2);

    if (nk == 32) {
        snprintf(base, sizeof(base), "%s_%s_%s", kernel_prefix, ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    } else {
        snprintf(base, sizeof(base), "%s_nk%d_%s_%s", kernel_prefix, int(nk), ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    }
    snprintf(name, sizeof(name), "%s_bci=%d_bco=%d_walk=%d_m5sg=%d", base, bc_inp, bc_out, int(env_walk), use_m5_sgmatrix);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, bc_out, FC_MUL_MM + 1);
        ggml_metal_cv_set_int16(cv, env_walk, FC_MUL_MM + 2);
        ggml_metal_cv_set_bool(cv, use_m5_sgmatrix, FC_MUL_MM_M5_SGMATRIX);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    // when the output size is not multiple of 64x32, we need extra smem to prevent out-of-bounds writes
    res.smem = std::max(tile_smem, (bc_out || f16_output) ? size_t(8192) : size_t(0));

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, const ggml_tensor * op, bool use_m5_expert, bool use_m5_sgmatrix) {
    return ggml_metal_library_get_pipeline_mul_mm_types(
            lib,
            op->src[0]->type,
            op->src[1]->type,
            op->src[0]->ne[0],
            op->ne[0],
            op->ne[1],
            op->op == GGML_OP_MUL_MAT_F16,
            use_m5_expert,
            use_m5_sgmatrix);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flashmoe_split_glu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_FLASHMOE_SPLIT_GLU);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);
    GGML_ASSERT(op->src[2] != nullptr);
    GGML_ASSERT(op->src[3] != nullptr);

    char base[256];
    char name[256];

    const int32_t glu_op = ggml_get_op_params_i32(op, 0);
    const char * glu_str = nullptr;

    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:
            glu_str = "swiglu";
            break;
        case GGML_GLU_OP_GEGLU:
            glu_str = "geglu";
            break;
        default:
            GGML_ABORT("unsupported FLASHMOE_SPLIT_GLU op");
    }

    snprintf(base, sizeof(base), "kernel_flashmoe_split_%s_%s_%s_%s",
            glu_str,
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[2]->type),
            ggml_type_name(op->src[3]->type));
    snprintf(name, sizeof(name), "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // The fused split-MLP kernel reuses the same threadgroup storage for:
    // 1. gate/up staged weights + input activations + gate/up accumulators
    // 2. down staged weights + fused GLU activations + output partials
    constexpr size_t flashmoe_split_ff_tile  = 32;
    constexpr size_t flashmoe_split_tok_tile = 32;
    constexpr size_t flashmoe_split_out_tile = 64;
    constexpr size_t flashmoe_split_nk       = 32;

    const size_t stage1_smem =
            2 * flashmoe_split_ff_tile * flashmoe_split_nk * sizeof(uint16_t) +
            flashmoe_split_tok_tile * flashmoe_split_nk * sizeof(uint16_t) +
            2 * flashmoe_split_ff_tile * flashmoe_split_tok_tile * sizeof(float);

    const size_t stage2_smem =
            flashmoe_split_out_tile * flashmoe_split_nk * sizeof(uint16_t) +
            flashmoe_split_tok_tile * flashmoe_split_nk * sizeof(uint16_t) +
            flashmoe_split_out_tile * flashmoe_split_tok_tile * sizeof(float);

    res.nsg = 1;
    res.smem = std::max(stage1_smem, stage2_smem);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";

    // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                if (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } else {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = 32*sizeof(float)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_F8_E4M3_B128:
            {
                nsg = N_SG_F8_E4M3_B128;
                nr0 = N_R0_F8_E4M3_B128;
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, int ne02, int ne20) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", base, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (size_t) ne02*ne20*sizeof(uint16_t);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, const ggml_tensor * op, bool use_m5_sgmatrix) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;

    snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_m5sg=%d", base, bc_inp, use_m5_sgmatrix);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, use_m5_sgmatrix, FC_MUL_MM_M5_SGMATRIX);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";

        // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*sizeof(float)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_F8_E4M3_B128:
            {
                nsg = N_SG_F8_E4M3_B128;
                nr0 = N_R0_F8_E4M3_B128;
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_attn_out_low(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    const int nsg = N_SG_Q8_0;
    const int nr0 = N_R0_Q8_0;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_attn_out_low_q8_0_f32");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = 1;
    res.nsg  = nsg;
    res.smem = 32*sizeof(float)*N_R0_Q8_0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_compressor_pair(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(op->src[0] != nullptr);
    const ggml_tensor * src1 =
            (op->op == GGML_OP_DSV4_COMPRESSOR_PAIR_PROJ ||
             op->op == GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE) ? op->src[2] : op->src[1];
    GGML_ASSERT(src1 != nullptr);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int nsg = std::min(4, ((int) op->src[0]->ne[0] + 127) / 128);
    const int nr0 = (op->src[0]->ne[1] == 512 || op->src[0]->ne[1] == 1024) && op->src[0]->ne[0] >= 4096 ? 4 : 2;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_mul_mv_pair_f16_f32_4");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = 1;
    res.nsg  = nsg;
    res.smem = 32*sizeof(float)*nr0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_pair(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);

    char base[256];
    char name[256];

    int nsg = 0;
    int nr0 = 0;
    int nr1 = 1;
    size_t smem = 0;

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    if (tsrc1 != GGML_TYPE_F32) {
        return {};
    }

    switch (tsrc0) {
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8 + 128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4 + 128;
            } break;
        default:
            {
                return {};
            }
    }

    snprintf(base, 256, "kernel_mul_mv_pair_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = nr1;
    res.nsg = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_pair_swiglu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);

    int nsg = 0;
    int nr0 = 0;
    size_t smem = 0;

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    if (tsrc0 != GGML_TYPE_IQ2_XXS || tsrc1 != GGML_TYPE_F32) {
        return {};
    }

    nsg = N_SG_IQ2_XXS;
    nr0 = N_R0_IQ2_XXS;
    smem = 256*8 + 128;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_mul_mv_pair_limited_swiglu_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_pair_swiglu_weighted(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    if (tsrc0 != GGML_TYPE_IQ2_XXS || tsrc1 != GGML_TYPE_F32) {
        return {};
    }

    const int nsg = N_SG_IQ2_XXS;
    const int nr0 = N_R0_IQ2_XXS;
    const size_t smem = 256*8 + 128;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_mul_mv_pair_limited_swiglu_weighted_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_ffn_moe_stage_v2_gate(ggml_metal_library_t lib) {
    const int nsg = N_SG_IQ2_XXS;
    const int nr0 = N_R0_IQ2_XXS;
    const size_t smem = 256*8 + 128;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_ffn_moe_stage_gate_up_iq2_xxs_f32");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_routed_moe_swiglu_slots(ggml_metal_library_t lib) {
    const int nr0 = 1;
    const int nsg = 1;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_routed_moe_swiglu_slots_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_ffn_moe_stage_v2_weight(ggml_metal_library_t lib) {
    const int nr0 = 1;
    const int nsg = 1;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_ffn_moe_stage_weight_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_ffn_moe_stage_v2_down(ggml_metal_library_t lib) {
    const int nsg = N_SG_Q2_K;
    const int nr0 = N_R0_Q2_K;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_ffn_moe_decode_stage_q2_K_f32");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_routed_moe_down_slots(ggml_metal_library_t lib) {
    const int nsg = N_SG_Q2_K;
    const int nr0 = N_R0_Q2_K;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_routed_moe_down_slots_q2_K_f32");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_routed_moe_shared_gate_up_swiglu(ggml_metal_library_t lib) {
    const int nsg = N_SG_Q8_0;
    const int nr0 = N_R0_Q8_0;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_routed_moe_shared_gate_up_swiglu_q8_0_f32");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = (size_t) 2 * 32 * nr0 * sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_routed_moe_shared_down_final(ggml_metal_library_t lib) {
    const int nsg = N_SG_Q8_0;
    const int nr0 = N_R0_Q8_0;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_routed_moe_shared_down_final_q8_0_f32");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = (size_t) 32 * nr0 * sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_shared_swiglu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    if (tsrc0 != GGML_TYPE_Q8_0 || tsrc1 != GGML_TYPE_F32) {
        return {};
    }

    const int nsg = N_SG_Q8_0;
    const int nr0 = N_R0_Q8_0;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_mul_mv_shared_swiglu_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = 2*32*sizeof(float)*N_R0_Q8_0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id_sum6(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);

    int nsg = 0;
    int nr0 = 0;
    size_t smem = 0;

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    if (tsrc0 != GGML_TYPE_Q2_K || tsrc1 != GGML_TYPE_F32) {
        return {};
    }

    nsg = N_SG_Q2_K;
    nr0 = N_R0_Q2_K;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_mul_mv_id_sum6_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id_sum6_weighted(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op != nullptr);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    if (tsrc0 != GGML_TYPE_Q2_K || tsrc1 != GGML_TYPE_F32) {
        return {};
    }

    const int nsg = N_SG_Q2_K;
    const int nr0 = N_R0_Q2_K;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_mul_mv_id_sum6_weighted_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*(sizeof(float) + sizeof(int32_t));

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

// note: reuse the argsort kernel for top_k
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    // note: the top_k kernel is always descending order
    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            base,
            has_mask,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            base,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        bool    use_metal4_qk,
        int32_t nsg,
        int32_t walk_mode,
        int32_t chunk_mode,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    // do bounds checks for the mask?
    const bool bc_mask = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_m4qk=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d_walk=%d_chunk=%d_nwg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            use_metal4_qk,
            bc_mask,
            ns10,
            ns20,
            nsg,
            walk_mode,
            chunk_mode,
            nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_set_bool(cv, bc_mask, FC_FLASH_ATTN_EXT + 10);
        ggml_metal_cv_set_bool(cv, use_metal4_qk, FC_FLASH_ATTN_EXT + 11);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);
        ggml_metal_cv_set_int32(cv, walk_mode, FC_FLASH_ATTN_EXT + 23);
        ggml_metal_cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT + 24);
        ggml_metal_cv_set_int32(cv, chunk_mode, FC_FLASH_ATTN_EXT + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext_vec",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT_VEC + 4);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_metal4(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext_vec_metal4",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT_VEC + 4);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", base, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;

    GGML_UNUSED(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, const ggml_tensor * op, int32_t n_fuse) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t1_str = ggml_type_name(op->src[1]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    const bool is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    const bool is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", base, op_num, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_set_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", base, op_num, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, false,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_L2_NORM);

    char base[256];
    char name[256];

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    snprintf(base, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.c4   = is_c4;
    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_GROUP_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, const ggml_tensor * op, int n_fuse) {
    assert(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    switch (op->op) {
        case GGML_OP_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        case GGML_OP_RMS_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_rms_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_rms_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROPE);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[2];

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_neox) {
        snprintf(base, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } else if ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } else if (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d", base, is_imrope ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, is_imrope, FC_ROPE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc_split_sinkhorn(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_HC_SPLIT_SINKHORN);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_hc_split_sinkhorn");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc_split_sinkhorn_weighted_sum(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_HC_SPLIT_SINKHORN);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_hc_split_sinkhorn_weighted_sum");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc_split_sinkhorn_weighted_sum_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_HC_SPLIT_SINKHORN);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_hc_split_sinkhorn_weighted_sum_norm");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (4096 + 16 + 32)*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc_weighted_sum(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_HC_WEIGHTED_SUM);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_hc_weighted_sum");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc_expand(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_HC_EXPAND);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_hc_expand");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc_expand4(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_HC_EXPAND);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_hc_expand4");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_q8_hc_expand(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT);
    assert(op->src[0]->type == GGML_TYPE_Q8_0);
    assert(op->src[1]->type == GGML_TYPE_F32);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    const int nsg = N_SG_Q8_0;
    const int nr0 = N_R0_Q8_0;
    const size_t smem = 32*sizeof(float)*N_R0_Q8_0;

    snprintf(base, 256, "kernel_dsv4_q8_hc_expand_q8_0");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = 1;
    res.nsg  = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_fp8_kv_quantize(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_FP8_KV_QUANTIZE);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_fp8_kv_quantize");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hadamard_fp4_quantize(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_HADAMARD_FP4_QUANTIZE);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_hadamard_fp4_quantize");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_indexer_weighted_score(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_INDEXER_WEIGHTED_SCORE);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_indexer_weighted_score");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_mixed_attn(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_MIXED_ATTN);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_mixed_attn_f16");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_decode_compress(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_DECODE_COMPRESS);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[1];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    snprintf(base, 256, "kernel_dsv4_decode_compress_%s", is_neox ? "neox" : "norm");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_compressor_update_decode(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[1];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    snprintf(base, 256, "kernel_dsv4_compressor_update_decode_%s", is_neox ? "neox" : "norm");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_compressor_update_decode_v2(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE_V2);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[1];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    snprintf(base, 256, "kernel_dsv4_compressor_update_decode_v2_%s", is_neox ? "neox" : "norm");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_kv_finalize_decode(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_KV_FINALIZE_DECODE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_dsv4_kv_finalize_decode_%s_%s",
            ggml_type_name(op->src[1]->type),
            ggml_type_name(op->src[2]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_ffn_moe_decode_stage(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_FFN_MOE_DECODE_STAGE);
    assert(op->src[0] != nullptr);
    assert(op->src[1] != nullptr);

    if (op->src[0]->type != GGML_TYPE_Q2_K || op->src[1]->type != GGML_TYPE_F32) {
        return {};
    }

    const int nsg = N_SG_Q2_K;
    const int nr0 = N_R0_Q2_K;

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_ffn_moe_decode_stage_%s_%s",
            ggml_type_name(op->src[0]->type),
            ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }

    res.nr0 = nr0;
    res.nr1 = 1;
    res.nsg = nsg;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_routed_moe_one_tensor_decode(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE);
    for (int i = 0; i < 9; ++i) {
        assert(op->src[i] != nullptr);
    }

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_routed_moe_one_tensor_decode_dryrun");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nr0 = 1;
    res.nr1 = 1;
    res.nsg = 1;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_decode_layer_executor_dryrun(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN);
    for (int i = 0; i < 9; ++i) {
        assert(op->src[i] != nullptr);
    }

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_decode_layer_executor_dryrun");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nr0 = 1;
    res.nr1 = 1;
    res.nsg = 1;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_decode_layer(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_DECODE_LAYER);
    assert(op->src[0] != nullptr);

    char base[256];
    char name[256];
    snprintf(base, 256, "kernel_dsv4_decode_layer_stub");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nr0 = 1;
    res.nr1 = 1;
    res.nsg = 1;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_rope_tail(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_ROPE_TAIL);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[1];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    snprintf(base, 256, "kernel_dsv4_rope_tail_%s_%s", is_neox ? "neox" : "norm", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_rope_hadamard_fp4_quantize(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_ROPE_TAIL);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[1];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    snprintf(base, 256, "kernel_dsv4_rope_hadamard_fp4_quantize_%s_%s", is_neox ? "neox" : "norm", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_rope_fp8_kv_quantize(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_DSV4_ROPE_TAIL);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[1];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    snprintf(base, 256, "kernel_dsv4_rope_fp8_kv_quantize_%s_%s", is_neox ? "neox" : "norm", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_rope_fp8_kv_set_rows(ggml_metal_library_t lib, const ggml_tensor * op, const ggml_tensor * set_rows) {
    assert(op->op == GGML_OP_DSV4_ROPE_TAIL);
    assert(set_rows->op == GGML_OP_SET_ROWS);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[1];
    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    snprintf(base, 256, "kernel_dsv4_rope_fp8_kv_set_rows_%s_%s_%s_%s",
            is_neox ? "neox" : "norm",
            ggml_type_name(op->src[0]->type),
            ggml_type_name(set_rows->type),
            ggml_type_name(set_rows->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_IM2COL);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_UPSCALE);

    char base[256];
    char name[256];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);
    const ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    const bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);

    if (mode == GGML_SCALE_MODE_BILINEAR) {
        snprintf(base, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } else if (mode == GGML_SCALE_MODE_BICUBIC) {
        snprintf(base, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", base, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (res.pipeline) {
        return res;
    }

    res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD_REFLECT_1D);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARANGE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_ADAMW);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_SGD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, const ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal(ggml_metal_library_t lib, const ggml_tensor *  op) {
    assert(op->op == GGML_OP_COUNT_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    char base[256];
    char name[256];

    int nsg = 1;
    while (32*nsg < ne00 && nsg < 32) {
        nsg *= 2;
    }

    snprintf(base, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_COUNT_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32 * sizeof(int32_t);
    res.nsg  = nsg;

    return res;
}
