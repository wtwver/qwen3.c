#include "npu_matmul.h"

#ifdef QWEN3_DISABLE_NPU
int npu_matmul_init(NpuMatmulContext *ctx, const Config *config,
                    const TransformerWeights *weights) {
    (void)config;
    (void)weights;
    if (ctx) ctx->enabled = 0;
    return 0;
}

void npu_matmul_shutdown(NpuMatmulContext *ctx) {
    if (ctx) ctx->enabled = 0;
}

int npu_matmul_run(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                   const float *in, float *out) {
    (void)ctx;
    (void)kind;
    (void)layer;
    (void)in;
    (void)out;
    return 0;
}

void npu_matmul_reset_stats(NpuMatmulContext *ctx) {
    if (!ctx) return;
    ctx->npu_ops = 0;
    ctx->cpu_ops = 0;
}
#else

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Silence verbose RKNN register/memory logs.
#define printf(...) ((void)0)
#define fprintf(...) ((void)0)
#include "rknnops.h"
#undef printf
#undef fprintf

static int is_supported_shape(int M, int K, int N) {
    if (M != 1) return 0;
    if (K == 1024 && N == 2048) return 1;
    if (K == 1024 && N == 1024) return 1;
    if (K == 2048 && N == 1024) return 1;
    if (K == 1024 && N == 3072) return 1;
    if (K == 3072 && N == 1024) return 1;
    if (K == 1024 && N == 151936) return 1;
    return 0;
}

static void dequantize_to_fp16(__fp16 *dst, const QuantizedTensor *src, int n) {
    for (int i = 0; i < n; i++) {
        float val = src->q[i] * src->s[i / GS];
        dst[i] = (__fp16)val;
    }
}

static void free_weights_array(__fp16 **weights, int n_layers) {
    if (!weights) return;
    for (int i = 0; i < n_layers; i++) {
        free(weights[i]);
    }
    free(weights);
}

static __fp16 **alloc_weights_array(int n_layers) {
    return (__fp16**)calloc((size_t)n_layers, sizeof(__fp16*));
}

static int any_weights_ready(const NpuMatmulContext *ctx) {
    if (!ctx) return 0;
    for (int l = 0; l < ctx->n_layers; l++) {
        if (ctx->wq && ctx->wq[l]) return 1;
        if (ctx->wk && ctx->wk[l]) return 1;
        if (ctx->wv && ctx->wv[l]) return 1;
        if (ctx->wo && ctx->wo[l]) return 1;
        if (ctx->w1 && ctx->w1[l]) return 1;
        if (ctx->w2 && ctx->w2[l]) return 1;
        if (ctx->w3 && ctx->w3[l]) return 1;
    }
    if (ctx->wcls) return 1;
    return 0;
}

static void maybe_log(NpuMatmulContext *ctx, const char *fmt, ...) {
    if (!ctx || !ctx->verbose) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

int npu_matmul_init(NpuMatmulContext *ctx, const Config *config,
                    const TransformerWeights *weights) {
    if (!ctx || !config || !weights) return 0;
    memset(ctx, 0, sizeof(*ctx));
    const char *env = getenv("NPU");
    if (!env || env[0] == '0') {
        return 0;
    }
    ctx->enabled = 1;
    ctx->verbose = getenv("QWEN3_NPU_VERBOSE") ? 1 : 0;
    ctx->dim = config->dim;
    ctx->hidden_dim = config->hidden_dim;
    ctx->n_layers = config->n_layers;
    ctx->all_heads_dim = config->n_heads * config->head_dim;
    ctx->kv_dim = config->n_kv_heads * config->head_dim;
    ctx->vocab_size = config->vocab_size;

    ctx->wq = alloc_weights_array(ctx->n_layers);
    ctx->wk = alloc_weights_array(ctx->n_layers);
    ctx->wv = alloc_weights_array(ctx->n_layers);
    ctx->wo = alloc_weights_array(ctx->n_layers);
    ctx->w1 = alloc_weights_array(ctx->n_layers);
    ctx->w2 = alloc_weights_array(ctx->n_layers);
    ctx->w3 = alloc_weights_array(ctx->n_layers);

    for (int l = 0; l < ctx->n_layers; l++) {
        if (is_supported_shape(1, ctx->dim, ctx->all_heads_dim)) {
            int n = ctx->dim * ctx->all_heads_dim;
            ctx->wq[l] = (__fp16*)malloc((size_t)n * sizeof(__fp16));
            if (ctx->wq[l]) dequantize_to_fp16(ctx->wq[l], &weights->wq[l], n);
        }
        if (is_supported_shape(1, ctx->dim, ctx->kv_dim)) {
            int n = ctx->dim * ctx->kv_dim;
            ctx->wk[l] = (__fp16*)malloc((size_t)n * sizeof(__fp16));
            ctx->wv[l] = (__fp16*)malloc((size_t)n * sizeof(__fp16));
            if (ctx->wk[l]) dequantize_to_fp16(ctx->wk[l], &weights->wk[l], n);
            if (ctx->wv[l]) dequantize_to_fp16(ctx->wv[l], &weights->wv[l], n);
        }
        if (is_supported_shape(1, ctx->all_heads_dim, ctx->dim)) {
            int n = ctx->all_heads_dim * ctx->dim;
            ctx->wo[l] = (__fp16*)malloc((size_t)n * sizeof(__fp16));
            if (ctx->wo[l]) dequantize_to_fp16(ctx->wo[l], &weights->wo[l], n);
        }
        if (is_supported_shape(1, ctx->dim, ctx->hidden_dim)) {
            int n = ctx->dim * ctx->hidden_dim;
            ctx->w1[l] = (__fp16*)malloc((size_t)n * sizeof(__fp16));
            ctx->w3[l] = (__fp16*)malloc((size_t)n * sizeof(__fp16));
            if (ctx->w1[l]) dequantize_to_fp16(ctx->w1[l], &weights->w1[l], n);
            if (ctx->w3[l]) dequantize_to_fp16(ctx->w3[l], &weights->w3[l], n);
        }
        if (is_supported_shape(1, ctx->hidden_dim, ctx->dim)) {
            int n = ctx->hidden_dim * ctx->dim;
            ctx->w2[l] = (__fp16*)malloc((size_t)n * sizeof(__fp16));
            if (ctx->w2[l]) dequantize_to_fp16(ctx->w2[l], &weights->w2[l], n);
        }
    }

    if (is_supported_shape(1, ctx->dim, ctx->vocab_size)) {
        int n = ctx->dim * ctx->vocab_size;
        ctx->wcls = (__fp16*)malloc((size_t)n * sizeof(__fp16));
        if (ctx->wcls) dequantize_to_fp16(ctx->wcls, weights->wcls, n);
    }

    ctx->tmp_in_dim = (__fp16*)malloc((size_t)ctx->dim * sizeof(__fp16));
    ctx->tmp_in_hidden = (__fp16*)malloc((size_t)ctx->hidden_dim * sizeof(__fp16));
    ctx->tmp_in_all_heads = (__fp16*)malloc((size_t)ctx->all_heads_dim * sizeof(__fp16));

    if (!any_weights_ready(ctx)) {
        maybe_log(ctx, "QWEN3_NPU: no supported matmul shapes for this model\n");
        npu_matmul_shutdown(ctx);
        return 0;
    }

    maybe_log(ctx, "QWEN3_NPU: enabled (dim=%d hidden=%d heads_dim=%d kv_dim=%d)\n",
              ctx->dim, ctx->hidden_dim, ctx->all_heads_dim, ctx->kv_dim);
    return 1;
}

void npu_matmul_shutdown(NpuMatmulContext *ctx) {
    if (!ctx) return;
    free_weights_array(ctx->wq, ctx->n_layers);
    free_weights_array(ctx->wk, ctx->n_layers);
    free_weights_array(ctx->wv, ctx->n_layers);
    free_weights_array(ctx->wo, ctx->n_layers);
    free_weights_array(ctx->w1, ctx->n_layers);
    free_weights_array(ctx->w2, ctx->n_layers);
    free_weights_array(ctx->w3, ctx->n_layers);
    free(ctx->wcls);
    free(ctx->tmp_in_dim);
    free(ctx->tmp_in_hidden);
    free(ctx->tmp_in_all_heads);
    memset(ctx, 0, sizeof(*ctx));
}

void npu_matmul_reset_stats(NpuMatmulContext *ctx) {
    if (!ctx) return;
    ctx->npu_ops = 0;
    ctx->cpu_ops = 0;
}

static __fp16 *select_input_buffer(NpuMatmulContext *ctx, int K) {
    if (!ctx) return NULL;
    if (K == ctx->dim) return ctx->tmp_in_dim;
    if (K == ctx->hidden_dim) return ctx->tmp_in_hidden;
    if (K == ctx->all_heads_dim) return ctx->tmp_in_all_heads;
    return NULL;
}

int npu_matmul_run(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                   const float *in, float *out) {
    if (!ctx || !ctx->enabled || !in || !out) return 0;
    if (kind != NPU_MATMUL_WCLS && (layer < 0 || layer >= ctx->n_layers)) return 0;

    const __fp16 *weights = NULL;
    int K = 0;
    int N = 0;
    switch (kind) {
        case NPU_MATMUL_WQ:
            weights = ctx->wq ? ctx->wq[layer] : NULL;
            K = ctx->dim;
            N = ctx->all_heads_dim;
            break;
        case NPU_MATMUL_WK:
            weights = ctx->wk ? ctx->wk[layer] : NULL;
            K = ctx->dim;
            N = ctx->kv_dim;
            break;
        case NPU_MATMUL_WV:
            weights = ctx->wv ? ctx->wv[layer] : NULL;
            K = ctx->dim;
            N = ctx->kv_dim;
            break;
        case NPU_MATMUL_WO:
            weights = ctx->wo ? ctx->wo[layer] : NULL;
            K = ctx->all_heads_dim;
            N = ctx->dim;
            break;
        case NPU_MATMUL_W1:
            weights = ctx->w1 ? ctx->w1[layer] : NULL;
            K = ctx->dim;
            N = ctx->hidden_dim;
            break;
        case NPU_MATMUL_W2:
            weights = ctx->w2 ? ctx->w2[layer] : NULL;
            K = ctx->hidden_dim;
            N = ctx->dim;
            break;
        case NPU_MATMUL_W3:
            weights = ctx->w3 ? ctx->w3[layer] : NULL;
            K = ctx->dim;
            N = ctx->hidden_dim;
            break;
        case NPU_MATMUL_WCLS:
            weights = ctx->wcls;
            K = ctx->dim;
            N = ctx->vocab_size;
            break;
        default:
            return 0;
    }

    if (!weights || !is_supported_shape(1, K, N)) return 0;

    __fp16 *input_fp16 = select_input_buffer(ctx, K);
    if (!input_fp16) return 0;

    for (int i = 0; i < K; i++) {
        input_fp16[i] = (__fp16)in[i];
    }

    fprintf(stderr, "matmul npu M=1 K=%d N=%d\n", K, N);
    float *result = float16_matmul(input_fp16, (__fp16*)weights, 11, 1, N, K);
    if (!result) return 0;
    memcpy(out, result, (size_t)N * sizeof(float));
    free(result);
    ctx->npu_ops++;
    return 1;
}
#endif
