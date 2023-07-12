#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "common.h"
#include "ggml/ggml.h"

std::vector<gpt_vocab::id> ws_tokenize(const gpt_vocab& vocab, const std::string& text)
{
    // Normalize: technically need to do NFKC and StripAccents, but will ignore
    // for now
    // 1. make text lowercase
    std::string lower = text;
    for (auto& c : lower)
        c = std::tolower(c);
    printf("%s: lower = '%s'\n", __func__, lower.c_str());

    // Pretokenize: convert to bytes and split digits
    // the gpt_tokenize function does this well
    // splitting into bytes is done by std::string::substr, which
    // operates on byte indices and not character indices
    // it doesn't split into digits, so multi-number inputs will take
    // longer to parse, but we'll optimize only if we need to
    // add space if first word doesn't start with a space
    if (lower[0] != ' ')
        lower = " " + lower;
    return gpt_tokenize(vocab, lower);
};

struct ws_hparams {
    int32_t d_model;
    int32_t n_heads;
    int32_t n_layers;
    int32_t vocab_size;
    int32_t d_head;
    int32_t ftype;

    int32_t max_tokens = 512; // n_ctx in other files but ctx is super overloaded
};

struct ws_attention {
    struct ggml_tensor* residual_to_qkv;
    struct ggml_tensor* concat_attention_to_residual;
};

struct ws_block {
    struct ggml_tensor* layer_norm_before_attention;
    struct ws_attention attention;
    struct ggml_tensor* layer_norm_before_ffn;

    // ffn definition is fused
    // the actual names of these are ffn.mlp.0.weight and ffn.mlp.2.weight
    struct ggml_tensor* ffn_in; // d_model to 4 * d_model
    struct ggml_tensor* ffn_out; // 4 * d_model to d_model
};

struct ws_model {
    struct ws_hparams hparams;
    struct ggml_tensor* tokens_to_embeddings;
    std::vector<ws_block> blocks;
    struct ggml_tensor* layer_norm_final;
    struct ggml_tensor* embeddings_to_logits; // this is the same as tokens_to_embeddings; remove eventually

    struct ggml_context* ctx;
    std::map<std::string, struct ggml_tensor*> tensors;
};

bool ws_model_load(const std::string& fname, ws_model& model, gpt_vocab& vocab)
{
    printf("%s: loading model from %s\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char*)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__,
                fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto& hparams = model.hparams;

        fin.read((char*)&hparams.d_model, sizeof(hparams.d_model));
        fin.read((char*)&hparams.n_heads, sizeof(hparams.n_heads));
        fin.read((char*)&hparams.n_layers, sizeof(hparams.n_layers));
        fin.read((char*)&hparams.vocab_size, sizeof(hparams.vocab_size));
        fin.read((char*)&hparams.d_head, sizeof(hparams.d_head));
        fin.read((char*)&hparams.ftype, sizeof(hparams.ftype));

        printf("%s: d_model    = %d\n", __func__, hparams.d_model);
        printf("%s: n_heads    = %d\n", __func__, hparams.n_heads);
        printf("%s: n_layers   = %d\n", __func__, hparams.n_layers);
        printf("%s: vocab_size = %d\n", __func__, hparams.vocab_size);
        printf("%s: d_head     = %d\n", __func__, hparams.d_head);
        printf("%s: ftype      = %d\n", __func__, hparams.ftype);
    }

    // load vocab
    {
        const int32_t vocab_size = model.hparams.vocab_size;

        std::string word;
        std::vector<char> buf(128);

        // TODO: need to think about special tokens
        for (int i = 0; i < vocab_size; i++) {
            uint32_t len;
            fin.read((char*)&len, sizeof(len));

            buf.resize(len);
            fin.read((char*)buf.data(), len);
            word.assign(buf.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // calculate total ctx size
    // could be done based on the model file (llama does this), but will do
    // manually for now
    size_t ctx_size = 0;
    const size_t d_model = model.hparams.d_model;
    const size_t n_heads = model.hparams.n_heads;
    const size_t n_layers = model.hparams.n_layers;
    const size_t vocab_size = model.hparams.vocab_size;
    // should just be all f32
    const ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)model.hparams.ftype);
    // believe this is important when quantizing
    // right now it's just size(float) -> 32 bits
    const size_t ftype_size = ggml_type_sizef(wtype);
    {
        ctx_size += vocab_size * d_model * ftype_size + ggml_tensor_overhead(); // tokens_to_embeddings

        // blocks
        ctx_size += n_layers * (d_model * ftype_size + ggml_tensor_overhead()); // layer_norm_before_attention
        ctx_size += n_layers * (d_model * (d_model * 3) * ftype_size + ggml_tensor_overhead()); // residual_to_qkv
        ctx_size += n_layers * (d_model * d_model * ftype_size + ggml_tensor_overhead()); // concat_attention_to_residual
        ctx_size += n_layers * (d_model * ftype_size + ggml_tensor_overhead()); // layer_norm_before_ffn
        ctx_size += n_layers * (d_model * (d_model * 4) * ftype_size + ggml_tensor_overhead()); // ffn_in
        ctx_size += n_layers * ((d_model * 4 * d_model) * ftype_size + ggml_tensor_overhead()); // ffn_out

        ctx_size += d_model * ftype_size + ggml_tensor_overhead(); // layer_norm_final
        ctx_size += vocab_size * d_model * ftype_size + ggml_tensor_overhead(); // embeddings_to_logits

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
    }

    // create context
    {
        struct ggml_init_params params;
        params.mem_size = ctx_size;
        params.mem_buffer = NULL;
        params.no_alloc = false;

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // create weight tensors
    {
        auto& ctx = model.ctx;
        model.blocks.resize(n_layers);

        // model.tokens_to_embeddings = ggml_new_tensor_2d(ctx, wtype, vocab_size, d_model);
        model.tokens_to_embeddings = ggml_new_tensor_2d(ctx, wtype, d_model, vocab_size);
        model.tensors["tokens_to_embeddings.weight"] = model.tokens_to_embeddings;

        for (int i = 0; i < (int)n_layers; i++) {
            auto& block = model.blocks[i];

            block.layer_norm_before_attention = ggml_new_tensor_1d(ctx, wtype, d_model);
            block.attention.residual_to_qkv = ggml_new_tensor_2d(ctx, wtype, d_model, d_model * 3);
            block.attention.concat_attention_to_residual = ggml_new_tensor_2d(ctx, wtype, d_model, d_model);
            block.layer_norm_before_ffn = ggml_new_tensor_1d(ctx, wtype, d_model);
            block.ffn_in = ggml_new_tensor_2d(ctx, wtype, d_model, d_model * 4);
            block.ffn_out = ggml_new_tensor_2d(ctx, wtype, d_model * 4, d_model);

            model.tensors["blocks." + std::to_string(i) + ".layer_norm_before_attention.weight"] = block.layer_norm_before_attention;
            model.tensors["blocks." + std::to_string(i) + ".attention.residual_to_qkv.weight"] = block.attention.residual_to_qkv;
            model.tensors["blocks." + std::to_string(i) + ".attention.concat_attention_to_residual.weight"] = block.attention.concat_attention_to_residual;
            model.tensors["blocks." + std::to_string(i) + ".layer_norm_before_ffn.weight"] = block.layer_norm_before_ffn;
            model.tensors["blocks." + std::to_string(i) + ".ffn.mlp.0.weight"] = block.ffn_in;
            model.tensors["blocks." + std::to_string(i) + ".ffn.mlp.2.weight"] = block.ffn_out;
        }

        model.layer_norm_final = ggml_new_tensor_1d(ctx, wtype, d_model);
        model.tensors["layer_norm_final.weight"] = model.layer_norm_final;

        model.embeddings_to_logits = model.tokens_to_embeddings;
        model.tensors["embeddings_to_logits.weight"] = model.embeddings_to_logits;
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;
        printf("%s: loading weights\n", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            // read number of dims, length of name, and data type
            fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char*>(&length), sizeof(length));
            fin.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

            if (fin.eof()) {
                // technically the first read takes us past eof,
                // but we check here for code cleanliness
                break;
            }

            // read dimension sizes
            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            // read name
            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            // error checking for tensor
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has %d elements, but model file has %d\n", __func__, name.data(), ggml_nelements(tensor), nelements);
                return false;
            }
            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr,
                    "%s: tensor '%s' has wrong shape in model file: got [%5d, "
                    "%5d], expected [%5d, %5d]\n",
                    __func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1], ne[0], ne[1]);
                return false;
            }
            if ((nelements * ftype_size) != ggml_nbytes(tensor)) {
                fprintf(stderr,
                    "%s: tensor '%s' has wrong size in model file: got %zu, "
                    "expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor), nelements * ftype_size);
                return false;
            }

            // read data
            fin.read(reinterpret_cast<char*>(tensor->data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 10 == 0) {
                printf("%s: loaded %d tensors, total size = %6.2f MB\n", __func__, n_tensors, total_size / (1024.0 * 1024.0));
            }
        }
        printf("%s: loaded %d tensors, total size = %6.2f MB\n", __func__, n_tensors, total_size / (1024.0 * 1024.0));
        printf("%s: done! model size = %8.2f MB / num tensors = %d\n", __func__, total_size / (1024.0 * 1024.0), n_tensors);
    }

    fin.close();

    return true;
}

// key-value cache for faster generation
// tbh this is definitely overengineering for how small
// this model is, but it's fun to implement
struct ws_kv_cache {
    struct ggml_tensor* k;
    struct ggml_tensor* v;

    struct ggml_context* ctx;

    ws_kv_cache(const ws_hparams& hparams)
    {
        size_t n_layers = hparams.n_layers;
        size_t d_model = hparams.d_model;

        ggml_init_params params;

        size_t buffer_mem = 0;

        // memory size explanation:
        // each token (hparams.max_tokens) produces a key and value vector of size d_model
        // there are n_layers layers, so we need to store n_layers * d_model per token

        buffer_mem += hparams.max_tokens * n_layers * d_model * sizeof(float); // k memory
        buffer_mem += hparams.max_tokens * n_layers * d_model * sizeof(float); // v memory

        params.mem_size = buffer_mem;
        params.mem_buffer = NULL;
        params.no_alloc = false;
        ctx = ggml_init(params);

        k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_layers * d_model);
        v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_layers * d_model);
    }

    ~ws_kv_cache()
    {
        ggml_free(ctx);
    }
};

// context for generation
struct ws_context {
    const ws_model& model;
    const gpt_vocab& vocab;

    struct ws_kv_cache kv_cache;

    // size_t mem_per_token = 0;
    // std::vector<float> logits;
    // std::vector<float> embedding; // ?

    ws_context(const ws_model& model, const gpt_vocab& vocab)
        : model(model)
        , vocab(vocab)
        , kv_cache(model.hparams)
    {
    }
};

// like a single forward pass
// input: token ids, output: logits
bool ws_eval(ws_context& wctx, const std::vector<gpt_vocab::id>& token_ids)
{
    const auto& model = wctx.model;
    const size_t d_model = wctx.model.hparams.d_model;
    const size_t n_heads = wctx.model.hparams.n_heads;
    const size_t n_layers = wctx.model.hparams.n_layers;
    const size_t vocab_size = wctx.model.hparams.vocab_size;

    // buffer for the inference context (100 MB)
    // ideally this is dynamic based on number of inputs
    static size_t buffer_size = 300 * 1024 * 1024;
    static void* buffer = malloc(buffer_size);

    struct ggml_init_params inf_params;
    inf_params.mem_size = buffer_size;
    inf_params.mem_buffer = buffer;
    inf_params.no_alloc = false;

    // inference context
    // needed to create the needed tensors
    struct ggml_context* inf_ctx = ggml_init(inf_params);

    // computational graph
    struct ggml_cgraph gf;

    const int num_tokens = token_ids.size();

    // tensor for ids
    struct ggml_tensor* input_ids = ggml_new_tensor_1d(inf_ctx, GGML_TYPE_I32, num_tokens);
    memcpy(input_ids->data, token_ids.data(), num_tokens * ggml_element_size(input_ids));

    // get embeddings
    // everything is transposed so it's [d_model, num_tokens, 1, 1]
    // this makes sense from a memory perspective
    struct ggml_tensor* residual = ggml_get_rows(inf_ctx, model.tokens_to_embeddings, input_ids);
    printf("%s: embeddings shape: [%zu, %zu, %zu, %zu]\n", __func__, residual->ne[0], residual->ne[1], residual->ne[2], residual->ne[3]);

    // blocks
    for (int i = 0; i < n_layers; i++) {
        struct ggml_tensor* cur;

        // layer norm before attention
        {
            // norm operates on first dimension, which works well with the transposed embeddings
            cur = ggml_norm(inf_ctx, residual); // (x - E[x]) / sqrt(Var[x] + eps)

            // input shapes: layer_norm_before_attention = [d_model, 1, 1, 1], cur = [d_model, num_tokens, 1, 1]
            // this repeat is weird. so mul needs src0 to be able to be represented as a repitition of src1
            // to allow broadcasting. not sure why, but in this case we can't broadcast cur to layer_norm's shape
            // since it's a bigger tensor. to fix this, we manually repeat layer_norm to be the same shape as cur

            // actually, maybe it's fine... multiply should be commutative
            cur = ggml_mul(inf_ctx, cur, model.blocks[i].layer_norm_before_attention); // x * gamma. should broadcast automatically
        }

        // self-attention
        {
            // matmul output shape is src0->ne[1] x src1->ne[1]
            // transpose happens inside the function
            // input shapes: W_qkv = [d_model, d_model * 3], cur = [d_model, num_tokens, 1, 1]
            // output shape: [d_model * 3, num_tokens, 1, 1]
            struct ggml_tensor* qkv = ggml_mul_mat(inf_ctx, model.blocks[i].attention.residual_to_qkv, cur); // x * W_qkv;

            // split into q, k, v
            // shapes: all are [d_model, num_tokens, 1, 1]
            // doesn't allocate new data - pass in the offset of where to start reading
            // also pass in the same stride (d_model x 3 x dtype) so it works
            struct ggml_tensor* q = ggml_view_2d(inf_ctx, qkv, d_model, num_tokens, qkv->nb[1], 0 * sizeof(float) * d_model);
            struct ggml_tensor* k = ggml_view_2d(inf_ctx, qkv, d_model, num_tokens, qkv->nb[1], 1 * sizeof(float) * d_model);
            struct ggml_tensor* v = ggml_view_2d(inf_ctx, qkv, d_model, num_tokens, qkv->nb[1], 2 * sizeof(float) * d_model);

            // memory storage
            // store key and value to memory
            struct ggml_tensor* k_mem = ggml_view_1d(inf_ctx, wctx.kv_cache.k, num_tokens * d_model,
                (ggml_element_size(wctx.kv_cache.k) * d_model) * (i));
            struct ggml_tensor* v_mem = ggml_view_1d(inf_ctx, wctx.kv_cache.v, num_tokens * d_model,
                (ggml_element_size(wctx.kv_cache.v) * d_model) * (i));

            ggml_build_forward_expand(&gf, ggml_cpy(inf_ctx, k, k_mem));
            ggml_build_forward_expand(&gf, ggml_cpy(inf_ctx, v, v_mem));

            // matrix multiplication
            // we want to rearrange q and k from [d_model, num_tokens, 1, 1] to [d_head, num_tokens, n_heads, 1]
            // rearranging is annoying because ggml doesn't support transposed/permuted matrices
            // nb0 must be sizeof(dtype) for all tensors
            // q, k, v calculation is also a bit different from training: the q/k/v matrices from above only give us
            // the new matrices from the new tokens, but we may have context from before
            // so q will have num_tokens cols, but k will have num_tokens + num_prev cols, same for v

            const size_t d_head = d_model / n_heads;

            // destination must be contiguous
            // for src tensors, nb0 has to be sizeof(dtype) - but no other restrictions
            // tried to use reshape for this, but required contiguous tensor
            struct ggml_tensor* q_3d = ggml_cpy(inf_ctx, q, ggml_new_tensor_3d(inf_ctx, GGML_TYPE_F32, d_head, n_heads, num_tokens));
            struct ggml_tensor* q_contiguous = ggml_permute(inf_ctx, q_3d, 0, 2, 1, 3);

            // struct ggml_tensor* k_3d = ggml_cpy(inf_ctx, k, ggml_new_tensor_3d(inf_ctx, GGML_TYPE_F32, d_head, n_heads, num_tokens));
            // struct ggml_tensor* k_contiguous = ggml_permute(inf_ctx, k_3d, 0, 2, 1, 3);

            struct ggml_tensor* k_contiguous = ggml_permute(inf_ctx,
                ggml_reshape_3d(inf_ctx,
                    ggml_view_1d(inf_ctx, wctx.kv_cache.k, (num_tokens)*d_model,
                        i * ggml_element_size(wctx.kv_cache.k) * d_model),
                    d_head, n_heads, num_tokens),
                0, 2, 1, 3);

            // shape of q and k are now [d_head, num_tokens, n_heads, 1]
            // Q K^T
            // shape: [num_tokens, num_tokens, n_heads, 1]
            struct ggml_tensor* q_kt = ggml_mul_mat(inf_ctx, k_contiguous, q_contiguous);

            // mask
            struct ggml_tensor* q_kt_masked = ggml_diag_mask_inf_inplace(inf_ctx, q_kt, 0); // n_past is 0 for now with the same q and k

            // scaled
            // scale is 1 / sqrt(d_head)
            struct ggml_tensor* q_kt_scaled = ggml_scale_inplace(inf_ctx, q_kt_masked, ggml_new_f32(inf_ctx, 1.0f / sqrt(float(d_head))));

            // softmax
            struct ggml_tensor* q_kt_softmax = ggml_soft_max_inplace(inf_ctx, q_kt_scaled);

            // if we did the same permutation for v as for q and k, we would have a
            // [d_head, num_tokens, n_heads, 1] matrix
            // however, this wouldn't work for the matmul below, which needs [num_tokens, d_head, n_heads, 1]
            // I guess ggml_mul_mat transposes the first matrix, so we need to transpose v's first two dims
            // shape: [num_tokens, d_head, n_heads, 1]
            // permute is a little confusing: way to think of it, is if the permute transpose is [1, 2, 0, 3],
            // then the 0th dim input -> 1st dim output; 1st dim input -> 2nd dim output; and so on
            // the matmul doesn't like v being transposed, so we need to copy it to another tensor

            // struct ggml_tensor* v_3d = ggml_cpy(inf_ctx, v, ggml_new_tensor_3d(inf_ctx, GGML_TYPE_F32, d_head, n_heads, num_tokens));
            // struct ggml_tensor* v_permute = ggml_permute(inf_ctx, v_3d, 1, 2, 0, 3);
            // struct ggml_tensor* v_contiguous = ggml_cpy(inf_ctx, v_permute, ggml_new_tensor_3d(inf_ctx, GGML_TYPE_F32, num_tokens, d_head, n_heads));

            struct ggml_tensor* v_contiguous = ggml_cpy(
                inf_ctx,
                ggml_permute(inf_ctx,
                    ggml_reshape_3d(inf_ctx, ggml_view_1d(inf_ctx, wctx.kv_cache.v, (num_tokens)*d_model, i * ggml_element_size(wctx.kv_cache.v) * d_model),
                        d_head, n_heads, num_tokens),
                    1, 2, 0, 3),
                ggml_new_tensor_3d(inf_ctx, GGML_TYPE_F32, num_tokens, d_head, n_heads));

            // softmax(Q K^T) V
            // input shapes: v_contiguous = [num_tokens, d_head, n_heads, 1], q_kt_softmax = [num_tokens, num_tokens, n_heads, 1]
            // output shape: [d_head, num_tokens, n_heads, 1]
            struct ggml_tensor* q_kt_softmax_v = ggml_mul_mat(inf_ctx, v_contiguous, q_kt_softmax);

            // transpose back to [d_model, num_tokens, 1, 1]
            // but first, we need to shuffle around the dims to get [d_head, n_heads, num_tokens, 1]
            // shape: [d_model, num_tokens, 1, 1]
            struct ggml_tensor* concat_attention = ggml_cpy(inf_ctx, ggml_permute(inf_ctx, q_kt_softmax_v, 0, 2, 1, 3), ggml_new_tensor_2d(inf_ctx, GGML_TYPE_F32, d_model, num_tokens));

            // concat_attention_to_residual
            // input shapes: concat_attention_to_residual = [d_model, d_model], concat_attention = [d_model, num_tokens, 1, 1],
            // output shape: [d_model, num_tokens, 1, 1]
            cur = ggml_mul_mat(inf_ctx, model.blocks[i].attention.concat_attention_to_residual, concat_attention);
        }

        // add residual with result
        residual = ggml_add(inf_ctx, residual, cur);

        // layer norm before ffn
        {
            cur = ggml_norm(inf_ctx, residual); // (x - E[x]) / sqrt(Var[x] + eps)
            cur = ggml_mul(inf_ctx, cur, model.blocks[i].layer_norm_before_ffn); // x * gamma
        }

        // ffn
        {
            // input shapes: ffn_in = [d_model, d_model * 4], cur = [d_model, num_tokens, 1, 1]
            // output shape: [d_model * 4, num_tokens, 1, 1]
            struct ggml_tensor* ffn = ggml_mul_mat(inf_ctx, model.blocks[i].ffn_in, cur);

            // gelu
            struct ggml_tensor* ffn_gelu = ggml_gelu(inf_ctx, ffn);

            // ffn_out
            // input shapes: ffn_out = [d_model * 4, d_model], ffn_gelu = [d_model * 4, num_tokens, 1, 1]
            // output shape: [d_model, num_tokens, 1, 1]
            cur = ggml_mul_mat(inf_ctx, model.blocks[i].ffn_out, ffn_gelu);
        }

        // add residual with result
        residual = ggml_add(inf_ctx, residual, cur);
    }

    // layer norm final
    {
        residual = ggml_norm(inf_ctx, residual);
        residual = ggml_mul(inf_ctx, residual, model.layer_norm_final); // x * gamma
    }

    // embeddings to logits
    residual = ggml_mul_mat(inf_ctx, model.embeddings_to_logits, residual);

    // run computation
    ggml_build_forward_expand(&gf, residual);
    ggml_graph_compute_with_ctx(inf_ctx, &gf, 1); // 1 is n_threads

    // get max logit in last row
    std::vector<float> logits(vocab_size);
    memcpy(logits.data(), (float*)ggml_get_data(residual) + (vocab_size * (num_tokens - 1)), vocab_size * sizeof(float));

    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        logits_id.push_back(std::make_pair(logits[i], i));
    }

    // top 5 tokens
    std::partial_sort(logits_id.begin(), logits_id.begin() + 5, logits_id.end(), std::greater<std::pair<double, gpt_vocab::id>>());
    // print
    for (int i = 0; i < 5; i++) {
        auto id = logits_id[i].second;
        auto token = wctx.vocab.id_to_token.find(id)->second.c_str();
        printf("%s: token[%d] = %s, logit = %6.2f\n", __func__, id, token, logits_id[i].first);
    }

    ggml_free(inf_ctx);

    return true;
}

int main(int argc, char** argv)
{
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    ws_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        const std::string model_path = "models/wabisabi/ggml-model-f32.bin";

        if (!ws_model_load(model_path, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__,
                model_path.c_str());
            return 1;
        }
    }

    // try tokenizing
    std::string prompt = "One day, a little girl named";
    printf("%s: Tokenizing prompt: '%s'\n", __func__, prompt.c_str());
    std::vector<gpt_vocab::id> tokenized_prompt = ws_tokenize(vocab, prompt);

    for (size_t i = 0; i < tokenized_prompt.size(); i++) {
        printf("%s: token[%zu] = %6d\n", __func__, i, tokenized_prompt[i]);
    }

    ws_context ctx = ws_context(model, vocab);
    ws_eval(ctx, tokenized_prompt);

    ggml_free(model.ctx);
    return 0;
}