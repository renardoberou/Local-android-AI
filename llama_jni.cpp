#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <atomic>
#include <thread>

#include "llama.h"
#include "ggml.h"

#define TAG "LocalLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ─── Global state ────────────────────────────────────────────────────────────
static llama_model   *g_model   = nullptr;
static llama_context *g_ctx     = nullptr;
static llama_sampler *g_sampler = nullptr;
static std::atomic<bool> g_stop{false};

// ─── Helpers ─────────────────────────────────────────────────────────────────
static std::string token_to_str(llama_context *ctx, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(llama_get_model(ctx), tok, buf, sizeof(buf), 0, true);
    if (n < 0) return "";
    return std::string(buf, n);
}

// ─── JNI: load model ─────────────────────────────────────────────────────────
extern "C" JNIEXPORT jboolean JNICALL
Java_com_localllm_app_LlamaEngine_nativeLoad(
        JNIEnv *env, jobject /*thiz*/,
        jstring model_path, jint n_gpu_layers, jint n_ctx) {

    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading model: %s  gpu_layers=%d  ctx=%d", path, n_gpu_layers, n_ctx);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(model_path, path);

    if (!g_model) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = (uint32_t) n_ctx;
    cparams.n_batch  = 512;
    cparams.n_ubatch = 512;
    cparams.flash_attn = true;

    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    // Greedy + temperature sampler chain
    g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

// ─── JNI: run completion ──────────────────────────────────────────────────────
extern "C" JNIEXPORT void JNICALL
Java_com_localllm_app_LlamaEngine_nativeCompletion(
        JNIEnv *env, jobject /*thiz*/,
        jstring j_prompt, jint n_predict,
        jobject callback) {

    if (!g_model || !g_ctx) {
        LOGE("Model not loaded");
        return;
    }

    const char *prompt_c = env->GetStringUTFChars(j_prompt, nullptr);
    std::string prompt(prompt_c);
    env->ReleaseStringUTFChars(j_prompt, prompt_c);

    // Tokenise
    int n_tokens_max = llama_n_ctx(g_ctx);
    std::vector<llama_token> tokens(n_tokens_max);
    int n_tokens = llama_tokenize(
            llama_get_model(g_ctx), prompt.c_str(), prompt.size(),
            tokens.data(), n_tokens_max, true, true);

    if (n_tokens < 0) {
        LOGE("Tokenisation failed: %d", n_tokens);
        return;
    }
    tokens.resize(n_tokens);

    // Clear KV cache, reset sampler
    llama_kv_cache_clear(g_ctx);
    llama_sampler_reset(g_sampler);
    g_stop = false;

    // Callback refs
    jclass   cb_class  = env->GetObjectClass(callback);
    jmethodID onToken  = env->GetMethodID(cb_class, "onToken", "(Ljava/lang/String;)V");

    // Initial prompt batch
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("Initial decode failed");
        return;
    }

    // Autoregressive generation
    llama_token eos = llama_token_eos(g_model);
    int generated = 0;

    while (generated < n_predict && !g_stop) {
        llama_token tok = llama_sampler_sample(g_sampler, g_ctx, -1);
        if (tok == eos) break;

        std::string piece = token_to_str(g_ctx, tok);
        jstring j_piece = env->NewStringUTF(piece.c_str());
        env->CallVoidMethod(callback, onToken, j_piece);
        env->DeleteLocalRef(j_piece);

        // Check for Java exceptions (stop requested)
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
            break;
        }

        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(g_ctx, next) != 0) break;
        generated++;
    }
}

// ─── JNI: stop generation ────────────────────────────────────────────────────
extern "C" JNIEXPORT void JNICALL
Java_com_localllm_app_LlamaEngine_nativeStop(JNIEnv *, jobject) {
    g_stop = true;
}

// ─── JNI: free model ─────────────────────────────────────────────────────────
extern "C" JNIEXPORT void JNICALL
Java_com_localllm_app_LlamaEngine_nativeFree(JNIEnv *, jobject) {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);              g_ctx     = nullptr; }
    if (g_model)   { llama_model_free(g_model);      g_model   = nullptr; }
    llama_backend_free();
    LOGI("Model freed");
}

// ─── JNI: model info ─────────────────────────────────────────────────────────
extern "C" JNIEXPORT jstring JNICALL
Java_com_localllm_app_LlamaEngine_nativeModelInfo(JNIEnv *env, jobject) {
    if (!g_model) return env->NewStringUTF("No model loaded");
    char buf[512];
    llama_model_desc(g_model, buf, sizeof(buf));
    return env->NewStringUTF(buf);
}
