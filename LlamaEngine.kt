package com.localllm.app

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Kotlin wrapper around the native llama.cpp JNI bridge.
 */
class LlamaEngine {

    // ── Token callback interface (implemented in JNI) ─────────────────────
    fun interface TokenCallback {
        fun onToken(piece: String)
    }

    // ── State ─────────────────────────────────────────────────────────────
    var isLoaded = false
        private set

    // ── Public API ────────────────────────────────────────────────────────

    suspend fun loadModel(
        modelPath: String,
        nGpuLayers: Int = 99,   // 99 = offload everything to GPU if available
        nCtx: Int = 4096
    ): Boolean = withContext(Dispatchers.IO) {
        val ok = nativeLoad(modelPath, nGpuLayers, nCtx)
        isLoaded = ok
        ok
    }

    suspend fun completion(
        prompt: String,
        nPredict: Int = 512,
        onToken: (String) -> Unit
    ) = withContext(Dispatchers.IO) {
        nativeCompletion(prompt, nPredict, TokenCallback { onToken(it) })
    }

    fun stop() = nativeStop()

    fun free() {
        nativeFree()
        isLoaded = false
    }

    fun modelInfo(): String = nativeModelInfo()

    /**
     * Build a Llama-3-Instruct formatted prompt from a conversation history.
     */
    fun buildPrompt(messages: List<ChatMessage>, systemPrompt: String): String {
        val sb = StringBuilder()
        sb.append("<|begin_of_text|>")
        sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
        sb.append(systemPrompt)
        sb.append("<|eot_id|>")
        for (msg in messages) {
            val role = if (msg.isUser) "user" else "assistant"
            sb.append("<|start_header_id|>$role<|end_header_id|>\n\n")
            sb.append(msg.text)
            sb.append("<|eot_id|>")
        }
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return sb.toString()
    }

    // ── Native declarations ───────────────────────────────────────────────
    private external fun nativeLoad(path: String, nGpuLayers: Int, nCtx: Int): Boolean
    private external fun nativeCompletion(prompt: String, nPredict: Int, callback: TokenCallback)
    private external fun nativeStop()
    private external fun nativeFree()
    private external fun nativeModelInfo(): String

    companion object {
        init {
            System.loadLibrary("localllm")
        }
    }
}
