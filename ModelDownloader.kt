package com.localllm.app

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.util.concurrent.TimeUnit

data class DownloadProgress(
    val bytesDownloaded: Long,
    val totalBytes: Long,
    val percent: Int = if (totalBytes > 0) ((bytesDownloaded * 100) / totalBytes).toInt() else 0,
    val done: Boolean = false,
    val error: String? = null
)

object ModelDownloader {

    // ── Default model: Llama-3.1-8B-Instruct Q4_K_M ~4.7 GB ─────────────
    const val DEFAULT_MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    const val DEFAULT_MODEL_URL  =
        "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.SECONDS)   // Unlimited — large file
        .build()

    fun modelFile(ctx: Context): File =
        File(ctx.getExternalFilesDir(null), DEFAULT_MODEL_NAME)

    fun isModelDownloaded(ctx: Context): Boolean =
        modelFile(ctx).let { it.exists() && it.length() > 1_000_000_000L }

    suspend fun download(
        ctx: Context,
        url: String = DEFAULT_MODEL_URL,
        onProgress: (DownloadProgress) -> Unit
    ) = withContext(Dispatchers.IO) {

        val dest = modelFile(ctx)
        val tmpFile = File(dest.parent, dest.name + ".tmp")

        // Resume support: check existing partial download
        val existingBytes = if (tmpFile.exists()) tmpFile.length() else 0L

        val reqBuilder = Request.Builder().url(url)
        if (existingBytes > 0) {
            reqBuilder.header("Range", "bytes=$existingBytes-")
        }

        try {
            val resp = client.newCall(reqBuilder.build()).execute()
            if (!resp.isSuccessful && resp.code != 206) {
                onProgress(DownloadProgress(0, 0, error = "HTTP ${resp.code}"))
                return@withContext
            }

            val contentLength = resp.body!!.contentLength()
            val totalBytes = if (existingBytes > 0 && resp.code == 206) {
                existingBytes + contentLength
            } else {
                contentLength
            }

            resp.body!!.byteStream().use { input ->
                tmpFile.outputStream().let { out ->
                    if (existingBytes > 0) tmpFile.outputStream() else tmpFile.outputStream()
                }.use { output ->
                    val buf = ByteArray(8 * 1024)
                    var downloaded = existingBytes
                    var n: Int
                    while (input.read(buf).also { n = it } != -1) {
                        output.write(buf, 0, n)
                        downloaded += n
                        onProgress(DownloadProgress(downloaded, totalBytes))
                    }
                }
            }

            tmpFile.renameTo(dest)
            onProgress(DownloadProgress(totalBytes, totalBytes, 100, done = true))

        } catch (e: Exception) {
            onProgress(DownloadProgress(0, 0, error = e.message ?: "Unknown error"))
        }
    }
}
