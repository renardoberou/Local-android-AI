package com.localllm.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class AppState {
    object Idle        : AppState()
    data class Downloading(val progress: DownloadProgress) : AppState()
    object Loading     : AppState()
    object Ready       : AppState()
    object Generating  : AppState()
    data class Error(val msg: String) : AppState()
}

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = LlamaEngine()

    private val _state    = MutableStateFlow<AppState>(AppState.Idle)
    val state: StateFlow<AppState> = _state.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    val systemPrompt = "You are a helpful assistant running locally on an Android device."

    fun initModel() {
        val ctx = getApplication<Application>()
        viewModelScope.launch {
            if (!ModelDownloader.isModelDownloaded(ctx)) {
                _state.value = AppState.Downloading(DownloadProgress(0, 0))
                ModelDownloader.download(ctx) { progress ->
                    _state.value = AppState.Downloading(progress)
                    if (progress.error != null) {
                        _state.value = AppState.Error("Download failed: ${progress.error}")
                        return@download
                    }
                }
                if (_state.value is AppState.Error) return@launch
            }

            _state.value = AppState.Loading
            val modelPath = ModelDownloader.modelFile(ctx).absolutePath
            val ok = engine.loadModel(modelPath)
            _state.value = if (ok) AppState.Ready
                           else AppState.Error("Failed to load model — is there enough RAM?")
        }
    }

    fun send(userText: String) {
        if (_state.value != AppState.Ready) return

        val history = _messages.value.toMutableList()
        history.add(ChatMessage(userText, isUser = true))
        // Add a streaming placeholder for the assistant
        history.add(ChatMessage("", isUser = false, isStreaming = true))
        _messages.value = history

        _state.value = AppState.Generating

        viewModelScope.launch {
            val prompt = engine.buildPrompt(
                messages = history.dropLast(1),   // exclude the empty placeholder
                systemPrompt = systemPrompt
            )

            val sb = StringBuilder()
            engine.completion(prompt, nPredict = 768) { token ->
                sb.append(token)
                val updated = _messages.value.toMutableList()
                updated[updated.lastIndex] = ChatMessage(sb.toString(), isUser = false, isStreaming = true)
                _messages.value = updated
            }

            // Mark streaming done
            val done = _messages.value.toMutableList()
            done[done.lastIndex] = ChatMessage(sb.toString(), isUser = false, isStreaming = false)
            _messages.value = done
            _state.value = AppState.Ready
        }
    }

    fun stopGeneration() {
        engine.stop()
    }

    fun clearChat() {
        _messages.value = emptyList()
    }

    fun modelInfo() = engine.modelInfo()

    override fun onCleared() {
        super.onCleared()
        engine.free()
    }
}
