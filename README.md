# Local LLM — Offline AI on Android

Run **Llama 3.1 8B Instruct** (or any GGUF model) fully on-device on your Android phone, with no internet required after the first model download.

## Features

- 🦙 **llama.cpp** inference engine, compiled natively for ARM64
- ⚡ **Vulkan GPU acceleration** enabled by default (falls back to CPU automatically)
- 📥 **Auto-downloads** the model from HuggingFace on first launch (~4.7 GB)
- ⏸ **Resume downloads** — picks up where it left off if interrupted
- 💬 Clean dark chat UI with streaming token output
- 🛑 Stop generation mid-response

## Requirements

- Android 9.0+ (API 28)
- ARM64 processor (Snapdragon, MediaTek, etc.)
- ~6–8 GB free RAM (for 8B model)
- ~5 GB storage for model file
- Internet for the first model download

## Build (GitHub Actions)

1. Fork / push this repo to GitHub
2. Go to **Actions → Build LocalLLM APK → Run workflow**
3. Wait ~20–40 minutes for the build to complete
4. Download the APK from the **Artifacts** section
5. Install on your phone (enable "Install from unknown sources")

The workflow automatically:
- Clones the latest `llama.cpp`
- Compiles native libs with the Android NDK (arm64-v8a)
- Enables Vulkan if the NDK finds it
- Packages everything into a single APK

## Changing the model

Edit `ModelDownloader.kt`:

```kotlin
const val DEFAULT_MODEL_NAME = "your-model.gguf"
const val DEFAULT_MODEL_URL  = "https://huggingface.co/repo/resolve/main/your-model.gguf"
```

Any GGUF-format model will work. Tested models:
| Model | Size (Q4_K_M) | Speed (Snapdragon 8 Gen 2) |
|-------|--------------|---------------------------|
| Llama 3.1 8B Instruct | 4.7 GB | ~8–12 tok/s |
| Mistral 7B Instruct v0.3 | 4.1 GB | ~10–14 tok/s |
| Llama 3.2 3B Instruct | 2.0 GB | ~20–28 tok/s |

## Prompt format

The app uses the **Llama 3 Instruct** format by default. If you switch to Mistral,
update `buildPrompt()` in `LlamaEngine.kt` to use `[INST]...[/INST]` format.

## GPU layers

By default, the app tries to offload **all layers** to Vulkan GPU (`nGpuLayers = 99`).
If your phone doesn't have enough VRAM, lower it in `LlamaEngine.kt`:

```kotlin
engine.loadModel(modelPath, nGpuLayers = 20)  // partial GPU offload
```

## License

Apache 2.0 — same as llama.cpp.
