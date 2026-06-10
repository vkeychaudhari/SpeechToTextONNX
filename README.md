# Local Speech-to-Text (ASR) in WPF using Moonshine ONNX

A high-performance, fully offline Speech-to-Text (Automatic Speech Recognition) desktop application built with **C# WPF** and **ONNX Runtime**. This project integrates Useful Sensors' lightweight and optimized **Moonshine** transformer model directly into a desktop environment using `Microsoft.ML.OnnxRuntime` and `NAudio` for microphone capture.

Unlike traditional cloud-dependent ASR solutions, this application executes complete audio tokenization, acoustic encoding, and autoregressive text decoding purely on the client's local machine, guaranteeing absolute data privacy and ultra-low latency.

## 🚀 Key Features
* **100% Local Inference:** Zero external API calls, cloud subscriptions, or telemetry dependencies.
* **Asynchronous Processing Architecture:** Background worker execution using `Task.Run` combined with WPF `Dispatcher` thread marshaling ensures a fully responsive, non-blocking UI during heavy matrix calculations.
* **Dynamic Model Binding:** Automatically inspects ONNX metadata inputs (`input_values`, `use_cache_branch`) at runtime instead of hardcoding node configurations, preventing breaking changes on model version swaps.
* **Autoregressive Growing Sequence Decoding:** Implements an un-cached iterative sequencing loop that maintains speech context dynamically to prevent repeating token loops (hallucinations).
* **Real-Time Diagnostics:** Step-by-step thread tracing logs timestamped operations directly onto the UI layout.

## 🛠️ Tech Stack & Dependencies
* **Runtime Environment:** .NET 8.0 / .NET Core Framework
* **UI Architecture:** Windows Presentation Foundation (WPF)
* **Acoustic Subsystem:** `NAudio` (v2.2.1+) for real-time PCM capture and resampling
* **Inference Core Engine:** `Microsoft.ML.OnnxRuntime` (v1.18+)
* **Data Serialization:** `System.Text.Json` for vocabulary compilation

## 📂 Repository File Architecture & Setup

### 1. Download Model Artifacts
Before running the application, download the required structural weights from the official Hugging Face repository (`onnx-community/moonshine-base-ONNX`):
* `encoder_model.onnx` -> Acoustic feature map extractor.
* `decoder_model_merged.onnx` -> Language transformer text producer.
* `tokenizer.json` -> Token ID-to-Text lookup vocabulary database.

### 2. Solution Layout Setup
Place the downloaded binaries directly inside your active execution directory, or add them directly to your Visual Studio Solution Explorer under the following configuration rules:
```text
WhiteboardApp/
│
├── bin/Debug/net8.0-windows/
│   ├── encoder_model.onnx
│   ├── decoder_model_merged.onnx
│   └── tokenizer.json
