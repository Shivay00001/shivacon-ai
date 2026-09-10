# Shivacon Ai

Shivacon AI - Production-Grade Multi-Modal Agent Framework with Fine-Tuning, Document Generation, and Benchmarking. Self-hostable. Commercial license available.

![Language](https://img.shields.io/badge/Language-Python-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-Non--Commercial-red)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

> **Name Origin:** SHIV = Bhagwan Shankar (Divine Consciousness) + CON = Connected (Networked Intelligence) → **SHIVACON = "Divine Consciousness AI"**
>
> **Mission:** Build AI that combines consciousness with intelligence.

## 🚀 Overview

Welcome to the **Shivacon Ai** repository. This project is built to deliver a robust and scalable solution tailored to modern development standards.

Shivacon is a comprehensive AI framework that combines:
- **Multi-modal learning** — Text, Image, Audio, Video, and Music encoders fused into a unified embedding space
- **Autonomous agent capabilities** — Tool calling, planning, memory, guardrails, and multi-agent collaboration
- **Fine-tuning support** — Full parameter training, LoRA, and layer freezing
- **Memory systems** — Short-term (conversation) and long-term (persistent) memory
- **Guardrails & safety** — Content filtering and sandboxed code execution
- **Benchmarking suite** — Built-in latency, throughput, and capability rating tools

## ✨ Features

- **High Performance:** Optimized for speed and efficiency (~3.7ms text encoding latency, ~2,585 samples/sec at batch=8).
- **Scalable Architecture:** Designed to grow with your needs.
- **Clean Codebase:** Follows best practices and industry standards.
- **Secure by Default:** Engineered with security in mind (guardrails, sandboxed code interpreter).

### Model Capabilities
- Text Encoding (Transformer), Image Encoding (ViT), Audio Encoding (CNN + Transformer)
- Video Encoding (Spatial-Temporal attention), Music Generation (Autoregressive)
- Multi-Modal Fusion with Cross-Attention and Similarity Search

### Agent Capabilities
- Tool Calling (12+ tools), Short/Long-term Memory, Planning/Reasoning
- Code Interpreter, File Operations, Guardrails, Multi-Agent Collaboration
- Streaming Responses, Observability/Tracing

### Training Capabilities
- Full Parameter Fine-tuning, **LoRA** (rank 8–64), Layer Freezing
- Checkpointing, LR Scheduling, Mixed Precision (FP16), Gradient Accumulation

## 🏗️ Architecture / How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      SHIVACON AI                            │
├─────────────────────────────────────────────────────────────┤
│  modalities/          core/               agent/            │
│  ├─ TextEncoder       ├─ MultiModalCore   ├─ Tool Registry  │
│  ├─ ImageEncoder(ViT) ├─ Cross-Modal Attn ├─ Memory (ST/LT) │
│  ├─ AudioEncoder      ├─ Unified Latent   ├─ Planning       │
│  ├─ VideoEncoder      │   Embedding Space ├─ Guardrails     │
│  └─ MusicEncoder      └─ Projection       └─ Multi-Agent    │
│                                                             │
│  training/                  inference/                      │
│  ├─ FineTuner (Full/LoRA/   ├─ MultiModalInference          │
│  │   Freeze)                ├─ Batch Processing             │
│  ├─ Checkpointing           ├─ Similarity Search            │
│  └─ LR Scheduling           └─ Music Generation             │
└─────────────────────────────────────────────────────────────┘
```

**Pipeline:**
1. **Encoders (`modalities/`)** — Each modality has a dedicated encoder producing embeddings (default `d_model=128`).
2. **Fusion (`core/multimodal_core.py`)** — `MultiModalCore` projects all modality embeddings into a unified latent space using cross-modal attention layers. Encoders are registered dynamically via `model.register_encoder(...)`.
3. **Agent (`agent/agent_v2.py`)** — `OmniCoreAgent` wraps the model with a tool registry, short-term memory (10 turns), persistent long-term memory, guardrails, and a plan-execute loop. `MultiAgentTeam` enables collaborative multi-agent runs.
4. **Training (`training/finetune.py`)** — `FineTuner` supports three modes: `full`, `lora` (injects low-rank adapter matrices, typically <5% trainable params), and `freeze`.
5. **Inference (`inference/engine.py`)** — `MultiModalInference` loads checkpoints and exposes `encode`, `batch_encode`, `get_similarity`, and `generate_music`.

**Data flow:** JSONL manifests (see `download_data.py` / `generate_synthetic_data.py`) → `MultiModalDataset` → BPE tokenizer (`.cache/tokenizer.json`) → training loop → PyTorch checkpoints (`checkpoints/*.pt`) → inference engine / agent.

## 🛠️ Prerequisites

- Python 3.10+
- PyTorch 2.1+ (CPU works; CUDA optional)
- ~20GB disk space if downloading COCO; much less for synthetic data
- Docker (optional, for containerized deployment)

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shivay00001/shivacon-ai.git
   ```
2. Navigate to the project directory:
   ```bash
   cd shivacon-ai
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🐳 Running with Docker

The repository ships with a `Dockerfile` and `docker-compose.yml` for portable deployment on any laptop or server.

### Option A: Docker Compose (recommended)

```bash
docker-compose up --build
```

This builds the image, starts the app, and maps host port **8104** → container port **8000**. The container runs `python main.py` with `PYTHONUNBUFFERED=1` for live logs and restarts automatically unless stopped.

To stop:
```bash
docker-compose down
```

### Option B: Plain Docker

```bash
# Build the image
docker build -t shivacon-ai .

# Run the container (map port 8000)
docker run -p 8000:8000 shivacon-ai
```

The image is based on `python:3.10-slim`, installs `requirements.txt`, copies the full source tree, and exposes port 8000.

> **Note:** If `main.py` is not present in your checkout, override the command to run a script that exists, e.g.:
> ```bash
> docker run shivacon-ai python full_benchmark.py
> ```

## 💻 Usage

### Generate training data (synthetic, fast)
```bash
python generate_synthetic_data.py --samples 500
```

### Download real datasets (COCO, LibriSpeech)
```bash
python download_data.py --datasets coco librispeech --samples 1000
# or everything:
python download_data.py --all
```

### Fine-tune with LoRA
```python
from training.finetune import FineTuner, FineTuneConfig

config = FineTuneConfig(
    mode="lora",           # "full" | "lora" | "freeze"
    learning_rate=1e-4,
    max_steps=1000,
    batch_size=4,
    lora_rank=8,
    lora_alpha=16,
)
tuner = FineTuner(model, tokenizer, config)
train_loader, val_loader = tuner.prepare_data("data/train.jsonl", "data/val.jsonl")
tuner.train(train_loader, val_loader)
tuner.save_checkpoint("final")
tuner.merge_lora()  # for inference
```

### Run inference / similarity
```python
from inference.engine import MultiModalInference
from data.tokenizer import BPETokenizer

engine = MultiModalInference.from_checkpoint("checkpoints/checkpoint_epoch0001_step00000046.pt", device="cpu")
tokenizer = BPETokenizer.load(".cache/tokenizer.json")
similarity = engine.get_similarity(inputs_a, inputs_b)
```

### Run the agent
```python
from agent.agent_v2 import OmniCoreAgent, AgentConfig

agent = OmniCoreAgent(model, tokenizer, AgentConfig(name="shivacon"))
result = agent.run("Calculate 10 + 20")
response = agent.chat("Hello!")
```

### Benchmarks & tests
```bash
python full_benchmark.py          # Full capability + rating report
python benchmark_finetuning.py    # LoRA throughput benchmark
python run_benchmark.py           # Real-world agent task benchmark
python test_agent.py              # Agent capability test suite
python docs.py                    # Print full embedded documentation
```

## ⚙️ Configuration

Configuration is managed via `config/settings.py` (with YAML support) and environment variables:

```yaml
text:
  vocab_size: 1000
  max_seq_len: 128
  d_model: 128
training:
  num_epochs: 50
  learning_rate: 0.001
  batch_size: 4
```

Environment overrides: `TEXT_VOCAB_SIZE`, `TEXT_MAX_SEQ_LEN`, `TRAIN_EPOCHS`, `TRAIN_BATCH_SIZE`, `TRAIN_LR`. Secrets go in `.env` (gitignored).

## 📊 Benchmarks (self-reported)

| Metric | Value |
|---|---|
| Text Encoding Latency | ~3.7 ms |
| Full Inference Latency | ~4.0 ms |
| Throughput (batch=8) | ~2,585 samples/sec |
| Model Size | ~26.6 MB |
| Parameters | ~2,145,472 |
| Memory Usage | ~310 MB |

## 🔍 Workability Assessment

Honest evaluation of the current state of this repository:

**What works / is promising:**
- The overall architecture is sound and well-organized: clean separation of modalities, fusion, agent, training, and inference.
- LoRA fine-tuning, checkpointing, memory systems, tool registry, and guardrails are genuinely implemented (not stubs), as evidenced by `benchmark_finetuning.py`, `test_agent.py`, and `full_benchmark.py`.
- Data tooling is pragmatic: real dataset downloaders (COCO, LibriSpeech) with automatic fallback to synthetic data, plus a BPE tokenizer trainer.
- Docker support exists and is correctly wired for a Python 3.10 app.

**Gaps and risks (must be addressed before production):**
- **Incomplete source in this snapshot:** Key modules referenced everywhere (`main.py`, `train.py`, `core/`, `agent/`, `training/`, `inference/`, `data/`, `config/`) are not shown here, and the Docker `CMD ["python", "main.py"]` will fail if `main.py` is absent. The FastAPI server dependency exists in `requirements.txt` but no server entrypoint is visible.
- **Not actually production-scale:** The model is ~2.1M parameters with a tiny vocab (1000–8000). This is a research/demo-scale model, not a competitive LLM. The docs themselves acknowledge "No LLM integration" and "Small model (2M params)."
- **Self-graded ratings:** The 7.2/10 scores in `docs.py`/`full_benchmark.py` are self-assigned, not independently validated. Treat them as aspirational.
- **Synthetic data is random noise:** Video/music "synthetic" samples are random tensors/tokens — fine for pipeline testing, useless for learning real representations.
- **License mismatch:** The old README badge said MIT, but `LICENSE` is a **custom Non-Commercial license** — commercial use requires a paid license from the authors. The badge has been corrected here.
- **Hardcoded checkpoint paths** (e.g., `checkpoint_epoch0001_step00000046.pt`) mean tests/benchmarks fail on a fresh clone without training first.

**Verdict:** A solid, well-structured **prototype/research framework** with real fine-tuning and agent infrastructure. It is **not production-ready** as-is: it needs the missing entrypoints verified, a real (larger) backbone or LLM integration, real training data, and independent evaluation. Good foundation for experimentation and education; significant work remains for commercial deployment.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

Priority areas: LLM backbone integration, scaling model size, more tools, RAG support, and expanded examples.

## 📝 License

This project is licensed under the **Shivacon AI Non-Commercial License v1.0** — free for non-profit and educational use only. Commercial use (selling, paid services, for-profit internal use) requires a separate commercial license.

For commercial licensing inquiries, contact **Shivay00001** or **@visionquantech**. See [LICENSE](LICENSE) for full terms.