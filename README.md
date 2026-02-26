# LLM Engineering Journey

> From mathematical foundations to production-grade AI systems — a deep technical research archive, not a tutorial.

---

## Overview

This repository documents a systematic, engineering-first exploration of Large Language Models. The goal is genuine understanding: how they learn, why they fail, how they scale, and how to deploy them responsibly.

**What's inside:**
- Transformer architecture built from scratch in PyTorch
- Custom training pipelines, optimization experiments, and scaling studies
- LangChain & LangGraph application development
- Hugging Face fine-tuning with LoRA / QLoRA
- Retrieval-Augmented Generation (RAG) systems end-to-end

---

## Repository Structure

```
LLM-Journey/
├── 01_Build_LLM_From_Scratch/     # Transformer, attention, training loop
├── 02_Refine_LLM_From_Scratch/    # RoPE, FlashAttention, RMSNorm, KV cache
├── 03_LangChain/                  # Chains, agents, memory, tool calling
├── 04_LangGraph/                  # Stateful multi-agent workflows
├── 05_HuggingFace/                # Fine-tuning, PEFT, quantization, deployment
├── 06_RAG/                        # Embeddings, vector DBs, retrieval strategies
├── experiments/                   # Hyperparameter sweeps, ablations
├── research_notes/                # Paper summaries, conceptual notes
├── datasets/                      # Preprocessing and dataset utilities
└── evaluation/                    # Metrics, benchmarking, hallucination analysis
```

---

## Modules

### 1 · Building an LLM from Scratch

Full transformer implementation using raw PyTorch — no high-level APIs.

**Mathematical foundations:** linear algebra, probability theory, information theory (entropy, KL divergence), optimization.

**Architecture:** scaled dot-product attention → multi-head attention → causal masking → positional encoding (sinusoidal & learned) → encoder-only, decoder-only, and encoder-decoder variants.

**Training pipeline:** BPE tokenization, vocabulary construction, batching, padding/masking, next-token prediction objective.

---

### 2 · Refining the LLM

Improving performance and efficiency on the baseline model.

- **Training:** mixed precision (FP16), gradient accumulation, gradient clipping, cosine LR schedule with warmup
- **Architecture:** RMSNorm, Rotary Positional Embeddings (RoPE), SwiGLU activation, KV caching, Flash Attention (conceptual study)
- **Regularization:** dropout tuning, label smoothing, weight decay, early stopping
- **Scaling studies:** parameter count vs. performance, dataset size, compute tradeoffs

---

### 3 · LangChain

Modular LLM application development.

Core: LLM wrappers, prompt templates, chains, memory, output parsers, tool/function calling, structured output.

Applications built: chatbot with persistent memory, document QA system, API-connected agent, multi-tool reasoning agent.

---

### 4 · LangGraph

Stateful, multi-step AI workflow orchestration.

Topics: graph-based execution, conditional branching, retry mechanisms, human-in-the-loop integration, multi-agent collaboration.

Implementations: multi-agent research assistant, tool-using planner agent, decision-tree LLM workflow.

---

### 5 · Hugging Face Ecosystem

Production-grade LLM tooling and fine-tuning.

- **Fine-tuning:** full fine-tuning, LoRA, QLoRA, PEFT methods
- **Deployment:** inference pipelines, model quantization, ONNX export, TorchScript, CPU vs. GPU benchmarking
- **Data:** dataset loading, preprocessing, streaming

---

### 6 · Retrieval-Augmented Generation (RAG)

Combining retrieval systems with LLMs for grounded, factual responses.

```
User Query → Embedding Model → Vector DB → Top-k Retrieval → Context Augmentation → LLM Response
```

**Embedding models:** Sentence Transformers, open-source alternatives.

**Vector databases:** FAISS, ChromaDB, Pinecone (conceptual).

**Retrieval strategies:** similarity search, hybrid search, Maximal Marginal Relevance (MMR).

**Evaluation:** retrieval recall, context relevance, answer faithfulness, hallucination analysis.

---

## Evaluation Framework

| Metric | Purpose |
|--------|---------|
| Perplexity | Language modeling quality |
| BLEU / ROUGE | Text similarity & summarization |
| Exact Match / F1 | QA and retrieval evaluation |
| Latency | Inference performance |
| GPU Memory | Efficiency and cost |

---

## Tech Stack

PyTorch · Hugging Face Transformers · LangChain · LangGraph · FAISS · ChromaDB · NumPy · Matplotlib · Weights & Biases

---

## Key Learnings

- Attention is a learned, weighted information routing mechanism — not magic.
- Scaling laws matter more than architectural novelty in most practical settings.
- Retrieval significantly reduces hallucination; it is not optional for factual applications.
- Fine-tuning is highly data-sensitive and expensive to do well.
- Prompt engineering has a ceiling that architecture and training do not.

---

## Roadmap

- [ ] RLHF implementation
- [ ] Direct Preference Optimization (DPO)
- [ ] Domain adaptation experiments
- [ ] Multimodal / Vision-Language models
- [ ] Quantized inference on edge devices

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2](https://openai.com/research/language-unsupervised) / [GPT-3](https://arxiv.org/abs/2005.14165)
- [LLaMA](https://arxiv.org/abs/2302.13971)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [RAG (Lewis et al., Meta)](https://arxiv.org/abs/2005.11401)
- [LoRA](https://arxiv.org/abs/2106.09685)

---

*This repository is a personal research archive. Discussions and ideas are welcome.*
