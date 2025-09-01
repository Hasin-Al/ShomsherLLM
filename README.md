# 🐉 ShomsherLLM

ShomsherLLM is a custom-built **Large Language Model (LLM)** designed with advanced architectural optimizations for **efficient training** and **scalable inference**.  

It combines **Decoupled Rotary Positional Encoding (DeRoPE)**, **KV Cache Compression**, and a **Mixture of Experts (MoE)** feedforward layer with **hybrid normalization strategies** for robust performance.  

---

## ✨ Features

- 📐 **Decoupled Rotary Positional Encoding (DeRoPE)**  
  Improves positional awareness and stability for long-context sequences.  

- ⚡ **KV Cache Compression**  
  Enables fast autoregressive decoding while reducing memory usage.  

- 🔄 **Hybrid Normalization**  
  - **Pre-Norm:** Layer Normalization (using variance & standard deviation).  
  - **Post-Norm:** RMS Normalization (RMSNorm).  

- 🧩 **Mixture of Experts (MoE) Feedforward**  
  Increases model capacity without linear parameter growth.  

- 🔤 **BPE Tokenizer**  
  Efficient subword tokenization, supports multilingual datasets.  

---

## 📂 Repository Structure

```bash
├── main.py             # Entry point for training/inference
├── train.py            # Training loop
├── model.py            # Core ShomsherLLM model
├── DeRoPE.py            # Decoupled Rotatry Positional encoding class
├── MultiHeadLatentAttention.py            # KV caching attetniton class
├── TransformerBlock.py            # Transformer class
├── TrainTokenizer.py            # To train our BPE tokenizer
├── TextGeneration.py            # To generate text using our trained model
├── neededclass.py      # Supporting classes:
│                       #    - MoE feedforward
│                       #    - Normalization layers
│                       #    - SwiGLU
│                       #  
├── DataClean.py        # Data cleaning & preprocessing
├── tokenizer/          # BPE tokenizer files
│   └── bpe_tokenizer.json
├── models/             # Saved checkpoints
│   └── model_latest.pt
├── ShamsherLLM.ipynb            # Core ShomsherLLM notebook
└── README.md           # Project documentation

