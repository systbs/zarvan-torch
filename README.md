# Zarvan: A Hybrid MoE Architecture for Advanced Sequence Modeling

[![PyPI Version](https://img.shields.io/pypi/v/zarvan.svg)](https://pypi.org/project/zarvan/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/systbs/zarvan-torch/main.yml?branch=main)](https://github.com/systbs/zarvan-torch/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)


**Zarvan** is an advanced neural network architecture designed to overcome the fundamental limitations of Transformers and RNNs. By unifying the strengths of parallel processing and stateful reasoning, Zarvan provides a powerful, scalable solution for the next generation of sequence modeling challenges.

This library is built on pure PyTorch, offering a lightweight, independent, and high-performance implementation of the Zarvan architecture.

---

## 🚀 Key Features

* **Hybrid Mixture-of-Experts (MoE) Architecture**: Employs an intelligent MoE system that dynamically chooses between three "experts" to process a sequence: two for global pattern recognition and a dedicated state machine for step-by-step reasoning.
* **Linear Time Complexity ($O(S)$)**: By replacing the quadratic ($O(S^2)$) self-attention mechanism, Zarvan is significantly more efficient and ideal for processing ultra-long sequences.
* **🧠 Stateful Sequential Reasoning**: Features the **Sequential Extractor**, a deterministic state machine that maintains a perfect, non-decaying memory of the sequence history, enabling it to solve complex, path-dependent tasks where Transformers fail.
* **⚡ Lightweight & Independent**: Built on pure PyTorch with **zero external dependencies** beyond `torch`, ensuring easy integration, maximum flexibility, and no version conflicts.

---

## 🏛️ Architecture Overview

The core of Zarvan is a stack of identical blocks. Each block is a Mixture-of-Experts model that dynamically combines the outputs of three specialist modules via a learned gating network.

1.  **Holistic Extractor**: Captures the "gist" or overall summary of the sequence.
2.  **Associative Extractor**: Acts as a "focused memory" retriever for salient, sparse information.
3.  **Sequential Extractor (The State Machine)**: Functions as a parallelized state machine that tracks the sequence history losslessly using gated accumulation and phase representation.

An **Expert Gate** then learns to weigh the outputs of these three modules for each token, allowing the model to adapt its strategy based on the input.

---

## 🚀 Installation

Install the package directly from PyPI:

```bash
pip install zarvan
```

Or after cloning the repository locally:
```bash
git clone [https://github.com/systbs/zarvan-torch.git](https://github.com/systbs/zarvan-torch.git)
cd zarvan-torch
pip install .
```

## ✨ Quick Start

Using the independent zarvan library is clean and simple.

```python
import torch
from zarvan import Zarvan, ZarvanConfig

# 1. Define the model configuration
# The ZarvanConfig object holds all architectural hyperparameters.
config = ZarvanConfig(
    vocab_size=10000,
    embed_dim=256,
    hidden_dim=1024,
    num_heads=4,
    num_layers=6,
    num_classes=2, # For a binary classification task
    max_len=128
)

# 2. Instantiate the model from the configuration
model = Zarvan(config)
model.eval() # Set to evaluation mode

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model created successfully with {num_params / 1e6:.2f}M parameters.")

# 3. Create dummy input data
input_ids = torch.randint(0, config.vocab_size, (2, 50)) # (Batch, Sequence Length)

# 4. Perform a forward pass
# The output is a simple torch.Tensor containing the logits.
with torch.no_grad():
    logits = model(input_ids)

print("\n--- I/O Shapes ---")
print("Input IDs shape:", input_ids.shape)
print("Logits shape:", logits.shape)
# Expected Logits shape: torch.Size([2, 50, 2])

# 5. Save and load the model using the built-in methods
save_directory = "./saved_zarvan_model"
model.save_pretrained(save_directory)

loaded_model = Zarvan.from_pretrained(save_directory)

# Verify that the loaded model works and produces the same output
with torch.no_grad():
    loaded_logits = loaded_model(input_ids)

assert torch.allclose(logits, loaded_logits, atol=1e-5)
print("\nSaved and loaded model outputs match. ✅")
```


