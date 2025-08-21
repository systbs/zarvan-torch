# Zarvan: A Hybrid MoE Architecture for Advanced Sequence Modeling

[![PyPI Version](https://img.shields.io/pypi/v/zarvan-torch.svg)](https://pypi.org/project/zarvan-torch/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/systbs/zarvan-torch/main.yml?branch=main)](https://github.com/systbs/zarvan-torch/actions)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

[**📄 Read the Paper (Zarvan v2)**](https://github.com/systbs/zarvan-v2) | [**GitHub Repository**](https://github.com/systbs/zarvan-torch)

**Zarvan** is an advanced neural network architecture designed to overcome the fundamental limitations of Transformers and RNNs. By unifying the strengths of parallel processing and stateful reasoning, Zarvan provides a powerful, scalable solution for the next generation of sequence modeling challenges.

Operating with **linear time complexity (O(S))**, Zarvan v2 has demonstrated its superior capabilities across 18 distinct benchmarks, ranging from algorithmic reasoning to real-world applications in vision and audio.

---

## 🚀 Key Features

* **Hybrid Mixture-of-Experts (MoE) Architecture**: Zarvan v2 employs an intelligent MoE system that dynamically chooses between three "experts" to process a sequence: two for global pattern recognition and a dedicated state machine for step-by-step reasoning.
* **Linear Time Complexity (O(S))**: By replacing the quadratic (O(S^2)) self-attention mechanism, Zarvan is significantly more efficient and ideal for processing ultra-long sequences found in high-resolution images, long-form documents, and audio signals.
* **🧠 Stateful Sequential Reasoning**: The core innovation of Zarvan v2 is the **Sequential Extractor**, a deterministic mathematical state machine that maintains a perfect, non-decaying memory of the sequence history. This allows the model to solve complex, path-dependent tasks where Transformers fail.
* **⚡️ Superior Long-Range Memory**: On classic long-range memory benchmarks like **Selective Copying**, Zarvan achieves near-perfect performance, overcoming the "catastrophic forgetting" problem that plagues LSTMs.
* **🌐 Versatility Across Domains**: Beyond reasoning tasks, Zarvan is a versatile general-purpose processor, demonstrating highly competitive performance in vision (MNIST, CIFAR-10) and audio (Google Speech Commands) domains.
* **🤗 Hugging Face Ecosystem Compatibility**: The package is fully compatible with the Hugging Face ecosystem, enabling seamless integration with popular tools like `Trainer` and `pipeline`.

---

## 🏛️ The Zarvan v2 Architecture

The core of Zarvan is a stack of identical blocks. Each block is a Mixture-of-Experts model that dynamically combines the outputs of three specialist modules via a learned gating network.



1.  **Holistic Extractor**: This expert captures the "gist" or overall summary of the sequence by computing a multi-head weighted average of all tokens, providing a global context vector.
2.  **Associative Extractor**: This module acts as a "focused memory" retriever. By identifying and aggregating information from the most salient tokens, it allows the model to focus on sparse but critical information within the sequence.
3.  **Sequential Extractor (The State Machine)**: This is the central innovation of Zarvan v2. It functions as a parallelized state machine that tracks the sequence history losslessly. It operates via two key mechanisms:
    * **Gated Accumulation**: Uses a cumulative sum (`cumsum`) to recurrently aggregate important information in a parallelizable manner.
    * **Phase Representation**: Represents the accumulated state as a point on a multi-dimensional unit circle using `sin` and `cos`. This representation is robust and ideal for modeling periodic or cyclic phenomena (e.g., binary state flips).

These three experts feed into an **Expert Gate**, which learns to decide, on a per-token basis, whether to rely on global patterns, focused memory, or sequential history to form its output.

---

## 📊 Performance Highlights

Empirical results unequivocally demonstrate Zarvan v2's advantages in key areas:

* **Reasoning & State Tracking**: On tasks like **Dynamic Pathfinding**, where the standard Transformer fails with <40% accuracy, Zarvan v2 successfully solves the task with **99.10%** accuracy.
* **Long-Range Memory**: On the **Selective Copying** task, LSTMs fail due to the vanishing gradient problem (9.91% accuracy), while Zarvan v2 achieves **99.69%** accuracy, proving its robust, non-decaying memory.
* **Computational Efficiency**: Training time benchmarks clearly show that Zarvan's runtime scales **linearly** with sequence length, whereas the Transformer's runtime grows **quadratically**, making it prohibitively slow for long sequences.

---

## 🚀 Installation

Install the package directly from GitHub:

```bash
pip install git+[https://github.com/systbs/zarvan-torch.git](https://github.com/systbs/zarvan-torch.git)
```

Or after cloning the repository locally:
```bash
git clone [https://github.com/systbs/zarvan-torch.git](https://github.com/systbs/zarvan-torch.git)
cd zarvan-torch
pip install .
```

## ✨ Quick Start

Using Zarvan is straightforward, especially if you are familiar with the Hugging Face `transformers` library.

```python
import torch
from zarvan import Zarvan, ZarvanConfig

# 1. Define the model configuration
config = ZarvanConfig(
    vocab_size=10000,
    embed_dim=256,
    hidden_dim=1024,
    num_heads=4,
    num_layers=6,
    num_classes=2, # for a binary classification task
)

# 2. Instantiate the model
model = Zarvan(config)

# 3. Create some dummy input
input_ids = torch.randint(0, 10000, (2, 50)) # Batch size 2, sequence length 50

# 4. Get model outputs
outputs = model(input_ids)
logits = outputs.logits

print("Logits shape:", logits.shape)
# Expected output: torch.Size([2, 50, 2])
```

##  архитектура (Architecture)

The core of Zarvan is the `_ZarvanBlock`, which contains three parallel context extractors:
- **`_HolisticExtractor`**: Captures global, attention-like relationships across the sequence.
- **`_AssociativeExtractor`**: A simpler, non-multi-head mechanism for weighted feature aggregation.
- **`_SequentialExtractor`**: Processes information based on its order and position in the sequence.

The outputs of these "experts" are dynamically combined using a learned gating mechanism.

---

