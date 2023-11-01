[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# HeptaPod Non-Linear Transformer

The HeptaPod Non-Linear Transformer is a novel deep learning architecture inspired by the linguistic capabilities of the Heptapods from the movie "Arrival". This transformer aims to generate text non-linearly in all directions simultaneously, revolutionizing the way we think about sequence generation.

## Table of Contents

- [Introduction](#introduction)
- [Architecture Overview](#architecture-overview)
  - [2D Rotary Embeddings](#2d-rotary-embeddings)
  - [Local 2D Attention](#local-2d-attention)
  - [Non-Linear Transformer Block](#non-linear-transformer-block)
- [Implementation](#implementation)
- [Usage Example](#usage-example)
- [License](#license)

## Introduction

Traditional transformers generate sequences linearly, token by token. The HeptaPod Non-Linear Transformer, however, works with 2D matrices of tokens, where each token is influenced by its neighbors in all directions. This architecture is designed to generate text resembling the Heptapod's logograms, which convey meaning non-linearly.

## Architecture Overview

The main components of the HeptaPod Non-Linear Transformer are:

### 2D Rotary Embeddings

Positional information is crucial for transformers. Unlike 1D embeddings used in traditional transformers, the HeptaPod transformer uses 2D rotary embeddings. These embeddings capture both row-wise and column-wise positional information, ensuring every token understands its position in the 2D matrix.

### Local 2D Attention

Instead of attending to all tokens in the sequence, the Local 2D Attention mechanism focuses on a localized window around each token. Each token attends only to its immediate neighbors, defined by a specified window size. This localized attention ensures that each token gathers context from its surroundings, making the generation process truly non-linear.

### Non-Linear Transformer Block

This is the core of the architecture. Each block consists of:
1. Layer normalization
2. Local 2D attention mechanism
3. A feed-forward neural network

These blocks can be stacked to deepen the architecture, allowing the model to learn more complex patterns and relationships in the data.

## Implementation

The implementation is done in PyTorch, one of the leading deep learning libraries. The design ensures modularity, allowing easy customization and experimentation.

Key features:
1. Modular design: Each component, like the Local 2D Attention mechanism, is implemented as a separate module, allowing for easy modifications and replacements.
2. Extensibility: The architecture is designed to be easily extensible. You can stack multiple Non-Linear Transformer Blocks to increase the model's depth.

## Usage

Here's a simple usage example to help you get started:

```python
# Import necessary libraries
import torch
from heptapod_transformer import NonLinearTransformer

# Initialize the model
model = NonLinearTransformer(vocab_size=10000, dim=512, depth=6, matrix_dim=5)

# Create a sample input matrix
input_matrix = torch.randint(0, 10000, (8, 5, 5))

# Pass the matrix through the model
output = model(input_matrix)

# The output is a 2D matrix with token predictions for each position
print(output.shape)  # Expected: torch.Size([8, 5, 5, 10000])
```

Remember to adjust hyperparameters like `dim`, `depth`, and `matrix_dim` as per your dataset and requirements.

# Deep Dive

## Architecture Details

### Token Representation in 2D

The representation of tokens in a 2D matrix is the foundation of the HeptaPod Non-Linear Transformer. Unlike traditional transformers that work with 1D sequences, this architecture treats input as a 2D grid. This inherently facilitates the capturing of relationships in multiple dimensions — both row-wise and column-wise.

### Hierarchical Processing

One potential advancement to this model is the introduction of hierarchical processing. After processing the entire matrix at a given resolution, the model could further abstract the matrix into larger "chunks" or "blocks", treating each chunk as a super-token. This hierarchical processing can help in capturing broader context, much like pooling layers in CNNs.

### Local vs. Global Attention

While the primary focus is on local attention, there could be merit in periodically applying global attention to capture long-range dependencies. A hybrid approach, where certain layers (or certain heads within layers) employ global attention, could offer a balance between local context and global understanding.

### Conditional Masking

Considering the non-linear nature of the text, it might be beneficial to apply conditional masks during training. Rather than always attending to the same local window, the model could be trained to decide where to look based on the token's content, allowing dynamic context windows.

## Potential Methods for Improvement

### Adaptive Window Sizes

While a fixed window size offers simplicity, an adaptive window mechanism that adjusts the size based on the token's context can capture varying degrees of local information.

### Multi-Scale Representation

Just as multi-scale feature maps are beneficial in image processing tasks, using multi-scale token representations could offer richer context. This involves processing the input matrix at different resolutions and integrating the results.

### Cross-Attention Between Hierarchies

If hierarchical processing is employed, introducing cross-attention mechanisms between different hierarchies can ensure better information flow.

### Sparse Attention Mechanisms

To efficiently capture long-range dependencies without the computational cost of global attention, sparse attention mechanisms like the ones proposed in models like the Longformer could be integrated.

## Further Work

### Integration with Vision Models

Given the 2D nature of the input, there's potential synergy with vision models. Combining the HeptaPod Non-Linear Transformer with architectures like Vision Transformers (ViTs) could yield models that excel in tasks involving both text and images.

### Transfer Learning & Pre-training

Exploring pre-training strategies on large corpora can make the HeptaPod Non-Linear Transformer more versatile. Fine-tuning on specific tasks post pre-training can lead to better performance, leveraging knowledge from vast amounts of data.

### Feedback Loops

Introducing feedback loops where the output is recursively fed back as input can help in refining the generated matrix, potentially leading to more coherent outputs.

### Custom Loss Functions

Given the non-linear generation process, custom loss functions that reward coherent formation in multiple directions can be beneficial. This would be in addition to the traditional token prediction losses.

### Token Merging Strategies

Post generation, there's potential in exploring strategies that merge or group tokens in the 2D matrix to form super-tokens, condensing information and making it more interpretable.

## Architectural Conclusion

The HeptaPod Non-Linear Transformer represents a paradigm shift in sequence generation. While the foundation is promising, the architecture offers numerous avenues for exploration, innovation, and improvement. As with any novel approach, iterative research, experimentation, and collaboration with the broader research community will be pivotal in realizing its full potential.

### License

This project is licensed under the MIT License. This ensures that the HeptaPod Non-Linear Transformer is free for all to use, modify, and distribute. We believe in open-source and encourage innovations and improvements to the concept.



