[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# HeptaPod Transformer: Non-linear Text Generation

The HeptaPod Transformer is inspired by the linguistic wonders of the Heptapods from the movie "Arrival". This architecture is designed to challenge traditional sequence-to-sequence generation by predicting text in a non-linear fashion, much like the intricate logograms used by the Heptapods.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
  - [Problem Definition](#problem-definition)
  - [Approach](#approach)
- [Implementation](#implementation)
- [Usage](#usage)
- [License](#license)

## Introduction

Traditionally, Transformers generate sequences in a linear manner, token after token. The HeptaPod Transformer, however, introduces a new paradigm where each token in a 2D matrix is influenced not just by its preceding tokens, but by all its neighbors, thereby allowing for generation in all directions at once.

## Architecture

### Problem Definition

The core challenge lies in modifying the understanding of sequence generation. To generate tokens in a non-linear fashion, we envision a 2D matrix where each position is influenced by tokens from all directions.

### Approach

1. **Iterative Refinement**: Begin with a matrix filled with a special token, like `[START]`, or a combination of seed tokens and uninitialized positions. This matrix undergoes iterative refinement, where each position updates based on its neighbors.

2. **Local Attention Mechanism**: Traditional transformers use a global attention mechanism. In contrast, the HeptaPod Transformer employs a local attention mechanism, wherein each token only attends to its immediate neighbors.

3. **Token Generation**: Post the attention phase, every position in the matrix updates its token based on the gathered context.

### Detailed PyTorch Implementation

Harnessing the local attention methodology, the HeptaPod Transformer fetches the local context for each token and predicts the subsequent token based on this context. Given the spatial essence of the problem, convolutional layers are intertwined with transformer layers to facilitate the generation.

## Implementation

The implementation revolves around two primary modules: `LocalAttention` and `NonLinearTransformer`.

`LocalAttention` focuses on capturing the local context of each token using convolutional layers. This local context is then fed into the `NonLinearTransformer` which, through iterative refinement, predicts the token for each position.

Here's a basic pseudocode of the architecture:

```pseudocode
FUNCTION LOCAL_ATTENTION(matrix):
    context = APPLY_CONVOLUTION(matrix)
    RETURN context

FUNCTION ITERATIVE_REFINEMENT(matrix, iterations):
    FOR i IN range(iterations):
        FOR position IN matrix:
            matrix[position] = LOCAL_ATTENTION(matrix, position)
    RETURN matrix

CLASS NonLinearTransformer:
    FUNCTION forward(matrix):
        matrix = EMBED(matrix)
        matrix = ITERATIVE_REFINEMENT(matrix, iterations)
        RETURN matrix
```

## Usage

Using the HeptaPod Transformer begins with initializing the model with the appropriate parameters. The model takes in a matrix of tokens and returns the refined matrix after the specified iterations.

Here's a simple usage example:

```python
# Initialize the model
model = NonLinearTransformer(vocab_size, embed_size, matrix_dim, window_size, iterations)

# Sample input matrix
input_matrix = ...  # Your matrix of tokens

# Generate refined matrix
output_matrix = model(input_matrix)
```

It's crucial to note that the HeptaPod Transformer, much like the Heptapod language, is a novel and experimental approach. Extensive training, fine-tuning, and experimentation are required to derive meaningful results.




# License
MIT



