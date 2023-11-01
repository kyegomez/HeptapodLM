import torch
import torch.nn as nn


class LocalAttention(nn.Module):
    def __init__(self, embed_size, window_size):
        super(LocalAttention, self).__init__()
        self.window_size = window_size
        self.conv = nn.Conv2d(
            embed_size,
            embed_size,
            kernel_size=window_size * 2 + 1,
            padding=window_size,
            groups=embed_size,
        )

    def forward(self, matrix):
        # Ensure that the embedding dimension is the channel dimension for convolution
        matrix = matrix.permute(0, 3, 1, 2)

        # Apply convolution to get local context
        context = self.conv(matrix)

        # Permute the tensor back to its original shape
        context = context.permute(0, 2, 3, 1)
        return context


class TokenPredictor(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(TokenPredictor, self).__init__()
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        return self.fc(x)


class NonLinearTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, matrix_dim, window_size, iterations):
        super(NonLinearTransformer, self).__init__()
        self.iterations = iterations
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.local_attention = LocalAttention(embed_size, window_size)
        self.token_predictor = TokenPredictor(embed_size, vocab_size)

    def forward(self, matrix):
        # Initial embedding
        matrix = self.embedding(matrix)

        # Iterative refinement
        for _ in range(self.iterations):
            # Fetch local context
            context = self.local_attention(matrix)

            # Predict tokens based on local context
            tokens = self.token_predictor(context)

            # Optionally, a softmax can be applied to get a distribution over tokens,
            # and sampling or argmax can be used to update the matrix
            matrix = torch.argmax(tokens, dim=-1)

        return matrix


vocab_size = 5
embed_size = 64
matrix_dim = 5
window_size = 2
iterations = 3

model = NonLinearTransformer(
    vocab_size, embed_size, matrix_dim, window_size, iterations
)

example_1 = [
    ["🌕", "⭐", "🌑", "☁️", "🌪"],
    ["🌪", "🌑", "⭐", "🌕", "☁️"],
    ["☁️", "🌪", "🌕", "⭐", "🌑"],
    ["🌑", "☁️", "🌪", "🌕", "⭐"],
    ["⭐", "🌕", "☁️", "🌪", "🌑"],
]

example_2 = [
    ["🌕", "🌕", "🌕", "🌕", "🌕"],
    ["🌑", "🌑", "🌑", "🌑", "🌑"],
    ["⭐", "⭐", "⭐", "⭐", "⭐"],
    ["☁️", "☁️", "☁️", "☁️", "☁️"],
    ["🌪", "🌪", "🌪", "🌪", "🌪"],
]

example_3 = [
    ["⭐", "☁️", "🌑", "🌪", "🌕"],
    ["🌕", "🌪", "⭐", "☁️", "🌑"],
    ["🌑", "🌕", "☁️", "⭐", "🌪"],
    ["🌪", "⭐", "🌕", "🌑", "☁️"],
    ["☁️", "🌑", "🌪", "🌕", "⭐"],
]

rune_to_token = {"🌕": 0, "🌑": 1, "⭐": 2, "☁️": 3, "🌪": 4}

tokenized_example_1 = [[rune_to_token[rune] for rune in row] for row in example_1]
tokenized_example_2 = [[rune_to_token[rune] for rune in row] for row in example_2]
tokenized_example_3 = [[rune_to_token[rune] for rune in row] for row in example_3]

output_1 = model(torch.tensor(tokenized_example_1))
output_2 = model(torch.tensor(tokenized_example_2))
output_3 = model(torch.tensor(tokenized_example_3))

token_to_rune = {v: k for k, v in rune_to_token.items()}
decoded_output_1 = [[token_to_rune[token.item()] for token in row] for row in output_1]
decoded_output_2 = [[token_to_rune[token.item()] for token in row] for row in output_2]
decoded_output_3 = [[token_to_rune[token.item()] for token in row] for row in output_3]
