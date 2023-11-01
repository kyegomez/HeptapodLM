import torch
import torch.nn as nn


class LocalAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size):
        super(LocalAttention, self).__init__()
        self.embed_size = embed_size
        self.window_size = window_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, matrix, position):
        Q = self.query(matrix)
        K = self.key(matrix)
        V = self.value(matrix)

        # Extract a local window around the current position
        x, y = position
        local_Q = Q[
            x - self.window_size : x + self.window_size + 1,
            y - self.window_size : y + self.window_size + 1,
        ]
        local_K = K[
            x - self.window_size : x + self.window_size + 1,
            y - self.window_size : y + self.window_size + 1,
        ]
        local_V = V[
            x - self.window_size : x + self.window_size + 1,
            y - self.window_size : y + self.window_size + 1,
        ]

        attention = torch.matmul(local_Q, local_K.transpose(-2, -1)) / (
            self.embed_size**0.5
        )
        attention = torch.nn.functional.softmax(attention, dim=-1)
        out = torch.matmul(attention, local_V)
        out = self.fc_out(out)

        # Index the output to match the shape of the target position
        return out[x, y]


class NonLinearTransformer(nn.Module):
    def __init__(
        self, vocab_size, embed_size, matrix_dim, heads, window_size, iterations
    ):
        super(NonLinearTransformer, self).__init__()
        self.iterations = iterations
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.local_attention = LocalAttention(embed_size, heads, window_size)

    def forward(self, matrix):
        matrix = self.embedding(matrix)
        for _ in range(self.iterations):
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    matrix[i, j] = self.local_attention(matrix, (i, j))
        return matrix


x = torch.randint(0, 100, (10, 10))

model = NonLinearTransformer(
    vocab_size=100, embed_size=128, matrix_dim=10, heads=8, window_size=3, iterations=2
)

out = model(x)
print(out.shape)
