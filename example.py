import torch
from heptapod.model import NonLinearTransformer

x = torch.randint(0, 100, (10, 10))

model = NonLinearTransformer(
    vocab_size=100, embed_size=128, matrix_dim=10, heads=8, window_size=3, iterations=2
)

out = model(x)
print(out.shape)
