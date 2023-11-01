import gzip
import random

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset

from heptapod.at import Autoregressive2DWrapper
from heptapod.model import NonLinearTransformer

# Constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

# Helpers
def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# Instantiate GPT-like decoder model
model = NonLinearTransformer(vocab_size=10000, dim=512, depth=6, matrix_dim=5)
model = Autoregressive2DWrapper(model)
# model.cuda()

# Prepare enwik8 data
with gzip.open("./data/enwik8.gz") as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.side_len = int(np.sqrt(seq_len))  # Calculate side length of the square matrix

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.side_len**2, (1,))
        seq = self.data[rand_start: rand_start + self.side_len**2].long()
        matrix = seq.view(self.side_len, self.side_len)  # Reshape the sequence into a matrix
        return matrix.cuda()

    def __len__(self):
        return self.data.size(0) // (self.side_len**2)


train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc='training'):
    model.train()
    
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        loss.backward()

    print(f"training loss: {loss.item()}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader))
            print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print("%s \n\n %s", (prime, '*'*100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str)
