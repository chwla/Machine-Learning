import torch
import torch.nn as nn
import tiktoken

# -------------------
# Config dictionary
# -------------------
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # Query-Key-Value bias
}


# -------------------
# Model components
# -------------------
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)  # (B, T, D)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (T, D)

        # Add position embeddings (broadcasts over batch)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (B, T, vocab_size)

        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Placeholder: no real attention or MLP
    def forward(self, x):
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x):
        return self.norm(x)


# -------------------
# Tokenization + batch prep
# -------------------
tokenizer = tiktoken.get_encoding("gpt2")

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

tok1 = torch.tensor(tokenizer.encode(txt1))
tok2 = torch.tensor(tokenizer.encode(txt2))

# Pad to same length (required for stacking)
max_len = max(len(tok1), len(tok2))
tok1 = torch.cat([tok1, torch.zeros(max_len - len(tok1), dtype=torch.long)])
tok2 = torch.cat([tok2, torch.zeros(max_len - len(tok2), dtype=torch.long)])

batch = torch.stack([tok1, tok2], dim=0)  # (B, T)
print("Batch shape:", batch.shape)


# -------------------
# Run model
# -------------------
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

logits = model(batch)
print("Output shape:", logits.shape)
