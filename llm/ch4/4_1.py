import torch
import torch.nn as nn

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
        
        x = tok_embeds + pos_embeds  # (B, T, D)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)  # (B, T, vocab_size)

        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # placeholder: no attention/MLP, just return x
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        return self.norm(x)
