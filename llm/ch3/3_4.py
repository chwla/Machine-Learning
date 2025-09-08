#A wrapper class to implement multi-head attention
class MultiHeadAttentionWrapper(nn.Module):
    def __init_(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            )
            for _ in range(num_heads)]
        )
    
    def forward(self , x):
        return torch.cat([head(x) for head in self.heads], dim = -1)
    
torch.manual_seed(123)
context_length = batch.shape[1] #This is the number of tokens
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads = 2
)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)