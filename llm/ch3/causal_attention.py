
# Causal attention

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length)) #tril function creates a mask where the values above the diagonal are zero
print(mask_simple)

masked_simple = attn_weights*mask_simple
print(masked_simple)

row_sums = masked_simple.sum(dim = -1, keepdim = True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm) #renormalizing the attention weights to sum up to 1 again
