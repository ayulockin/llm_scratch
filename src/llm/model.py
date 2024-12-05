import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, model_dim, reduction_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_layer = nn.Linear(
            model_dim, reduction_ratio * model_dim, bias=False
        )
        self.expansion_layer = nn.Linear(
            reduction_ratio * model_dim, model_dim, bias=False
        )

    def forward(self, inputs):
        compressed_maps = self.compression_layer(inputs)
        decompressed_maps = self.expansion_layer(compressed_maps)
        return decompressed_maps


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.proj_q = nn.Linear(model_dim, model_dim, bias=True)
        self.proj_k = nn.Linear(model_dim, model_dim, bias=True)
        self.proj_v = nn.Linear(model_dim, model_dim, bias=True)
    
    def compute_attention(self, query, key, value):
        similarity = torch.matmul(query, key.T)
        normalized_similarity = torch.divide(similarity, torch.sqrt(self.model_dim // self.num_heads))
        attention_scores = torch.softmax(normalized_similarity)
        outputs = attention_scores * value
        return outputs
    
    def forward(self, query, key, value):
        projected_query = self.proj_q(query)
        projected_key = self.proj_k(key)
        projected_value = self.proj_v(value)
        
        query_splits = torch.split(projected_query, split_size_or_sections=self.num_heads, dim=-1)
        key_splits = torch.split(projected_key, split_size_or_sections=self.num_heads, dim=-1)
        value_splits = torch.split(projected_value, split_size_or_sections=self.num_heads, dim=-1)

        multi_head_attention_outputs = [self.compute_attention(q, k, v) for q,k,v in zip(query_splits, key_splits, value_splits)]
        attention_outputs = torch.concat(multi_head_attention_outputs, dim=-1)

        return attention_outputs


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        return inputs


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        return inputs
