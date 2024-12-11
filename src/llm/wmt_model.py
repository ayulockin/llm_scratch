from typing import List
import torch
import math
from torch import Tensor
from torch import nn
from dataclasses import dataclass
from jaxtyping import Float


@dataclass
class TransformerConfig:
    model_dim: int
    expansion_dim: int
    num_heads: int
    num_blocks: int


class FeedForward(nn.Module):
    def __init__(
        self,
        model_dim: int,
        expansion_dim: int,
    ):
        super().__init__()
        self.layer1 = nn.Linear(model_dim, expansion_dim, bias=True)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(expansion_dim, model_dim, bias=True)

    def forward(
        self,
        inputs: Float[Tensor, "batch seq_len model_dim"],
    ) -> Float[Tensor, "batch seq_len model_dim"]:
        x = self.layer1(inputs)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dim_k: int,
        dim_v: int,
    ):
        super().__init__()
        self.W_query = nn.Linear(model_dim, dim_k, bias=False)
        self.W_key = nn.Linear(model_dim, dim_k, bias=False)
        self.W_value = nn.Linear(model_dim, dim_v, bias=False)
        self.dim_k = dim_k

    def forward(
        self,
        query: Float[Tensor, "batch seq_len model_dim"],
        key: Float[Tensor, "batch seq_len model_dim"],
        value: Float[Tensor, "batch seq_len model_dim"],
        is_causal: bool = False,
    ) -> Float[Tensor, "batch seq_len dim_v"]:
        # project key and query
        key = self.W_key(key)
        query = self.W_query(query)

        similarity = torch.matmul(query, torch.permute(key, dims=(0, 2, 1)))
        scaled_similarity = torch.divide(similarity, math.sqrt(self.dim_k))

        if is_causal:
            scaled_similarity = scaled_similarity.masked_fill(
                mask=torch.triu(torch.ones_like(scaled_similarity).bool(), diagonal=1),
                value=-torch.inf,
            )
        attention_scores = torch.softmax(
            scaled_similarity, dim=-1
        )  # Careful with the `dim`

        # project the value
        value = self.W_value(value)
        outputs = torch.matmul(attention_scores, value)

        return outputs


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, (
            "model dimensions should be divisible by number of heads"
            f"model_dim was {model_dim} and num_heads was {num_heads}"
        )
        dim_k = dim_v = model_dim // num_heads
        self.projection_heads = list()
        for _ in range(num_heads):
            self.projection_heads.append(
                ScaledDotProductAttention(model_dim=model_dim, dim_k=dim_k, dim_v=dim_v)
            )
        self.W_out = nn.Linear(dim_v * num_heads, model_dim)

    def forward(
        self,
        query: Float[Tensor, "batch seq_len model_dim"],
        key: Float[Tensor, "batch seq_len model_dim"],
        value: Float[Tensor, "batch seq_len model_dim"],
        is_causal: bool = False,
    ) -> Float[Tensor, "batch seq_len model_dim"]:
        outputs = list()
        for head in self.projection_heads:
            outputs.append(head(query, key, value, is_causal=is_causal))

        outputs = torch.concat(outputs, dim=-1)
        outputs = self.W_out(outputs)
        return outputs


class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        expansion_dim: int,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            model_dim=model_dim, num_heads=num_heads
        )
        self.feed_forward = FeedForward(
            model_dim=model_dim, expansion_dim=expansion_dim
        )
        self.layer_norm1 = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=model_dim)

    def forward(
        self, inputs: Float[Tensor, "batch enc_seq_len model_dim"]
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:
        residual = inputs

        x = self.multi_head_attention(query=inputs, key=inputs, value=inputs)
        x = x + residual
        x = self.layer_norm1(x)

        residual = x

        x = self.feed_forward(x)
        x = x + residual
        x = self.layer_norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        expansion_dim: int,
    ):
        super().__init__()
        self.encoder_layers = list()
        for _ in range(num_blocks):
            self.encoder_layers.append(
                EncoderBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    expansion_dim=expansion_dim,
                )
            )

    def forward(
        self,
        encoder_input: Float[Tensor, "batch enc_seq_len model_dim"],
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:
        encoder_outputs = list()
        for encoder_layer in self.encoder_layers:
            encoder_input = encoder_layer(encoder_input)
            encoder_outputs.append(encoder_input)
        return encoder_outputs


class DecoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        expansion_dim: int,
    ):
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(
            model_dim=model_dim, num_heads=num_heads
        )
        self.multi_head_attention = MultiHeadAttention(
            model_dim=model_dim, num_heads=num_heads
        )
        self.feed_forward = FeedForward(
            model_dim=model_dim, expansion_dim=expansion_dim
        )
        self.layer_norm1 = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=model_dim)
        self.layer_norm3 = nn.LayerNorm(normalized_shape=model_dim)

    def forward(
        self,
        decoder_input: Float[Tensor, "batch dec_seq_len model_dim"],
        encoder_outputs: Float[Tensor, "batch enc_seq_len model_dim"],
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:
        residual = decoder_input

        x = self.masked_multi_head_attention(
            query=decoder_input,
            key=decoder_input,
            value=decoder_input,
            is_causal=True,
        )
        x = x + residual
        x = self.layer_norm1(x)

        residual = x

        x = self.multi_head_attention(
            query=x, key=encoder_outputs, value=encoder_outputs
        )
        x = x + residual
        x = self.layer_norm2(x)

        residual = x

        x = self.feed_forward(x)
        x = x + residual
        x = self.layer_norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        expansion_dim: int,
    ):
        super().__init__()
        self.decoder_layers = list()
        for _ in range(num_blocks):
            self.decoder_layers.append(
                DecoderBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    expansion_dim=expansion_dim,
                )
            )

    def forward(
        self,
        decoder_input: Float[Tensor, "batch dec_seq_len model_dim"],
        encoder_outputs: List[Float[Tensor, "batch enc_seq_len model_dim"]],
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:
        x = decoder_input
        for idx, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(decoder_input=x, encoder_outputs=encoder_outputs[idx])
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        self.encoder = Encoder(
            model_dim=config.model_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            expansion_dim=config.expansion_dim,
        )
        self.decoder = Decoder(
            model_dim=config.model_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            expansion_dim=config.expansion_dim,
        )

    def forward(
        self,
        encoder_input: Float[Tensor, "batch enc_seq_len model_dim"],
        decoder_input: Float[Tensor, "batch dec_seq_len model_dim"],
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:
        encoder_outputs = self.encoder(
            encoder_input=encoder_input,
        )  # this is a list of encoder outputs from each layer
        decoder_output = self.decoder(
            decoder_input=decoder_input, encoder_outputs=encoder_outputs
        )
        return decoder_output