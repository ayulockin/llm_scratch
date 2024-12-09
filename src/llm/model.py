import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    model_dim: int
    expansion_dim: int
    num_heads: int
    num_layers: int


class FeedForward(nn.Module):
    def __init__(
        self,
        model_dim,
        expansion_dim,
    ):
        super().__init__()
        self.layer1 = nn.Linear(model_dim, expansion_dim, bias=True)
        self.layer2 = nn.Linear(expansion_dim, model_dim, bias=True)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        model_dim,
        dim_k,
        dim_v,
    ):
        super().__init__()
        self.W_query = nn.Linear(model_dim, dim_k, bias=False)
        self.W_key = nn.Linear(model_dim, dim_k, bias=False)
        self.W_value = nn.Linear(model_dim, dim_v, bias=False)
        self.dim_k = dim_k

    def forward(self, query, key, value, is_causal=False):
        # project key and query
        key = self.W_key(key)
        query = self.W_query(query)

        similarity = torch.matmul(query, key.T)
        scaled_similarity = torch.divide(similarity, torch.sqrt(self.dim_k))

        if is_causal:
            scaled_similarity = scaled_similarity.masked_fill(
                mask=torch.triu(torch.ones_like(scaled_similarity), diagonal=1),
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
        model_dim,
        num_heads,
    ):
        super().__init__()
        dim_k = dim_v = model_dim // num_heads
        self.projection_heads = list()
        for _ in range(num_heads):
            self.projection_heads.append(
                ScaledDotProductAttention(model_dim=model_dim, dim_k=dim_k, dim_v=dim_v)
            )
        self.W_out = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        outputs = list()
        for head in self.projection_heads:
            outputs.append(head(query, key, value))

        outputs = torch.concat(outputs, dim=-1)
        outputs = self.W_out(outputs)
        return outputs


class EncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
        expansion_dim,
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

    def forward(self, inputs):
        residual = inputs

        x = self.multi_head_attention(query=inputs, key=inputs, value=inputs)
        x = self.layer_norm1(x) + residual

        residual = x

        x = self.feed_forward(x)
        x = self.layer_norm2(x) + residual
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim,
        num_layers,
        num_heads,
        expansion_dim,
    ):
        super().__init__()
        self.encoder_layers = list()
        for _ in range(num_layers):
            self.encoder_layers.append(
                EncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    expansion_dim=expansion_dim,
                )
            )

    def forward(self, encoder_inputs):
        encoder_outputs = list()
        for encoder_layer in self.encoder_layers:
            encoder_inputs = encoder_layer(encoder_inputs)
            encoder_outputs.append(encoder_inputs)
        return encoder_outputs


class DecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
        expansion_dim,
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

    def forward(self, decoder_inputs, encoder_outputs):
        residual = decoder_inputs

        x = self.masked_multi_head_attention(
            query=decoder_inputs,
            key=decoder_inputs,
            value=decoder_inputs,
            is_causal=True,
        )
        x = self.layer_norm1(x) + residual

        residual = x

        x = self.multi_head_attention(
            query=x, key=encoder_outputs, value=encoder_outputs
        )
        x = self.layer_norm1(x) + residual

        residual = x

        x = self.feed_forward(x)
        x = self.layer_norm2(x) + residual
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        model_dim,
        num_layers,
        num_heads,
        expansion_dim,
    ):
        super().__init__()
        self.decoder_layers = list()
        for _ in range(num_layers):
            self.decoder_layers.append(
                EncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    expansion_dim=expansion_dim,
                )
            )

    def forward(self, decoder_inputs, encoder_outputs):
        x = decoder_inputs
        for idx, decoder_layer in enumerate(self.decoder_layers):
            x = decoder_layer(decoder_inputs=x, encoder_outputs=encoder_outputs[idx])
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.encoder = Encoder(
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            expansion_dim=config.expansion_dim,
        )
        self.decoder = Decoder(
            model_dim=config.model_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            expansion_dim=config.expansion_dim,
        )

    def forward(self, encoder_inputs, decoder_inputs):
        # We assume the encoder inputs and the decoder inputs
        # are tokens after adding positional information into them
        encoder_outputs = self.encoder(
            encoder_inputs=encoder_inputs,
        )  # this is a list of encoder outputs from each layer
        decoder_outputs = self.decoder(
            decoder_inputs=decoder_inputs, encoder_outputs=encoder_outputs
        )
        return decoder_outputs
