import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from jaxtyping import Float
from torch import Tensor, nn


@dataclass
class TransformerConfig:
    model_dim: int
    expansion_dim: int
    num_heads: int
    num_blocks: int
    dropout_rate: float
    vocab_src_size: int
    vocab_tgt_size: int
    max_seq_len: int


class ResidualDropout(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        inputs: Float[Tensor, "batch seq_len model_dim"],  # type: ignore
        residual: Optional[Float[Tensor, "batch seq_len model_dim"]] = None,  # type: ignore
    ) -> Float[Tensor, "batch seq_len model_dim"]:  # type: ignore
        inputs = self.dropout(inputs)
        if residual is not None:
            return inputs + residual
        return inputs


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
        inputs: Float[Tensor, "batch seq_len model_dim"],  # type: ignore
    ) -> Float[Tensor, "batch seq_len model_dim"]:  # type: ignore
        x = self.layer1(inputs)
        # TODO: might add dropout here
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
        query: Float[Tensor, "batch seq_len model_dim"],  # type: ignore
        key: Float[Tensor, "batch seq_len model_dim"],  # type: ignore
        value: Float[Tensor, "batch seq_len model_dim"],  # type: ignore
        attention_mask: Optional[Float[Tensor, "batch seq_len seq_len"]] = None,
        is_causal: bool = False,
    ) -> Float[Tensor, "batch seq_len dim_v"]:  # type: ignore
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

        if attention_mask is not None:
            scaled_similarity = scaled_similarity.masked_fill(
                # reversing the mask because we want to mask out the padding tokens (True in attention mask
                # means padding token)
                mask=~attention_mask,
                value=-torch.inf,
            )

        attention_scores = torch.softmax(
            scaled_similarity, dim=-1
        )  # Careful with the `dim`

        # If a row is fully -inf, the softmax is all NaN. We can clamp that row to 0.
        # This usually happens with short sequences. We can safely ignore these rows by
        # clamping them to 0.
        mask_invalid = torch.isnan(attention_scores)
        if mask_invalid.any():
            attention_scores[mask_invalid] = 0.0

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
        # this registers the heads as submodules
        self.projection_heads = nn.ModuleList(self.projection_heads)  # type: ignore
        self.W_out = nn.Linear(dim_v * num_heads, model_dim)

    def forward(
        self,
        query: Float[Tensor, "batch seq_len model_dim"],  # type: ignore
        key: Float[Tensor, "batch seq_len model_dim"],  # type: ignore
        value: Float[Tensor, "batch seq_len model_dim"],  # type: ignore
        attention_mask: Optional[Float[Tensor, "batch seq_len seq_len"]] = None,
        is_causal: bool = False,
    ) -> Float[Tensor, "batch seq_len model_dim"]:  # type: ignore
        outputs = list()
        for head in self.projection_heads:
            outputs.append(
                head(
                    query,
                    key,
                    value,
                    attention_mask=attention_mask,
                    is_causal=is_causal,
                )
            )

        outputs = torch.concat(outputs, dim=-1)
        outputs = self.W_out(outputs)
        return outputs


class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float,
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
        self.residual_dropout_1 = ResidualDropout(dropout_rate=dropout_rate)
        self.residual_dropout_2 = ResidualDropout(dropout_rate=dropout_rate)

    def forward(
        self,
        inputs: Float[Tensor, "batch enc_seq_len model_dim"],  # type: ignore
        attention_mask: Optional[Float[Tensor, "batch enc_seq_len enc_seq_len"]] = None,
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:  # type: ignore
        residual = inputs

        x = self.multi_head_attention(
            query=inputs, key=inputs, value=inputs, attention_mask=attention_mask
        )
        x = self.residual_dropout_1(x, residual)
        x = self.layer_norm1(x)

        residual = x

        x = self.feed_forward(x)
        x = self.residual_dropout_2(x, residual)
        x = self.layer_norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.encoder_layers = list()
        for _ in range(num_blocks):
            self.encoder_layers.append(
                EncoderBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    expansion_dim=expansion_dim,
                    dropout_rate=dropout_rate,
                )
            )
        # this registers the layers as submodules
        self.encoder_layers = nn.ModuleList(self.encoder_layers)

    def forward(
        self,
        encoder_input: Float[Tensor, "batch enc_seq_len model_dim"],  # type: ignore
        attention_mask: Optional[Float[Tensor, "batch enc_seq_len enc_seq_len"]] = None,
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:  # type: ignore
        # We pass the input through each encoder layer and return the final output
        x = encoder_input
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(
                inputs=x,
                attention_mask=attention_mask,
            )

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float,
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
        self.residual_dropout_1 = ResidualDropout(dropout_rate=dropout_rate)
        self.residual_dropout_2 = ResidualDropout(dropout_rate=dropout_rate)
        self.residual_dropout_3 = ResidualDropout(dropout_rate=dropout_rate)

    def forward(
        self,
        decoder_input: Float[Tensor, "batch dec_seq_len model_dim"],  # type: ignore
        encoder_output: Float[Tensor, "batch enc_seq_len model_dim"],  # type: ignore
        self_attention_mask: Optional[
            Float[Tensor, "batch dec_seq_len dec_seq_len"]
        ] = None,
        cross_attention_mask: Optional[
            Float[Tensor, "batch dec_seq_len enc_seq_len"]
        ] = None,
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:  # type: ignore
        residual = decoder_input

        x = self.masked_multi_head_attention(
            query=decoder_input,
            key=decoder_input,
            value=decoder_input,
            attention_mask=self_attention_mask,
            is_causal=True,
        )
        x = self.residual_dropout_1(x, residual)
        x = self.layer_norm1(x)

        residual = x

        x = self.multi_head_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attention_mask=cross_attention_mask,
        )
        x = self.residual_dropout_2(x, residual)
        x = self.layer_norm2(x)

        residual = x

        x = self.feed_forward(x)
        x = self.residual_dropout_3(x, residual)
        x = self.layer_norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.decoder_layers = list()
        for _ in range(num_blocks):
            self.decoder_layers.append(
                DecoderBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    expansion_dim=expansion_dim,
                    dropout_rate=dropout_rate,
                )
            )
        # this registers the layers as submodules
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

    def forward(
        self,
        decoder_input: Float[Tensor, "batch dec_seq_len model_dim"],  # type: ignore
        encoder_output: Float[Tensor, "batch enc_seq_len model_dim"],  # type: ignore
        self_attention_mask: Optional[
            Float[Tensor, "batch dec_seq_len dec_seq_len"]
        ] = None,
        cross_attention_mask: Optional[
            Float[Tensor, "batch dec_seq_len enc_seq_len"]
        ] = None,
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:  # type: ignore
        x = decoder_input
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                decoder_input=x,
                encoder_output=encoder_output,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
            )
        return x


class Embedding(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.scale = math.sqrt(model_dim)

    def forward(self, tokens: Float[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len model_dim"]:  # type: ignore
        # as suggested in the paper (https://arxiv.org/abs/1706.03762) in section 3.4
        embedding = self.embedding(tokens) * self.scale
        return embedding


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_seq_len: int, dropout_rate: float):
        super().__init__()
        self.position_encoding = torch.zeros(
            max_seq_len, model_dim
        )  # [max_seq_len, model_dim]
        positions = torch.arange(0, max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = 1 / (
            torch.pow(10000, torch.arange(0, model_dim, 2) / model_dim)
        )  # [model_dim//2]
        freq = positions * div_term  # [max_seq_len, model_dim//2]
        # all the positions but even spaced dimensions
        self.position_encoding[:, 0::2] = torch.sin(freq)
        self.position_encoding[:, 1::2] = torch.cos(freq)

        self.residual_dropout = ResidualDropout(dropout_rate=dropout_rate)

    def forward(
        self, inputs: Float[Tensor, "batch seq_len model_dim"]  # type: ignore
    ) -> Float[Tensor, "batch seq_len model_dim"]:  # type: ignore
        x = inputs + self.position_encoding[: inputs.size(1)]
        x = self.residual_dropout(x)  # this is just dropout of x (no residuals)
        return x


class LMHead(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(model_dim, vocab_size)

    def forward(
        self, inputs: Float[Tensor, "batch seq_len model_dim"]  # type: ignore
    ) -> Float[Tensor, "batch seq_len vocab_size"]:  # type: ignore
        x = self.linear(inputs)  # raw logits
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        self.source_embedding = Embedding(
            model_dim=config.model_dim, vocab_size=config.vocab_src_size
        )
        self.target_embedding = Embedding(
            model_dim=config.model_dim, vocab_size=config.vocab_tgt_size
        )
        self.positional_encoding = PositionalEncoding(
            model_dim=config.model_dim,
            max_seq_len=config.max_seq_len,
            dropout_rate=config.dropout_rate,
        )
        self.encoder = Encoder(
            model_dim=config.model_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            expansion_dim=config.expansion_dim,
            dropout_rate=config.dropout_rate,
        )
        self.decoder = Decoder(
            model_dim=config.model_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            expansion_dim=config.expansion_dim,
            dropout_rate=config.dropout_rate,
        )
        self.lm_head = LMHead(
            model_dim=config.model_dim, vocab_size=config.vocab_tgt_size
        )

    def forward(
        self,
        encoder_input: Float[Tensor, "batch enc_seq_len"],  # type: ignore
        decoder_input: Float[Tensor, "batch dec_seq_len"],  # type: ignore
        pad_id: int = 1,
    ) -> Float[Tensor, "batch dec_seq_len vocab_tgt_size"]:  # type: ignore
        ############# ENCODER #############

        # calculate the attention mask for the encoder and decoder
        encoder_attention_mask = encoder_input != pad_id

        # calculate the attention mask for the encoder (self attention)
        batch_size, enc_seq_len = encoder_attention_mask.size()
        encoder_self_attention_mask = encoder_attention_mask.unsqueeze(1).expand(
            batch_size, enc_seq_len, enc_seq_len
        )  # [batch, enc_seq_len, enc_seq_len]

        # Embed the source input and add positional encoding
        encoder_input = self.source_embedding(encoder_input)
        encoder_input = self.positional_encoding(
            encoder_input
        )  # [batch, enc_seq_len, model_dim]

        # Encode the source input
        encoder_output = self.encoder(
            encoder_input, attention_mask=encoder_self_attention_mask
        )  # [batch, enc_seq_len, model_dim]

        ############# DECODER #############

        # calculate the attention mask for the decoder (self attention)
        decoder_attention_mask = decoder_input != pad_id

        # calculate the attention mask for the decoder (self attention)
        batch_size, dec_seq_len = decoder_attention_mask.size()
        decoder_self_attention_mask = decoder_attention_mask.unsqueeze(1).expand(
            batch_size, dec_seq_len, dec_seq_len
        )  # [batch, dec_seq_len, dec_seq_len]

        # calculate the attention mask for the decoder (cross attention)
        decoder_attention_mask = decoder_attention_mask.unsqueeze(
            2
        )  # [batch, dec_len, 1]
        encoder_attention_mask = encoder_attention_mask.unsqueeze(
            1
        )  # [batch, 1, enc_len]
        decoder_cross_attention_mask = (
            decoder_attention_mask & encoder_attention_mask
        )  # shape [batch, dec_len, enc_len]

        # Embed the target input and add positional encoding
        decoder_input = self.target_embedding(decoder_input)
        decoder_input = self.positional_encoding(
            decoder_input
        )  # [batch, dec_seq_len, model_dim]

        # Decode the target input
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            self_attention_mask=decoder_self_attention_mask,
            cross_attention_mask=decoder_cross_attention_mask,
        )  # [batch, dec_seq_len, model_dim]

        # Get the logits for the next token
        logits = self.lm_head(decoder_output)  # [batch, dec_seq_len, vocab_size]
        return logits


if __name__ == "__main__":
    config = TransformerConfig(
        model_dim=32,
        expansion_dim=64,
        num_heads=4,
        num_blocks=2,
        dropout_rate=0.1,
        vocab_src_size=1500,
        vocab_tgt_size=1500,
        max_seq_len=100,
    )
    model = Transformer(config=config)
    print(model)

    batch_size = 2
    encoder_padded_seq_len = 30
    decoder_padded_seq_len = 40
    encoder_input = torch.randint(0, config.vocab_src_size, (batch_size, 20))
    encoder_input = torch.cat(
        [
            encoder_input,
            torch.ones((batch_size, encoder_padded_seq_len - 20), dtype=torch.int64),
        ],
        dim=-1,
    )
    decoder_input = torch.randint(0, config.vocab_tgt_size, (batch_size, 30))
    decoder_input = torch.cat(
        [
            decoder_input,
            torch.ones((batch_size, decoder_padded_seq_len - 30), dtype=torch.int64),
        ],
        dim=-1,
    )
    print(f"{encoder_input.shape=}")
    print(f"{decoder_input.shape=}")

    out = model(encoder_input, decoder_input)
    print(f"{out.shape=}")
    print(f"{out=}")
    print(f"{torch.softmax(out, dim=-1)=}")
    print(f"{torch.softmax(out, dim=-1).shape=}")

    print(f"{torch.argmax(torch.softmax(out, dim=-1), dim=-1)=}")
    print(f"{torch.argmax(torch.softmax(out, dim=-1), dim=-1).shape=}")
