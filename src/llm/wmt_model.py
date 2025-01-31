import math
from dataclasses import dataclass
from typing import Optional

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
    vocab_size: int
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
        x = self.activation(x)
        x = self.layer2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_k):
        super().__init__()
        self.scale = math.sqrt(dim_k)

    def forward(self, 
        q: torch.Tensor, 
        k: torch.Tensor,
        v: torch.Tensor, 
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,
    ):
        """
        q, k, v: [batch, heads, seq_len, dim_k]
        attn_mask: [batch, 1, q_len, seq_len]
        """
        # (B, heads, q_len, dim_k) x (B, heads, dim_k, k_len) -> (B, heads, q_len, k_len)
        # ENCODER:
        # q, k, v => [B, 8, q_len, 64] where dim_k = model_dim//num_heads = 64
        # attn_mask => [B, q_len], where enc_seq_len = q_len => expand to [B, 1, q_len, q_len]
        # scores => [B, 8, q_len, q_len]
        # scores.masked_fill_(~attn_mask, float('-inf'))
        #
        # DECODER:
        # Self attention: here we do causal attention
        # q, k, v => [B, 8, q_len, 64] where dim_k = model_dim//num_heads = 64
        # attn_mask => [B, q_len], where dec_seq_len = q_len => expand to [B, 1, q_len, q_len]
        # scores => [B, 8, q_len, q_len]
        # causal_mask => triu(ones_like(scores), diagonal=1) => [B, 8, q_len, q_len]
        # total_mask => attn_mask + causal_mask => [B, 8, q_len, q_len]
        # scores.masked_fill_(total_mask, float('-inf'))
        #
        # Cross attention: here we do not do causal attention
        # q, k, v => [B, 8, q_len, 64], [B, 8, enc_seq_len, 64], [B, 8, enc_seq_len, 64], here dec_seq_len = q_len
        # attn_mask => [B, enc_seq_len] => expand to [B, 1, q_len, enc_seq_len]
        # scores => [B, 8, q_len, enc_seq_len]
        # scores.masked_fill_(~attn_mask, float('-inf'))
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Causal mask (upper triangular)
        if is_causal:
            causal_mask = torch.triu(
                torch.ones_like(scores, dtype=torch.bool), diagonal=1
            )
            attn_mask = torch.logical_not(
                torch.logical_and(causal_mask, ~attn_mask.bool())
            )

        # In the implementation, mask is always available
        scores.masked_fill_(~attn_mask.bool(), float('-inf'))

        attn = torch.softmax(scores, dim=-1)

        # Multiply by values: (B, heads, q_len, k_len) x (B, heads, k_len, dim_k) -> (B, heads, q_len, dim_k)
        out = torch.matmul(attn, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        assert model_dim % num_heads == 0, (
            f"model_dim={model_dim} must be divisible by num_heads={num_heads}"
        )
        self.num_heads = num_heads
        self.dim_k = self.dim_v = model_dim // num_heads

        # Instead of a separate projection for each head, do one big projection for Q, K, V
        self.W_q = nn.Linear(model_dim, model_dim, bias=False)
        self.W_k = nn.Linear(model_dim, model_dim, bias=False)
        self.W_v = nn.Linear(model_dim, model_dim, bias=False)

        self.scaled_dot = ScaledDotProductAttention(dim_k=self.dim_k)
        self.W_out = nn.Linear(model_dim, model_dim)  # final linear after concat

    def forward(self, 
        query: torch.Tensor,  # [batch, q_len, model_dim]
        key: torch.Tensor,    # [batch, k_len, model_dim]
        value: torch.Tensor,  # [batch, k_len, model_dim]
        attention_mask: Optional[torch.Tensor] = None, # [batch, seq_len]
        is_causal: bool = False
    ):
        B, q_len, _ = query.size()
        B, k_len, _ = key.size()
    
        # 1) Project Q, K, V
        q = self.W_q(query)  # [B, q_len, model_dim]
        k = self.W_k(key)    # [B, k_len, model_dim]
        v = self.W_v(value)  # [B, k_len, model_dim]

        # 2) Reshape for multi‐head: [B, q_len, num_heads, dim_k]
        #    then transpose to [B, num_heads, q_len, dim_k]
        q = q.view(B, q_len, self.num_heads, self.dim_k).transpose(1, 2) # [batch, num_heads, q_len, dim_k]
        k = k.view(B, k_len, self.num_heads, self.dim_k).transpose(1, 2) # [batch, num_heads, k_len, dim_k]
        v = v.view(B, k_len, self.num_heads, self.dim_v).transpose(1, 2) # [batch, num_heads, k_len, dim_v]

        # Shape mask
        attention_mask = attention_mask.unsqueeze(1) # [B, 1, seq_len]
        attention_mask = attention_mask.expand(-1, q_len, -1) # [B, q_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1) # [B, 1, q_len, seq_len]

        # 3) Scaled dot‐product attention per head
        #    q,k,v => [B, heads, q_len, dim_k]
        out = self.scaled_dot(q, k, v, attention_mask, is_causal=is_causal)

        # 4) Transpose/reshape back to [B, q_len, heads * dim_k]
        out = out.transpose(1, 2).contiguous()  # => [B, q_len, heads, dim_k]
        out = out.view(B, q_len, self.num_heads * self.dim_k)

        # 5) Final linear
        out = self.W_out(out)  # => [B, q_len, model_dim]
        return out


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
        encoder_self_attention_mask: Optional[Float[Tensor, "batch enc_seq_len"]] = None,
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:  # type: ignore
        residual = inputs

        x = self.multi_head_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=encoder_self_attention_mask,
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
        encoder_self_attention_mask: Optional[Float[Tensor, "batch enc_seq_len"]] = None,
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:  # type: ignore
        # We pass the input through each encoder layer and return the final output
        x = encoder_input
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(
                inputs=x,
                encoder_self_attention_mask=encoder_self_attention_mask,
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
        encoder_self_attention_mask: Optional[Float[Tensor, "batch enc_seq_len"]] = None,
        decoder_self_attention_mask: Optional[Float[Tensor, "batch dec_seq_len"]] = None,
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:  # type: ignore
        residual = decoder_input

        x = self.masked_multi_head_attention(
            query=decoder_input,
            key=decoder_input,
            value=decoder_input,
            attention_mask=decoder_self_attention_mask,
            is_causal=True,
        )
        x = self.residual_dropout_1(x, residual)
        x = self.layer_norm1(x)

        residual = x

        x = self.multi_head_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attention_mask=encoder_self_attention_mask,
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
        encoder_self_attention_mask: Optional[Float[Tensor, "batch enc_seq_len"]] = None,
        decoder_self_attention_mask: Optional[Float[Tensor, "batch dec_seq_len"]] = None,
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:  # type: ignore
        x = decoder_input
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                decoder_input=x,
                encoder_output=encoder_output,
                encoder_self_attention_mask=encoder_self_attention_mask,
                decoder_self_attention_mask=decoder_self_attention_mask,
            )
        return x


class Embedding(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=model_dim
        )
        self.scale = math.sqrt(model_dim)

    def forward(self, tokens: Float[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len model_dim"]:  # type: ignore
        # as suggested in the paper (https://arxiv.org/abs/1706.03762) in section 3.4
        embedding = self.embedding(tokens) * self.scale
        return embedding


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_seq_len: int, dropout_rate: float):
        super().__init__()
        position_encoding = torch.zeros(
            max_seq_len, model_dim
        )  # [max_seq_len, model_dim]
        positions = torch.arange(0, max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        div_term = 1 / (
            torch.pow(10000, torch.arange(0, model_dim, 2) / model_dim)
        )  # [model_dim//2]
        freq = positions * div_term  # [max_seq_len, model_dim//2]
        # all the positions but even spaced dimensions
        position_encoding[:, 0::2] = torch.sin(freq)
        position_encoding[:, 1::2] = torch.cos(freq)

        # Register it as a buffer so that .to(device) moves it properly
        self.register_buffer("position_encoding", position_encoding)

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


# TODO: add subclassing for Mixin for transformer integration
class Transformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
    ):
        super().__init__()
        self.embedding = Embedding(
            model_dim=config.model_dim, vocab_size=config.vocab_size
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
            model_dim=config.model_dim, vocab_size=config.vocab_size
        )
        self.lm_head.weight = self.embedding.embedding.weight

    def forward(
        self,
        encoder_input: Float[Tensor, "batch enc_seq_len"],  # type: ignore
        decoder_input: Float[Tensor, "batch dec_seq_len"],  # type: ignore
        encoder_self_attention_mask: Float[Tensor, "batch enc_seq_len"],  # type: ignore
        decoder_self_attention_mask: Float[Tensor, "batch dec_seq_len"],  # type: ignore
    ) -> Float[Tensor, "batch dec_seq_len vocab_size"]:  # type: ignore
        # Embed the source input and add positional encoding
        encoder_input = self.embedding(encoder_input)
        encoder_input = self.positional_encoding(encoder_input)

        # Encode the source input
        encoder_output = self.encoder(
            encoder_input, encoder_self_attention_mask=encoder_self_attention_mask
        )

        # Embed the target input and add positional encoding
        decoder_input = self.embedding(decoder_input)
        decoder_input = self.positional_encoding(decoder_input)

        # Decode the target input
        decoder_output = self.decoder(
            decoder_input,
            encoder_output,
            encoder_self_attention_mask=encoder_self_attention_mask,
            decoder_self_attention_mask=decoder_self_attention_mask,
        )

        # Get the logits for the next token
        logits = self.lm_head(decoder_output)  # [batch, dec_seq_len, vocab_size]
        return logits


if __name__ == "__main__":
    from llm.wmt_data_utils import _collate_fn
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained("llm-scratch/wmt-14-en-de-tok")
    tokenizer.enable_padding(pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=512)

    batch = [
        {
            "translation": {
                "en": "Hello, how are you?",
                "de": "Hallo, wie geht es dir?",
            }
        },
        {
            "translation": {
                "en": "I am fine, thank you!",
                "de": "Ich bin gut, danke!",
            }
        },
    ]

    inputs = _collate_fn(batch, "en", "de", tokenizer)
    print(f"{inputs=}")

    encoder_input_ids = inputs["source_input"]["input_ids"]
    decoder_input_ids = inputs["target_input"]["input_ids"]
    encoder_self_attention_mask = inputs["source_input"]["self_attention_mask"]
    decoder_self_attention_mask = inputs["target_input"]["self_attention_mask"]

    print(f"{encoder_input_ids.shape=}")
    print(f"{decoder_input_ids.shape=}")
    print(f"{encoder_self_attention_mask.shape=}")
    print(f"{decoder_self_attention_mask.shape=}")

    config = TransformerConfig(
        model_dim=32,
        expansion_dim=64,
        num_heads=4,
        num_blocks=1,
        dropout_rate=0.1,
        vocab_size=37000,
        max_seq_len=100,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Transformer(config=config)
    model.to(device)
    print(model)

    out = model(
        encoder_input=encoder_input_ids.to(device),
        decoder_input=decoder_input_ids.to(device),
        encoder_self_attention_mask=encoder_self_attention_mask.to(device),
        decoder_self_attention_mask=decoder_self_attention_mask.to(device),
    )

    print(f"{out.shape=}")
    print(f"{out=}")
    print(f"{torch.softmax(out, dim=-1)=}")
    print(f"{torch.softmax(out, dim=-1).shape=}")

    print(f"{torch.argmax(torch.softmax(out, dim=-1), dim=-1)=}")
    print(f"{torch.argmax(torch.softmax(out, dim=-1), dim=-1).shape=}")

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=1,
        reduction="none",
    )

    out = out.view(-1, out.shape[-1])
    targets = decoder_input_ids.to(device).view(-1)

    loss = loss_fn(out, targets)
    print(f"{loss=}")
