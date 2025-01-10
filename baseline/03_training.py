from tokenizers import Tokenizer
from datasets import load_dataset

import torch
from torch import nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import math
from jaxtyping import Float, Bool
from dataclasses import dataclass

from tqdm import tqdm

import wandb
from huggingface_hub import PyTorchModelHubMixin
from functools import partial
import config


def collate_fn(batch, tokenizer):
    source = []
    target = []
    for sample in batch:
        source.append(sample["de"])
        target.append(f"[BOS] {sample['en']} [EOS]")

    source = tokenizer.encode_batch(source)
    target = tokenizer.encode_batch(target)

    return {
        "input_ids": torch.tensor([s.ids for s in source]).int(),
        "input_attention_mask": torch.tensor([s.attention_mask for s in source]).bool(),
        "labels": torch.tensor([t.ids for t in target]).int(),
        "labels_attention_mask": torch.tensor(
            [t.attention_mask for t in target]
        ).bool(),
    }


def learning_rate_schedule(step_num, model_dim):
    if step_num == 0:
        step_num = 1  # Prevent division by zero
    scale = model_dim**-0.5
    step_term = min(step_num**-0.5, step_num * warmup_steps**-1.5)
    return scale * step_term


class Embedding(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.scale = math.sqrt(model_dim)

    def forward(
        self, tokens: Float[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len model_dim"]:
        # as suggested in the paper (https://arxiv.org/abs/1706.03762) in section 3.4
        embedding = self.embedding(tokens) * self.scale
        return embedding


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_seq_len: int, dropout_rate: float):
        super().__init__()
        self.position_encoding = torch.empty(max_seq_len, model_dim)

        positions = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = 1 / (torch.pow(10000, torch.arange(0, model_dim, 2) / model_dim))
        freq = positions * div_term

        # all the positions but even spaced dimensions
        self.position_encoding[:, 0::2] = torch.sin(freq)
        self.position_encoding[:, 1::2] = torch.cos(freq)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        inputs: Float[Tensor, "batch seq_len model_dim"],
    ) -> Float[Tensor, "batch seq_len model_dim"]:
        x = inputs + self.position_encoding[: inputs.size(1)]
        x = self.dropout(x)
        return x


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
    def __init__(self, dim_k, dim_v, model_dim):
        super().__init__()
        self.dim_k = dim_k

        self.W_q = nn.Linear(model_dim, dim_k, bias=False)
        self.W_k = nn.Linear(model_dim, dim_k, bias=False)
        self.W_v = nn.Linear(model_dim, dim_v, bias=False)

    def forward(self, query, key, value, key_attention_mask, is_causal=False):
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        similarity = torch.einsum("bqd, bkd -> bqk", query, key)
        scaled_similarity = torch.divide(similarity, math.sqrt(self.dim_k))

        key_attention_mask = key_attention_mask.unsqueeze(1)

        if is_causal:
            causal_mask = torch.triu(
                torch.ones_like(scaled_similarity).bool(), diagonal=1
            )
            mask = torch.logical_or(key_attention_mask, causal_mask)
        else:
            mask = key_attention_mask

        scaled_similarity = scaled_similarity.masked_fill(
            torch.logical_not(mask), value=-torch.inf
        )

        attention_scores = torch.softmax(scaled_similarity, dim=-1)
        outputs = torch.einsum("bqk, bkd -> bqd", attention_scores, value)

        return outputs


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
    ):
        super().__init__()
        dim_k = dim_v = model_dim // num_heads
        self.projection_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.projection_heads.append(
                ScaledDotProductAttention(dim_k=dim_k, dim_v=dim_v, model_dim=model_dim)
            )
        self.W_out = nn.Linear(dim_v * num_heads, model_dim)

    def forward(
        self,
        query: Float[Tensor, "batch q_seq_len model_dim"],
        key: Float[Tensor, "batch k_seq_len model_dim"],
        value: Float[Tensor, "batch k_seq_len model_dim"],
        key_attention_mask: Bool[Tensor, "batch k_seq_len"],
        is_causal: bool = False,
    ) -> Float[Tensor, "batch seq_len model_dim"]:
        outputs = list()
        for head in self.projection_heads:
            output = head(
                query=query,
                key=key,
                value=value,
                key_attention_mask=key_attention_mask,
                is_causal=False,
            )
            outputs.append(output)

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
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(
        self,
        inputs: Float[Tensor, "batch enc_seq_len model_dim"],
        key_attention_mask: Bool[Tensor, "batch enc_seq_len"],
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:
        residual = inputs

        x = self.multi_head_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            key_attention_mask=key_attention_mask,
        )
        x = self.dropout1(x) + residual
        x = self.layer_norm1(x)

        residual = x

        x = self.feed_forward(x)
        x = self.dropout2(x) + residual
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
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.encoder_layers.append(
                EncoderBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    expansion_dim=expansion_dim,
                    dropout_rate=dropout_rate,
                )
            )

    def forward(
        self,
        encoder_input: Float[Tensor, "batch enc_seq_len model_dim"],
        key_attention_mask: Bool[Tensor, "batch enc_seq_len"],
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:
        # We pass the input through each encoder layer and return the final output
        for encoder_layer in self.encoder_layers:
            encoder_input = encoder_layer(encoder_input, key_attention_mask)

        return encoder_input


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
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(
        self,
        decoder_input: Float[Tensor, "batch dec_seq_len model_dim"],
        encoder_output: Float[Tensor, "batch enc_seq_len model_dim"],
        enc_key_attention_mask: Bool[Tensor, "batch enc_seq_len"],
        dec_key_attention_mask: Bool[Tensor, "batch dec_seq_len"],
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:
        residual = decoder_input

        x = self.masked_multi_head_attention(
            query=decoder_input,
            key=decoder_input,
            value=decoder_input,
            is_causal=True,
            key_attention_mask=dec_key_attention_mask,
        )
        x = self.dropout1(x) + residual
        x = self.layer_norm1(x)

        residual = x

        x = self.multi_head_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            key_attention_mask=enc_key_attention_mask,
        )
        x = self.dropout2(x) + residual
        x = self.layer_norm2(x)

        residual = x

        x = self.feed_forward(x)
        x = self.dropout3(x) + residual
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
        self.decoder_layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.decoder_layers.append(
                DecoderBlock(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    expansion_dim=expansion_dim,
                    dropout_rate=dropout_rate,
                )
            )

    def forward(
        self,
        decoder_input: Float[Tensor, "batch dec_seq_len model_dim"],
        encoder_output: Float[Tensor, "batch enc_seq_len model_dim"],
        enc_key_attention_mask: Bool[Tensor, "batch enc_seq_len"],
        dec_key_attention_mask: Bool[Tensor, "batch dec_seq_len"],
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:
        x = decoder_input
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                decoder_input=x,
                encoder_output=encoder_output,
                enc_key_attention_mask=enc_key_attention_mask,
                dec_key_attention_mask=dec_key_attention_mask,
            )
        return x


@dataclass
class TransformerConfig:
    model_dim: int
    expansion_dim: int
    num_heads: int
    num_blocks: int
    dropout_rate: float
    vocab_size: int
    max_seq_len: int


class Transformer(nn.Module, PyTorchModelHubMixin):
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

        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.embedding.weight

    def forward(
        self,
        encoder_input: Float[Tensor, "batch enc_seq_len"],
        decoder_input: Float[Tensor, "batch dec_seq_len"],
        enc_key_attention_mask: Bool[Tensor, "batch enc_seq_len"],
        dec_key_attention_mask: Bool[Tensor, "batch dec_seq_len"],
    ) -> Float[Tensor, "batch dec_seq_len vocab_tgt_size"]:
        # Embed the source input and add positional encoding
        encoder_input = self.embedding(encoder_input)
        encoder_input = self.positional_encoding(encoder_input)

        # Embed the target input and add positional encoding
        decoder_input = self.embedding(decoder_input)
        decoder_input = self.positional_encoding(decoder_input)

        # Encode the source input
        encoder_output = self.encoder(
            encoder_input=encoder_input,
            key_attention_mask=enc_key_attention_mask,
        )

        # Decode the target input
        decoder_output = self.decoder(
            decoder_input=decoder_input,
            encoder_output=encoder_output,
            enc_key_attention_mask=enc_key_attention_mask,
            dec_key_attention_mask=dec_key_attention_mask,
        )

        # Get the logits for the next token
        logits = self.lm_head(decoder_output)
        return logits


if __name__ == "__main__":
    train_dataset = load_dataset(config.DE_EN_SPLIT_DATASET, split="train")
    val_dataset = load_dataset(config.DE_EN_SPLIT_DATASET, split="validation")

    tokenizer = Tokenizer.from_pretrained(config.TOKENIZER_ID)
    tokenizer.enable_padding(pad_token=config.SPECIAL_TOKENS["pad_token"])
    tokenizer.enable_truncation(config.MAX_SEQ_LENGTH)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model_config = TransformerConfig(
        model_dim=config.MODEL_DIM,
        expansion_dim=config.EXPANSION_DIM,
        num_heads=config.NUM_HEADS,
        num_blocks=config.NUM_HEADS,
        dropout_rate=config.DROPOUT_RATE,
        max_seq_len=config.MAX_SEQ_LENGTH,
        vocab_size=config.VOCAB_SIZE,
    )

    total_num_batches = len(train_dataloader)
    total_steps = total_num_batches * config.NUM_EPOCHS
    warmup_steps = int(config.WARMUP_PERCENTAGE * total_steps)

    model = Transformer(config).to("cuda")
    model.positional_encoding.position_encoding = (
        model.positional_encoding.position_encoding.to("cuda")
    )

    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS, eps=config.EPSILON)
    scheduler = LambdaLR(optimizer, lr_lambda=partial(learning_rate_schedule, model_dim=config.MODEL_DIM))

    loss_fn = nn.CrossEntropyLoss()

    run = wandb.init(entity=config.WANDD_ENTITY, project=config.WANDB_PROJECT)

    for epoch in range(config.NUM_EPOCHS):

        # Training Loop
        for idx, batch in enumerate(train_dataloader):
            encoder_input = batch["input_ids"].to("cuda")
            decoder_input = batch["labels"][..., :-1].to("cuda")
            enc_key_attention_mask = batch["input_attention_mask"].to("cuda")
            dec_key_attention_mask = batch["labels_attention_mask"][..., :-1].to("cuda")

            logits = model(
                encoder_input=encoder_input,
                decoder_input=decoder_input,
                enc_key_attention_mask=enc_key_attention_mask,
                dec_key_attention_mask=dec_key_attention_mask,
            )

            logits = logits.view(-1, logits.size(-1))
            target = batch["labels"][..., 1:].to("cuda").view(-1).long()

            loss = loss_fn(input=logits, target=target)
            run.log({"train-loss": loss.item()})

            loss.backward()

            if idx % config.GRAD_ACCUMULATION_STEP == 0:
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()


        # Validation Loop
        model.eval()

        with torch.no_grad():  # Disable gradient computation for validation
            for batch in val_dataloader:
                encoder_input = batch["input_ids"].to("cuda")
                decoder_input = batch["labels"][..., :-1].to("cuda")
                enc_key_attention_mask = batch["input_attention_mask"].to("cuda")
                dec_key_attention_mask = batch["labels_attention_mask"][..., :-1].to(
                    "cuda"
                )

                logits = model(
                    encoder_input=encoder_input,
                    decoder_input=decoder_input,
                    enc_key_attention_mask=enc_key_attention_mask,
                    dec_key_attention_mask=dec_key_attention_mask,
                )

                logits = logits.view(-1, logits.size(-1))
                target = batch["labels"][..., 1:].to("cuda").view(-1).long()

                val_loss = loss_fn(input=logits, target=target)
                run.log({"val-loss": val_loss.item()})

        model.train()  # Set model back to training mode

        model.push_to_hub(config.MODEL_NAME, commit_message=f"epoch_{epoch}")
        
    run.finish()
