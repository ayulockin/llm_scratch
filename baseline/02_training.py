import config as cfg

import torch
from torch import nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer
from datasets import load_dataset

import math
import wandb
from tqdm import tqdm
from rich.console import Console
from rich import table
from rich import progress
from functools import partial
from jaxtyping import Integer, Float, Bool
from dataclasses import dataclass


################################################################################
# Embeddings
################################################################################
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


################################################################################
# FFN
################################################################################
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


################################################################################
# Attention
################################################################################
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_k, dim_v, model_dim):
        super().__init__()
        self.dim_k = dim_k

        self.W_q = nn.Linear(model_dim, dim_k, bias=False)
        self.W_k = nn.Linear(model_dim, dim_k, bias=False)
        self.W_v = nn.Linear(model_dim, dim_v, bias=False)

    def forward(
        self,
        query: Float[Tensor, "batch q_seq_len model_dim"],
        key: Float[Tensor, "batch k_seq_len model_dim"],
        value: Float[Tensor, "batch v_seq_len model_dim"],
        padding_mask: Integer[Tensor, "batch k_seq_len"],
        is_causal: Bool = False,
    ):
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        similarity = torch.einsum("bqd, bkd -> bqk", query, key)
        scaled_similarity = torch.divide(similarity, math.sqrt(self.dim_k))

        # This is an interesting bit, the scaled_similarity is of shape (b, q, k)
        # The causal mask is of the shape (b, q, k)
        # The padding mask comes with the shape of (b, k) and we need to add an extra dim
        # in the first axis to simulate the query dim making it (b, 1, k)
        # this is done in order to match the shape with the scaled similarity
        padding_mask = padding_mask.unsqueeze(1)  # 1=attend 0=pad

        if is_causal:
            # create the causal mask and use the padding mask to create
            # the ultimate mask that is use before softmax
            causal_mask = torch.tril(
                torch.ones_like(scaled_similarity),
            )  # 1=attend 0=mask
            mask = torch.logical_and(
                padding_mask, causal_mask
            )  # True=value False=masked
        else:
            mask = padding_mask

        # Masked fill work by filling the True positions with a value
        # Here we would need to invert the mask so that we position the negative infinity
        # correctly.
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
        padding_mask: Bool[Tensor, "batch k_seq_len"],
        is_causal: bool = False,
    ) -> Float[Tensor, "batch seq_len model_dim"]:
        outputs = list()
        for head in self.projection_heads:
            output = head(
                query=query,
                key=key,
                value=value,
                padding_mask=padding_mask,
                is_causal=is_causal,
            )
            outputs.append(output)

        outputs = torch.concat(outputs, dim=-1)
        outputs = self.W_out(outputs)
        return outputs


################################################################################
# Encoder
################################################################################
class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.multi_head_self_attention = MultiHeadAttention(
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
        padding_mask: Bool[Tensor, "batch enc_seq_len"],
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:
        residual = inputs

        x = self.multi_head_self_attention(
            query=inputs,
            key=inputs,
            value=inputs,
            padding_mask=padding_mask,
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
        padding_mask: Bool[Tensor, "batch enc_seq_len"],
    ) -> Float[Tensor, "batch enc_seq_len model_dim"]:
        # We pass the input through each encoder layer and return the final output
        for encoder_layer in self.encoder_layers:
            encoder_input = encoder_layer(encoder_input, padding_mask)

        return encoder_input


################################################################################
# Decoder
################################################################################
class DecoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.masked_multi_head_self_attention = MultiHeadAttention(
            model_dim=model_dim, num_heads=num_heads
        )
        self.multi_head_cross_attention = MultiHeadAttention(
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
        enc_padding_mask: Bool[Tensor, "batch enc_seq_len"],
        dec_padding_mask: Bool[Tensor, "batch dec_seq_len"],
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:
        residual = decoder_input

        x = self.masked_multi_head_self_attention(
            query=decoder_input,
            key=decoder_input,
            value=decoder_input,
            is_causal=True,
            padding_mask=dec_padding_mask,
        )
        x = self.dropout1(x) + residual
        x = self.layer_norm1(x)

        residual = x

        x = self.multi_head_cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            padding_mask=enc_padding_mask,
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
        enc_padding_mask: Bool[Tensor, "batch enc_seq_len"],
        dec_padding_mask: Bool[Tensor, "batch dec_seq_len"],
    ) -> Float[Tensor, "batch dec_seq_len model_dim"]:
        x = decoder_input
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(
                decoder_input=x,
                encoder_output=encoder_output,
                enc_padding_mask=enc_padding_mask,
                dec_padding_mask=dec_padding_mask,
            )
        return x


################################################################################
# Transformer
################################################################################
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
        self.encoder_embedding = Embedding(
            model_dim=config.model_dim, vocab_size=config.vocab_size
        )
        self.decoder_embedding = Embedding(
            model_dim=config.model_dim, vocab_size=config.vocab_size
        )
        # the positional encoding layer does not have a learnable parameter
        # hence we can reuse it for the encoder and the decoder
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
        self.lm_head.weight = self.decoder_embedding.embedding.weight

    def forward(
        self,
        encoder_input: Float[Tensor, "batch enc_seq_len"],
        decoder_input: Float[Tensor, "batch dec_seq_len"],
        enc_padding_mask: Bool[Tensor, "batch enc_seq_len"],
        dec_padding_mask: Bool[Tensor, "batch dec_seq_len"],
    ) -> Float[Tensor, "batch dec_seq_len vocab_tgt_size"]:
        # Embed the source input and add positional encoding
        encoder_input = self.encoder_embedding(tokens=encoder_input)
        encoder_input = self.positional_encoding(inputs=encoder_input)

        # Embed the target input and add positional encoding
        decoder_input = self.decoder_embedding(tokens=decoder_input)
        decoder_input = self.positional_encoding(decoder_input)

        # Encode the source input
        encoder_output = self.encoder(
            encoder_input=encoder_input,
            padding_mask=enc_padding_mask,
        )

        # Decode the target input
        decoder_output = self.decoder(
            encoder_output=encoder_output,
            decoder_input=decoder_input,
            enc_padding_mask=enc_padding_mask,
            dec_padding_mask=dec_padding_mask,
        )

        # Get the logits for the next token
        logits = self.lm_head(decoder_output)
        return logits

    def generate(self,):
        pass


################################################################################
# Training
################################################################################
def collate_fn(batch):
    source = [sample["de"] for sample in batch]
    target = [sample["en"] for sample in batch]
    return source, target


def learning_rate_schedule(step_num, model_dim, warmup_steps):
    if step_num == 0:
        step_num = 1  # Prevent division by zero
    scale = model_dim**-0.5
    step_term = min(step_num**-0.5, step_num * warmup_steps**-1.5)
    return scale * step_term


def calculate_total_tokens(dataset, tokenizer):
    # We just want to count the number of tokens for the source of the dataset.
    temp_dataloader = DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
    )
    total_tokens = 0
    for batch in tqdm(temp_dataloader):
        source, _ = batch
        source_inputs = tokenizer(text=source)["input_ids"]
        # Since the tokenizer is adding the bos and eos tokens, we need to subtract them
        total_tokens += len(sum(source_inputs, [])) - 256*2
    return total_tokens


if __name__ == "__main__":
    console = Console()

    console.print("[bold green]Loading dataset from hub...[/bold green]")
    train_dataset = load_dataset(cfg.DE_EN_SPLIT_DATASET, split="train")
    val_dataset = load_dataset(cfg.DE_EN_SPLIT_DATASET, split="validation")

    console.print("[bold green]Loading tokenizer from hub...[/bold green]")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.TOKENIZER_ID,
        add_bos_token=True,
        add_eos_token=True,
        legacy=True,  # explaination: https://github.com/huggingface/transformers/pull/24565
    )
    tokenizer.pad_token = tokenizer.eos_token

    console.print("[bold green]Building data loader...[/bold green]")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    if cfg.CALCULATE_TOTAL_TOKENS:
        with console.status("[bold yellow]Calculating total tokens in dataset...[/bold yellow]"):
            total_tokens = calculate_total_tokens(train_dataset, tokenizer)
            console.print(f"[bold cyan]Total tokens in dataset: {total_tokens:,}[/bold cyan]")

    console.print("[bold green]Building the model...[/bold green]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = TransformerConfig(
        model_dim=cfg.MODEL_DIM,
        expansion_dim=cfg.EXPANSION_DIM,
        num_heads=cfg.NUM_HEADS,
        num_blocks=cfg.NUM_HEADS,
        dropout_rate=cfg.DROPOUT_RATE,
        max_seq_len=cfg.MAX_SEQ_LENGTH,
        vocab_size=tokenizer.vocab_size,
    )
    model = Transformer(model_config).to(device)

    model.positional_encoding.position_encoding = (
        model.positional_encoding.position_encoding.to(device)
    )

    # Calculate the total number of steps needed to reach cfg.NUM_TOKENS
    tokens_per_epoch = cfg.TOTAL_TOKENS_IN_DATASET / len(train_dataloader)
    console.print(f"[bold cyan]Tokens per step (approx): {tokens_per_epoch:,}[/bold cyan]")
    total_steps = cfg.NUM_TOKENS / tokens_per_epoch
    warmup_steps = int(cfg.WARMUP_PERCENTAGE * total_steps)

    optimizer = Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        betas=cfg.BETAS,
        eps=cfg.EPSILON,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=partial(learning_rate_schedule, model_dim=cfg.MODEL_DIM, warmup_steps=warmup_steps)
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    run = wandb.init(entity=cfg.WANDD_ENTITY, project=cfg.WANDB_PROJECT)
    
    wandb.define_metric("train_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("val_step")
    wandb.define_metric("val/*", step_metric="val_step")

    with progress.Progress(
        progress.TextColumn("[bold blue]{task.description}[/bold blue]"),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TextColumn("•"),
        progress.TimeRemainingColumn(),
    ) as progress:
        token_task = progress.add_task(
            "Training progress", total=cfg.NUM_TOKENS
        )

        idx = 0
        num_tokens = 0
        train_step = 0
        val_step = 0
        
        train_iter = iter(train_dataloader)

        while num_tokens < cfg.NUM_TOKENS:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            source, target = batch

            # inputs now have input_ids and attention_mask
            # input_ids are tokens
            # attention_mask is the mask for padding 1=attend 0=not_attend
            # notice the padding and truncation -- we pad a batch to the max length, but also truncate is something overflows
            source_inputs = tokenizer(
                text=source,
                padding=True,
                return_tensors="pt",
                padding_side="right",
                truncation=True,
                max_length=cfg.MAX_SEQ_LENGTH,
            ).to(device)
            target_inputs = tokenizer(
                text=target,
                padding=True,
                return_tensors="pt",
                padding_side="right",
                truncation=True,
                max_length=cfg.MAX_SEQ_LENGTH,
            ).to(device)

            # idx is the number of batches processed needed for gradient accumulation
            idx += 1
            # We count the number of tokens for the source of the batch
            batch_tokens = source_inputs["input_ids"].ne(tokenizer.pad_token_id).sum().item()
            num_tokens += batch_tokens
            progress.update(token_task, advance=batch_tokens)

            logits = model(
                encoder_input=source_inputs["input_ids"],
                decoder_input=target_inputs["input_ids"][:, 0:-1],
                enc_padding_mask=source_inputs["attention_mask"],
                dec_padding_mask=target_inputs["attention_mask"][:, 0:-1],
            )
            logits = logits.view(-1, logits.size(-1))

            labels = (target_inputs["input_ids"] * target_inputs["attention_mask"].bool()) + (
                -100 * torch.logical_not(target_inputs["attention_mask"].bool())
            )
            labels = labels[:, 1:].flatten()

            loss = loss_fn(input=logits, target=labels)

            run.log({"train/loss": loss.item(), "train_step": train_step})
            train_step += 1

            loss.backward()

            if idx % cfg.GRAD_ACCUMULATION_STEP == 0:
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

            # Validation Loop
            if idx % cfg.VALIDATION_STEP == 0:
                model.eval()

                with torch.no_grad():  # Disable gradient computation for validation
                    for batch in tqdm(val_dataloader, desc="Validation progress"):
                        source, target = batch
                        source_inputs = tokenizer(
                            text=source,
                            padding=True,
                            return_tensors="pt",
                            padding_side="right",
                            truncation=True,
                            max_length=cfg.MAX_SEQ_LENGTH,
                        ).to(device)

                        target_inputs = tokenizer(
                            text=target,
                            padding=True,
                            return_tensors="pt",
                            padding_side="right",
                            truncation=True,
                            max_length=cfg.MAX_SEQ_LENGTH,
                        ).to(device)

                        logits = model(
                            encoder_input=source_inputs["input_ids"],
                            decoder_input=target_inputs["input_ids"][:, 0:-1],
                            enc_padding_mask=source_inputs["attention_mask"],
                            dec_padding_mask=target_inputs["attention_mask"][:, 0:-1],
                        )
                        logits = logits.view(-1, logits.size(-1))

                        labels = (target_inputs["input_ids"] * target_inputs["attention_mask"].bool()) + (
                            -100 * torch.logical_not(target_inputs["attention_mask"].bool())
                        )
                        labels = labels[:, 1:].flatten()

                        val_loss = loss_fn(input=logits, target=labels)
                        run.log({"val/loss": val_loss.item(), "val_step": val_step})
                        val_step += 1

                model.train()  # Set model back to training mode

    # Save the model
    console.print("[bold green]Training complete! Saving final model...[/bold green]")
    model.push_to_hub(cfg.MODEL_NAME, commit_message=f"epoch_{epoch}")
    tokenizer.push_to_hub(cfg.MODEL_NAME)

    run.finish()
