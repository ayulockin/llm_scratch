import os
import wandb
from tqdm import tqdm

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from llm.wmt_data_utils import get_wmt_dataset, get_wmt_dataloaders
from llm.wmt_model import TransformerConfig, Transformer

from tokenizers import Tokenizer


class TrainerConfig:
    source_lang = "de"
    target_lang = "en"
    pad_id = 0
    pad_token = "[PAD]"
    batch_size = 64
    num_epochs = 2
    learning_rate = 1e-4
    max_seq_len = 256
    label_smoothing = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adam_beta_1 = 0.91
    adam_beta_2 = 0.98
    adam_eps = 1e-09
    model_dim = 512
    warmup_steps = 4000
    expansion_dim = 2048
    num_heads = 8
    num_blocks = 6
    dropout_rate = 0.1
    tokenizer_id = "llm-scratch/wmt-14-en-de-tok"
    use_mixed_precision=True
    checkpoint_dir="models/"


def train_step(
    batch: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    trainer_config: TrainerConfig,
) -> float:
    model.train()
    # move the data to the device
    encoder_input_ids = batch["source_input"]["input_ids"].to(trainer_config.device)
    decoder_input_ids = batch["target_input"]["input_ids"].to(trainer_config.device)
    encoder_self_attention_mask = batch["source_input"]["self_attention_mask"].to(trainer_config.device)
    decoder_self_attention_mask = batch["target_input"]["self_attention_mask"].to(trainer_config.device)
    decoder_cross_attention_mask = batch["target_input"]["cross_attention_mask"].to(trainer_config.device)

    # forward pass and loss calculation
    optimizer.zero_grad()

    with torch.autocast(device_type=trainer_config.device, dtype=torch.float16, enabled=trainer_config.use_mixed_precision):
        logits = model(
            encoder_input=encoder_input_ids,
            decoder_input=decoder_input_ids,
            encoder_self_attention_mask=encoder_self_attention_mask,
            decoder_self_attention_mask=decoder_self_attention_mask,
            decoder_cross_attention_mask=decoder_cross_attention_mask,
        )

        batch_size, _, vocab_size = logits.shape   # batch, decoder_seq_len, vocab_size
        logits_flat = logits.view(-1, vocab_size)  # [1 * decoder_seq_len, vocab_size]
        targets_flat = decoder_input_ids.view(-1)  # [1 * decoder_seq_len]

        loss = loss_fn(logits_flat, targets_flat)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


def valid_step(
    batch: dict,
    model: nn.Module,
    loss_fn: nn.Module,
    trainer_config: TrainerConfig,
) -> float:
    """
    Similar to train_step, but no gradient updates. Used for validation.
    Returns the loss for this batch (as a float).
    """
    model.eval()
    
    with torch.no_grad():
        # Move data to device
        encoder_input_ids = batch["source_input"]["input_ids"].to(trainer_config.device)
        decoder_input_ids = batch["target_input"]["input_ids"].to(trainer_config.device)
        encoder_self_attention_mask = batch["source_input"]["self_attention_mask"].to(trainer_config.device)
        decoder_self_attention_mask = batch["target_input"]["self_attention_mask"].to(trainer_config.device)
        decoder_cross_attention_mask = batch["target_input"]["cross_attention_mask"].to(trainer_config.device)

        # Forward pass
        logits = model(
            encoder_input=encoder_input_ids,
            decoder_input=decoder_input_ids,
            encoder_self_attention_mask=encoder_self_attention_mask,
            decoder_self_attention_mask=decoder_self_attention_mask,
            decoder_cross_attention_mask=decoder_cross_attention_mask,
        )

        # Flatten for loss calculation
        batch_size, _, vocab_size = logits.shape  # [batch, seq_len, vocab_size]
        logits_flat = logits.view(-1, vocab_size) # [batch * seq_len, vocab_size]
        targets_flat = decoder_input_ids.view(-1) # [batch * seq_len]

        # Compute loss (no backward or optimizer step for validation)
        loss = loss_fn(logits_flat, targets_flat)

    return loss.item()


if __name__ == "__main__":
    run = wandb.init(
        project="wmt14-de-en",
        entity="llm-scratch",
    )

    trainer_config = TrainerConfig()
    wandb.config.update({"trainer_config": vars(trainer_config)})

    # Get the dataset
    datasets = get_wmt_dataset(
        name=f"{trainer_config.source_lang}-{trainer_config.target_lang}"
    )

    # Additionally get the tokenizer to get the vocab_size
    tokenizer = Tokenizer.from_pretrained(trainer_config.tokenizer_id)
    tokenizer.enable_padding(pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=trainer_config.max_seq_len)

    # Get the dataloaders
    dataloaders = get_wmt_dataloaders(
        datasets=datasets,
        batch_size=trainer_config.batch_size,
        source_lang=trainer_config.source_lang,
        target_lang=trainer_config.target_lang,
        tokenizer=tokenizer,
    )

    # Build the transformer config
    transformer_config = TransformerConfig(
        model_dim=trainer_config.model_dim,
        expansion_dim=trainer_config.expansion_dim,
        num_heads=trainer_config.num_heads,
        num_blocks=trainer_config.num_blocks,
        dropout_rate=trainer_config.dropout_rate,
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=trainer_config.max_seq_len,
    )
    wandb.config.update({"transformer_config": transformer_config.__dict__})

    # Build the model
    model = Transformer(config=transformer_config)
    model.to(trainer_config.device)
    print(model)

    # Build the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_config.learning_rate,
        betas=(trainer_config.adam_beta_1, trainer_config.adam_beta_2),
        eps=trainer_config.adam_eps,
    )

    # loss function
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=trainer_config.label_smoothing,
        ignore_index=trainer_config.pad_id,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler(trainer_config.device, enabled=trainer_config.use_mixed_precision)

    # Train the model
    for epoch in range(trainer_config.num_epochs):
        # --------------------
        # Training
        # --------------------
        progress_bar = tqdm(dataloaders["train"], desc=f"Epoch {epoch+1} - Training", leave=True)
        total_train_loss = 0.0

        for idx, batch in enumerate(progress_bar):
            step_train_loss = train_step(batch, model, optimizer, loss_fn, scaler, trainer_config)
            total_train_loss += step_train_loss

            wandb.log({"train_loss": step_train_loss})
            progress_bar.set_postfix(loss=step_train_loss)

            # if (idx + 1) % 100 == 0:
            #     avg_loss = total_train_loss / (idx + 1)
            #     print(f"  Step {idx+1} - Avg Train Loss: {avg_loss:.4f}")

            if (idx+1) % 100 == 0:
                break

        # Final average for the epoch
        steps_this_epoch = idx + 1  # because idx is zero-based
        epoch_avg_train_loss = total_train_loss / steps_this_epoch
        print(f"Epoch {epoch+1} - Average Train Loss: {epoch_avg_train_loss:.4f}")
        wandb.log({"epoch/train_loss": epoch_avg_train_loss})

        # --------------------
        # Validation
        # --------------------
        progress_bar = tqdm(dataloaders["val"], desc=f"Epoch {epoch+1} - Validation", leave=True)
        total_val_loss = 0.0

        for idx, batch in enumerate(progress_bar):
            step_val_loss = valid_step(batch, model, loss_fn, trainer_config)
            total_val_loss += step_val_loss

            progress_bar.set_postfix(loss=step_val_loss)

            if (idx+1) % 1000 == 0:
                break

        # Final average for the epoch
        steps_this_epoch = idx + 1  # because idx is zero-based
        epoch_avg_val_loss = total_val_loss / steps_this_epoch
        print(f"Epoch {epoch+1} - Average Validation Loss: {epoch_avg_val_loss:.4f}")
        wandb.log({"epoch/val_loss": epoch_avg_val_loss})

        # --------------------
        # Checkpointing
        # --------------------
        # 1) Always save a checkpoint each epoch
        ckpt_path = os.path.join(trainer_config.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': epoch_avg_train_loss,
            'val_loss': epoch_avg_val_loss,
        }, ckpt_path)
