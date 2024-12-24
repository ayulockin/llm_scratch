import wandb

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn

from llm.wmt_data_utils import get_wmt_dataset, get_wmt_dataloaders, get_wmt_tokenizers
from llm.wmt_model import TransformerConfig, Transformer


class TrainerConfig:
    source_lang = "de"
    target_lang = "en"
    pad_id = 1
    pad_token = "<pad>"
    batch_size = 16
    num_epochs = 2
    learning_rate = 1e-4
    max_len = 256
    label_smoothing = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adam_beta_1 = 0.91
    adam_beta_2 = 0.98
    adam_eps = 1e-09
    model_dim = 512
    expansion_dim = 2048
    num_heads = 8
    num_blocks = 3
    dropout_rate = 0.1


def train_step(batch: dict, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module):
    model.train()
    # move the data to the device
    encoder_input_ids = batch["source_input"]["input_ids"].to(trainer_config.device)
    decoder_input_ids = batch["target_input"]["input_ids"].to(trainer_config.device)
    encoder_self_attention_mask = batch["source_input"]["self_attention_mask"].to(trainer_config.device)
    decoder_self_attention_mask = batch["target_input"]["self_attention_mask"].to(trainer_config.device)
    decoder_cross_attention_mask = batch["target_input"]["cross_attention_mask"].to(trainer_config.device)

    # forward pass and loss calculation
    optimizer.zero_grad()
    logits = model(
        encoder_input=encoder_input_ids,
        decoder_input=decoder_input_ids,
        encoder_self_attention_mask=encoder_self_attention_mask,
        decoder_self_attention_mask=decoder_self_attention_mask,
        decoder_cross_attention_mask=decoder_cross_attention_mask,
    )

    batch_size, _, vocab_size = logits.shape  # batch, decoder_seq_len, vocab_size
    logits_flat = logits.view(-1, vocab_size)  # [1 * decoder_seq_len, vocab_size]
    targets_flat = decoder_input_ids.view(-1)  # [1 * decoder_seq_len]

    loss = loss_fn(logits_flat, targets_flat)
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    run = wandb.init(
        project="wmt14-de-en",
        entity="llm-scratch",
    )

    trainer_config = TrainerConfig()
    wandb.config.update(trainer_config.__dict__)

    # Get the dataset
    datasets = get_wmt_dataset(
        name=f"{trainer_config.source_lang}-{trainer_config.target_lang}"
    )

    # Get the dataloaders
    dataloaders = get_wmt_dataloaders(
        datasets=datasets,
        batch_size=trainer_config.batch_size,
        source_lang=trainer_config.source_lang,
        target_lang=trainer_config.target_lang,
    )

    # Additionally get the tokenizer to get the vocab_size
    tokenizers = get_wmt_tokenizers(
        dataset_name=f"{trainer_config.source_lang}-{trainer_config.target_lang}"
    )

    # Build the transformer config
    transformer_config = TransformerConfig(
        model_dim=trainer_config.model_dim,
        expansion_dim=trainer_config.expansion_dim,
        num_heads=trainer_config.num_heads,
        num_blocks=trainer_config.num_blocks,
        dropout_rate=trainer_config.dropout_rate,
        vocab_src_size=tokenizers[trainer_config.source_lang].get_vocab_size(),
        vocab_tgt_size=tokenizers[trainer_config.target_lang].get_vocab_size(),
        max_seq_len=trainer_config.max_len,
    )
    wandb.config.update(transformer_config.__dict__)

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

    # loss function (guessing this is right)
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=trainer_config.label_smoothing,
        ignore_index=trainer_config.pad_id,
    )

    # Train the model
    for epoch in range(trainer_config.num_epochs):
        print(f"Epoch {epoch+1} - Training...")
        total_train_loss = 0.0
        total_val_loss = 0.0

        for idx, batch in enumerate(dataloaders["train"]):
            step_loss = train_step(batch, model, optimizer, loss_fn)
            total_train_loss += step_loss

            wandb.log({"train_loss": step_loss})

            # Print every 10 steps (running average)
            if (idx + 1) % 100 == 0:
                avg_loss = total_train_loss / (idx + 1)
                print(f"  Step {idx+1} - Avg Train Loss: {avg_loss:.4f}")

        # Final average for the epoch
        steps_this_epoch = idx + 1  # because idx is zero-based
        epoch_avg_loss = total_train_loss / steps_this_epoch
        print(f"Epoch {epoch+1} - Average Train Loss: {epoch_avg_loss:.4f}")
