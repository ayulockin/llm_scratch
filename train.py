import torch
import torch.nn as nn

from llm.wmt_data_utils import get_wmt_dataset, get_wmt_dataloaders, get_wmt_tokenizers
from llm.wmt_model import TransformerConfig, Transformer


class TrainerConfig:
    source_lang = "de"
    target_lang = "en"
    batch_size = 32
    num_epochs = 10
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
    num_blocks = 6
    dropout_rate = 0.1


def train_step(batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module):
    model.train()
    # move the data to the device
    src_tokens, tgt_tokens, encoder_attention_mask, decoder_attention_mask = batch
    print(f"{src_tokens.shape=}")
    print(f"{tgt_tokens.shape=}")
    print(f"{encoder_attention_mask.shape=}")
    print(f"{decoder_attention_mask.shape=}")

    src_tokens = src_tokens.to(trainer_config.device)
    tgt_tokens = tgt_tokens.to(trainer_config.device)
    encoder_attention_mask = encoder_attention_mask.to(trainer_config.device)
    decoder_attention_mask = decoder_attention_mask.to(trainer_config.device)
    
    # forward pass and loss calculation
    optimizer.zero_grad()
    output = model(src_tokens, tgt_tokens, encoder_attention_mask, decoder_attention_mask)

    # softmax to get the logits
    logits = torch.softmax(output, dim=-1)
    pred_tokens = torch.argmax(logits, dim=-1)
    loss = loss_fn(pred_tokens, tgt_tokens)
    loss.backward()
    optimizer.step()

    return model, {"loss": loss}


def eval_step(batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module, loss_fn: nn.Module):
    model.eval()

    # move the data to the device
    src_tokens, tgt_tokens, encoder_attention_mask, decoder_attention_mask = batch
    src_tokens = src_tokens.to(trainer_config.device)
    tgt_tokens = tgt_tokens.to(trainer_config.device)
    encoder_attention_mask = encoder_attention_mask.to(trainer_config.device)
    decoder_attention_mask = decoder_attention_mask.to(trainer_config.device)
    
    # forward pass and loss calculation
    output = model(src_tokens, tgt_tokens, encoder_attention_mask, decoder_attention_mask)

    # softmax to get the logits
    logits = torch.softmax(output, dim=-1)
    loss = loss_fn(logits, tgt_tokens)

    return model, {"eval_loss": loss}


if __name__ == "__main__":
    trainer_config = TrainerConfig()

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
    loss_fn = nn.CrossEntropyLoss(label_smoothing=trainer_config.label_smoothing)

    # Train the model
    for epoch in range(trainer_config.num_epochs):
        train_loss = 0
        eval_loss = 0
        for idx, batch in enumerate(dataloaders["train"]):
            model, results = train_step(batch, model, optimizer, loss_fn)
            train_loss += results["loss"] / len(batch)
            if idx % 10 == 0:
                print(f"Epoch {epoch+1} - Train Loss: {train_loss}")

            if idx == 20:
                break

        for idx, batch in enumerate(dataloaders["eval"]):
            model, results = eval_step(batch, model, loss_fn)
            eval_loss += results["eval_loss"] / len(batch)
            if idx % 10 == 0:
                print(f"Epoch {epoch+1} - Eval Loss: {eval_loss}")

            if idx == 20:
                break

        print(f"Epoch {epoch+1} - Train Loss: {train_loss}, Eval Loss: {eval_loss}")
