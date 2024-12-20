import torch
import torch.nn as nn

from llm.wmt_data_utils import get_wmt_dataset, get_wmt_dataloaders
from llm.wmt_model import TransformerConfig, Transformer


class TrainerConfig:
    source_lang = "de"
    target_lang = "en"
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    max_len = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adam_beta_1 = 0.91
    adam_beta_2 = 0.98
    adam_eps = 1e-09


transformer_config = TransformerConfig(
    model_dim=512,
    expansion_dim=2048,
    num_heads=8,
    num_blocks=6,
    dropout_rate=0.1,
    vocab_src_size=15000,
    vocab_tgt_size=15000,
    max_seq_len=256,
)


def train_step(batch: tuple[torch.Tensor, torch.Tensor], model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: nn.Module):
    # move the data to the device
    src_tokens, tgt_tokens = batch
    src_tokens = src_tokens.to(trainer_config.device)
    tgt_tokens = tgt_tokens.to(trainer_config.device)
    
    # forward pass and loss calculation
    optimizer.zero_grad()
    output = model(src_tokens, tgt_tokens)
    # softmax to get the logits
    logits = torch.log_softmax(output, dim=-1)
    loss = loss_fn(logits, tgt_tokens)
    loss.backward()
    optimizer.step()


    return model


if __name__ == "__main__":
    trainer_config = TrainerConfig()
    print(trainer_config)

    datasets = get_wmt_dataset(
        name=f"{trainer_config.source_lang}-{trainer_config.target_lang}"
    )
    dataloaders = get_wmt_dataloaders(
        datasets=datasets,
        batch_size=trainer_config.batch_size,
        source_lang=trainer_config.source_lang,
        target_lang=trainer_config.target_lang,
    )
    print(dataloaders)

    print(next(iter(dataloaders["train"])))

    model = Transformer(config=transformer_config)
    model.to(trainer_config.device)
    print(model)

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_config.learning_rate,
        betas=(trainer_config.adam_beta_1, trainer_config.adam_beta_2),
        eps=trainer_config.adam_eps,
    )

    # TODO: add learning rate scheduler


