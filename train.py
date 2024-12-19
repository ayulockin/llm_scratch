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


def train_step(data, modelxs):
    pass


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

    model = Transformer(config=transformer_config)
    print(model)

    for epoch in range(trainer_config.num_epochs):
        for data in dataloaders["train"]:
            train_step(data, model)
