from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
    processors,
)
from datasets import load_dataset
from functools import partial

from huggingface_hub import HfApi
import config


def batch_iterator(dataset, batch_size=100_000):
    for sample in dataset.iter(batch_size):
        texts = sample["en"] + sample["de"]
        yield texts


if __name__ == "__main__":
    dataset = load_dataset(config.DE_EN_SPLIT_DATASET, split="train+test+validation")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=config.VOCAB_SIZE,
        min_frequency=config.MIN_FREQUENCY,
        special_tokens=config.SPECIAL_TOKENS.values(),
    )

    tokenizer.train_from_iterator(
        partial(batch_iterator(dataset=dataset)), trainer=trainer, length=len(dataset)
    )

    tokenizer.save("tokenizer.json")

    api = HfApi()
    api.upload_file(
        path_or_fileobj="tokenizer.json",
        path_in_repo="tokenizer.json",
        repo_id=config.TOKENIZER_ID,
        repo_type="model",
    )
