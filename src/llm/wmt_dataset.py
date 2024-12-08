from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial
import torch
from .tokenizer import get_tokenizers


def collate_fn(batch, source, target, tokenizers):
    zipped_list = [
        (
            sample["translation"][source],
            sample["translation"][target],
        )
        for sample in batch
    ]
    source_batch, target_batch = list(zip(*zipped_list))

    # enable padding for the tokenizers
    # we do the longest sequence length for padding
    tokenizers[source].enable_padding(pad_id=1, pad_token="<pad>")
    tokenizers[target].enable_padding(pad_id=1, pad_token="<pad>")

    # tokenize the batches
    tokenized_source_batch = [token.ids for token in tokenizers[source].encode_batch(source_batch)]
    tokenized_target_batch = [token.ids for token in tokenizers[target].encode_batch(target_batch)]

    # torch tensors
    tokenized_source_batch = torch.tensor(tokenized_source_batch)
    tokenized_target_batch = torch.tensor(tokenized_target_batch)

    return tokenized_source_batch, tokenized_target_batch


def get_dataloader(dataset, batch_size, source_lang, target_lang, tokenizer_path="data"):
    tokenizers = get_tokenizers(f"{source_lang}-{target_lang}", tokenizer_path)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, source=source_lang, target=target_lang, tokenizers=tokenizers),
    )
