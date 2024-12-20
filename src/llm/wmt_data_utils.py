from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial
import torch

from typing import Union

from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from datasets import load_dataset
from fire import Fire


def get_wmt_dataset(name: str):
    """Get the dataset for the specified name. This will load the WMT14 dataset.

    Dataset: https://huggingface.co/datasets/wmt/wmt14

    Args:
        name (str): The name of the dataset. The valid names are `cs-en`, `de-en`, `fr-en`, `hi-en`, `ru-en`.

    Returns:
        dict: The `train`, `val` and `test` datasets for the specified name.
    """
    valid_names = [
        "cs-en",
        "de-en",
        "fr-en",
        "hi-en",
        "ru-en",
    ]
    # Handle reversed language pairs (e.g., en-cs -> cs-en)
    reversed_pairs = {
        f"{b}-{a}": f"{a}-{b}" for a, b in [pair.split("-") for pair in valid_names]
    }
    if name in reversed_pairs:
        name = reversed_pairs[name]

    assert (
        name in valid_names
    ), f"Invalid dataset name: {name}. Please choose from {valid_names}."

    # Load the dataset
    train_dataset = load_dataset(
        "wmt/wmt14", name=name, split="train", num_proc=8
    )  # speed up the loading of the dataset
    val_dataset = load_dataset("wmt/wmt14", name=name, split="validation")
    test_dataset = load_dataset("wmt/wmt14", name=name, split="test")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def get_wmt_tokenizers(dataset_name, path="data"):
    """
    Get the tokenizers for the specified dataset name. We have two tokenizers for each language pair. Thus if you want to get the tokenizers for `en-de`, you will get two tokenizers, one for English and one for German.

    Args:
        dataset_name (str): The name of the dataset.
        path (str): The path to the tokenizers.

    Returns:
        dict: The tokenizers for the specified dataset name.
    """
    try:
        tokenizers = {}
        for lang in dataset_name.split("-"):
            tokenizers[lang] = Tokenizer.from_file(f"{path}/{lang}_tokenizer.json")
        return tokenizers
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Tokenizers for {dataset_name} not found. Please train the tokenizers first using `python src/llm/wmt_data_utils.py`."
        )


def decode_wmt_tokens(
    tokenizer, tokens: Union[list[int], list[list[int]]], batch=False
):
    """
    Decode the tokens into a string. The tokenizer passed here should be loaded using `get_tokenizers`.

    Args:
        tokenizer (Tokenizer): The tokenizer.
        tokens (Union[list[int], list[list[int]]]): The tokens to decode.
        batch (bool): Whether the tokens are a batch.

    Returns:
        Union[str, list[str]]: The decoded tokens.
    """
    # TODO: infer batch from the shape of `tokens`
    if batch:
        return tokenizer.decode_batch(tokens)
    else:
        return tokenizer.decode(tokens)


def _collate_fn(batch, source, target, tokenizers):
    """
    Collate function for the dataloader. This will collate the batches into a single tensor.

    Args:
        batch (list): The batch of data.
        source (str): The source language.
        target (str): The target language.
        tokenizers (dict): The tokenizers for the specified dataset name.

    Returns:
        tuple: The collated batches.
    """
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
    tokenized_source_batch = [
        token.ids for token in tokenizers[source].encode_batch(source_batch)
    ]
    tokenized_target_batch = [
        token.ids for token in tokenizers[target].encode_batch(target_batch)
    ]
    # Add start token (<s>) to beginning of each target sequence to shift tokens right by 1.
    tokenized_target_batch = [
        [tokenizers[target].token_to_id("<s>")] + token
        for token in tokenized_target_batch
    ]

    # torch tensors
    tokenized_source_batch = torch.tensor(tokenized_source_batch)
    tokenized_target_batch = torch.tensor(tokenized_target_batch)

    return tokenized_source_batch, tokenized_target_batch


def get_wmt_dataloader(
    dataset,
    batch_size,
    source_lang,
    target_lang,
    tokenizer_path="data",
):
    """
    Get the dataloader for the specified dataset name.

    Args:
        dataset (str): The name of the dataset.
        batch_size (int): The batch size.
        source_lang (str): The source language.
        target_lang (str): The target language.
        tokenizer_path (str): The path to the tokenizers.

    Returns:
        DataLoader: The dataloader for the specified dataset name.
    """
    tokenizers = get_wmt_tokenizers(f"{source_lang}-{target_lang}", tokenizer_path)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(
            _collate_fn, source=source_lang, target=target_lang, tokenizers=tokenizers
        ),
    )


def create_separate_wmt_tokenizers(
    vocab_size: int = 30000,
    min_frequency: int = 2,
    special_tokens: list[str] = ["<s>", "<pad>", "</s>", "<unk>", "<sep>"],
    batch_size: int = 1000,
    dataset_name: str = "de-en",
):
    """
    Create and train two separate tokenizers for Source and Target languages from the WMT dataset.

    Args:
        vocab_size (int): Total vocabulary size for the tokenizers. If vocab size is 30000, source and target tokenizers will have 15000 tokens each.
        min_frequency (int): Minimum frequency for a token to be included in the vocabulary.
        special_tokens (list): List of special tokens to be included in the vocabulary.
        batch_size (int): Batch size for the iterator.
        dataset_name (str): The name of the dataset. Valid names are `de-en`, `fr-en`, `cs-en`, `hi-en`, `ru-en`.

    Returns:
        Tuple[Tokenizer, Tokenizer]: The trained Source and Target tokenizers.
    """
    # Load training dataset
    print(f"Loading dataset WMT14: {dataset_name}...")
    ds = get_wmt_dataset(dataset_name)["train"]

    # Verify that the specified languages exist in the dataset
    langs = dataset_name.split("-")
    print(f"Training tokenizers for {langs}...")

    for lang in langs:
        print(f"Training {lang} tokenizer...")

        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Define normalization pipeline
        normalizer_list = normalizers.Sequence(
            [
                normalizers.NFD(),
                normalizers.StripAccents(),  # Remove accents
            ]
        )
        tokenizer.normalizer = normalizer_list

        # Set pre-tokenizer and decoder to ByteLevel
        # ByteLevel preserves the whitespace spacing between words.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
        tokenizer.decoder = decoders.ByteLevel()

        # Define BPE trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size // 2,  # vocab size for each language
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        # Define batch iterator for Source and Target
        def batch_iterator(lang):
            tok_dataset = ds.select_columns("translation")
            for batch in tok_dataset.iter(batch_size):
                batch = batch["translation"]
                yield [ex[lang] for ex in batch]

        # Train tokenizers
        tokenizer.train_from_iterator(
            batch_iterator(lang), trainer=trainer, length=len(ds)
        )

        # Save the trained tokenizers
        tokenizer.save(f"data/{lang}_tokenizer.json")

        del tokenizer


if __name__ == "__main__":
    Fire(create_separate_wmt_tokenizers)
