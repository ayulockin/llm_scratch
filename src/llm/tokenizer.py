from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
)
from datasets import load_dataset


def get_dataset(name):
    # Load the dataset
    train_dataset = load_dataset("wmt/wmt14", name=name, split="train")
    val_dataset = load_dataset("wmt/wmt14", name=name, split="validation")
    test_dataset = load_dataset("wmt/wmt14", name=name, split="test")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def get_tokenizers(dataset_name, path="data"):
    """
    Get the tokenizers for the specified dataset name.
    """
    try:
        tokenizers = {}
        for lang in dataset_name.split("-"):
            tokenizers[lang] = Tokenizer.from_file(f"{path}/{lang}_tokenizer.json")
        return tokenizers
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Tokenizers for {dataset_name} not found. Please train the tokenizers first using python src/llm/train_tokenizer.py."
        )


def create_separate_wmt_tokenizers(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=None,
    batch_size=1000,
    dataset_name="de-en",
):
    """
    Create and train two separate tokenizers for English and German from the WMT dataset.

    Args:
        vocab_size (int): Total vocabulary size for the tokenizers. If vocab size is 30000, english and german tokenizers will have 15000 tokens each.
        min_frequency (int): Minimum frequency for a token to be included in the vocabulary.
        special_tokens (list): List of special tokens to be included in the vocabulary.
        batch_size (int): Batch size for the iterator.
        en_tokenizer_path (str): Path to save the trained English tokenizer.
        de_tokenizer_path (str): Path to save the trained German tokenizer.

    Returns:
        Tuple[Tokenizer, Tokenizer]: The trained English and German tokenizers.
    """
    if special_tokens is None:
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<sep>"]

    # Load training dataset
    print(f"Loading dataset WMT14:{dataset_name}...")
    ds = get_dataset(dataset_name)["train"]

    # Verify that the specified languages exist in the dataset
    langs = dataset_name.split("-")
    print(f"Training tokenizers for {langs}...")

    for lang in langs:
        print(f"Training {lang} tokenizer...")

        # Initialize tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Define normalization pipeline
        normalizer_list = normalizers.Sequence([
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

        # Define batch iterator for English and German
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
    create_separate_wmt_tokenizers()
