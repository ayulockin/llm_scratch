from torch.utils.data import DataLoader
from datasets import load_dataset
from functools import partial


def collate_fn(batch, source, target):
    zipped_list = [
        (sample["translation"][source], sample["translation"][target])
        for sample in batch
    ]
    source_batch, target_batch = list(zip(*zipped_list))
    return source_batch, target_batch


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


def get_dataloader(dataset, batch_size, source_lang, target_lang):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, source=source_lang, target=target_lang),
    )
