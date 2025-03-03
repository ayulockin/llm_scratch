from datasets import load_dataset
import config


def split_translation(row):
    return {"en": row["translation"]["en"], "de": row["translation"]["de"]}


if __name__ == "__main__":
    print("[INFO] loading the train, val, and test dataset ...")
    train_dataset = load_dataset(config.ORIGINAL_DATASET, "de-en", split="train")
    val_dataset = load_dataset(config.ORIGINAL_DATASET, "de-en", split="validation")
    test_dataset = load_dataset(config.ORIGINAL_DATASET, "de-en", split="test")

    train_dataset = train_dataset.map(split_translation, remove_columns=["translation"])
    test_dataset = test_dataset.map(split_translation, remove_columns=["translation"])
    val_dataset = val_dataset.map(split_translation, remove_columns=["translation"])

    train_dataset.push_to_hub(config.DE_EN_SPLIT_DATASET, split="train")
    test_dataset.push_to_hub(config.DE_EN_SPLIT_DATASET, split="test")
    val_dataset.push_to_hub(config.DE_EN_SPLIT_DATASET, split="validation")
