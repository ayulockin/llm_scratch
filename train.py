from llm import get_dataset, get_dataloader

if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_dataset(
        name="fr-en",
    )

    print(train_ds[0])

    train_dl = get_dataloader(
        dataset=train_ds, batch_size=2, source_lang="en", target_lang="fr"
    )

    print(next(iter(train_dl)))
