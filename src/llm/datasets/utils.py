from datasets import load_dataset


def download_dataset(
        dataset_name: str,
        subset: str | None = None,
        num_proc: int = 8,
        cache_dir: str | None = "data/"
) -> dict:
    """Downloads a dataset from HuggingFace datasets.
    
    Args:
        dataset_name: Name of the dataset to download
        subset: Optional subset/configuration of the dataset
        
    Returns:
        The downloaded dataset dictionary
    """
    if subset:
        return load_dataset(dataset_name, subset, num_proc=num_proc, cache_dir=cache_dir)
    return load_dataset(dataset_name, num_proc=num_proc, cache_dir=cache_dir)
