from datasets import load_dataset
from typing import Dict, Any, Optional


def download_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    num_proc: int = 8,
    cache_dir: Optional[str] = "data/"
) -> Dict[str, Any]:
    """Downloads a dataset from HuggingFace datasets.

    Args:
        dataset_name (str): Name of the dataset to download.
        subset (Optional[str]): Optional subset/configuration of the dataset.
        split (Optional[str]): Optional split of the dataset to load.
        num_proc (int): Number of processes to use (default is 8).
        cache_dir (Optional[str]): Directory to cache the dataset (default is "data/").

    Returns:
        Dict[str, Any]: The downloaded dataset object.
    """
    # Prepare arguments for load_dataset
    load_args = {
        "path": dataset_name,
        "num_proc": num_proc,
        "cache_dir": cache_dir
    }
    if subset:
        load_args["name"] = subset
    if split:
        load_args["split"] = split

    # Load the dataset with the prepared arguments
    return load_dataset(**load_args)
