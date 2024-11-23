from torch.utils.data import Dataset
from src.llm.datasets.utils import download_dataset


class WMT14EnDeDataset(Dataset):
    def __init__(self, split='train'):
        # Load the specified split of the dataset
        assert split in ['train', 'validation', 'test']
        self.dataset = download_dataset('wmt/wmt14', 'de-en', split=split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Fetch the sample at the given index
        sample = self.dataset[idx]
        # Return the English and German strings
        return sample['translation']['en'], sample['translation']['de']
