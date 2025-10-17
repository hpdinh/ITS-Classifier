from typing import List, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class TicketDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ) -> None:
        self.texts: List[str] = dataframe['combined'].tolist()
        self.labels: List[int] = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
