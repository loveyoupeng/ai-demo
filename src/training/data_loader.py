import numpy as np
from typing import Tuple, Iterator
from model.transformer import Transformer
from loss import CrossEntropyLoss
from tokenizer.char_tokenizer import CharTokenizer

class TextDataLoader:
    """
    A simple data loader that provides batches of tokenized text.
    """
    def __init__(
        self, 
        text: str, 
        tokenizer: CharTokenizer, 
        batch_size: int, 
        seq_len: int
    ):
        self.text = text
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.encoded = self.tokenizer.encode(text)
        self.num_samples = len(self.encoded) - seq_len - 1

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size:
                continue
                
            x_batch = []
            y_batch = []
            
            for idx in batch_indices:
                x = self.encoded[idx : idx + self.seq_len]
                y = self.encoded[idx + 1 : idx + self.seq_len + 1]
                x_batch.append(x)
                y_batch.append(y)
                
            yield np.array(x_batch, dtype=np.int32), np.array(y_batch, dtype=np.int32)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
