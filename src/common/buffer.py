from threading import Lock
from typing import Optional, Sequence

import numpy as np

from ..utils.utils import multimap


def _cast(x):
    B, T = x.shape[0], x.shape[1]
    return x.reshape(B * T, *x.shape[2:])


def _assign(xb=None, xe=None, yb=None, ye=None):
    def fn(x, y):
        x[xb: xe] = y[yb: ye]
    return fn


class Buffer:
    def __init__(self, max_size: int, sequence_length: int = 1):
        self._lock = Lock()
        self.max_size = max_size
        self.sequence_length = sequence_length
        self.max_num_sequences = max_size // sequence_length
        assert self.max_num_sequences > 0
        self.real_num_sequences = 0
        self.next_idx = 0
        self.storage = {}
    
    def __len__(self):
        return self.real_num_sequences

    def keys(self):
        return self.storage.keys()

    def update_next_idx(self, size):
        with self._lock:
            idx = self.next_idx
            self.next_idx = (self.next_idx + size) % self.max_num_sequences
            return idx
    
    def add(self, scheduler_step: int, data: dict, idx: int, size: int) -> None:
        if not self.storage:
            self.storage = multimap(
                lambda x: np.zeros(shape=(self.max_num_sequences, *x.shape[1:]), dtype=x.dtype),
                data
            )
        # Add data
        idx_end = idx + size
        multimap(_assign(xb=idx, xe=idx_end), self.storage, data)
        # Set size
        with self._lock:
            self.real_num_sequences = min(
                self.real_num_sequences + size, self.max_num_sequences
            )
            
    def get_stats_mean(self, key: str, index: Optional[int] = None):
        if index is None:
            return np.mean(self.storage[key][:self.real_num_sequences])
        return np.mean(self.storage[key][index])
    
    def get_by_indices(self, indices: Sequence[int]):
        indices = np.asarray(indices, dtype=np.int64)
        batch = {}
        for key, value in self.storage.items():
            if isinstance(value, dict):
                batch[key] = {}
                for key_, value_ in value.items():
                    batch[key][key_] = value_[indices]
            else:
                batch[key] = value[indices]
        return batch
    
    def get_all(self):
        return self.storage
    
    def get_all_feedforward(self):
        batch = {}
        for key, value in self.storage.items():
            batch[key] = multimap(lambda x: np.expand_dims(_cast(x), 1), value)
        return batch
    
    def get_all_recurrent(self):
        batch = {}
        for key, value in self.storage.items():
            if key == "rnn_states":
                batch[key] = value[:, 0].copy()
            else:
                batch[key] = multimap(lambda x: x.copy(), value)
        return batch
    
    def get_all_truncated_recurrent(self, data_chunk_length):
        assert self.sequence_length % data_chunk_length == 0, \
            "Data_chunk_length must be a factor of sequence_length."
        batch_size = self.real_num_sequences * self.sequence_length
        data_chunks = batch_size // data_chunk_length
        
        flatten_storage = {}
        for key, value in self.storage.items():
            # [N, T, ...] -> [N * T, ...]
            flatten_storage[key] = multimap(lambda x: _cast(x).copy(), value)
            
        batch = {}
        for key, value in flatten_storage.items():
            if key == "rnn_states":
                batch[key] = np.zeros(shape=(data_chunks, *value.shape[1:]), dtype=value.dtype)
            else:
                # [N * T / L, L, ...]
                batch[key] = multimap(lambda x: np.zeros(shape=(data_chunks, data_chunk_length, *x.shape[1:]),
                                      dtype=x.dtype), value)
        for i in range(data_chunks):
            beg, end = i * data_chunk_length, (i + 1) * data_chunk_length
            for key, value in flatten_storage.items():
                if key == "rnn_states":
                    batch[key][i] = value[beg].copy()
                else:
                    multimap(_assign(xb=i, xe=i + 1, yb=beg, ye=end), batch[key], value)
        
        return batch


if __name__ == "__main__":
    buffer = Buffer(max_size=50, sequence_length=10)
    
    batch = {
        "a": np.ones((5, 10)),
        "b": {
            "b_1": np.ones((5, 10, 3)) * 2,
            "b_2": np.ones((5, 10, 2, 6)) * 3,
        }
    }
    
    idx = buffer.update_next_idx(size=5)
    buffer.add(data=batch, idx=idx, size=5)
    
    batch_1 = buffer.get_all_feedforward()
    batch_2 = buffer.get_all_recurrent()
    batch_3 = buffer.get_all_truncated_recurrent(data_chunk_length=5)
    
    batch = {
        "a": np.ones((5, 10)) * 4,
        "b": {
            "b_1": np.ones((5, 10, 3)) * 5,
            "b_2": np.ones((5, 10, 2, 6)) * 6,
        }
    }
    
    idx = buffer.update_next_idx(size=5)
    buffer.add(data=batch, idx=idx, size=5)
