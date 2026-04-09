from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _pack_to_model_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim == 2:
        t, n = arr.shape
        out = np.zeros((t, n, 7), dtype=np.float32)
        out[..., 0:3] = arr[..., None]
        return out
    if arr.ndim == 3:
        t, n, f = arr.shape
        if f >= 7:
            return arr[..., :7].astype(np.float32)
        out = np.zeros((t, n, 7), dtype=np.float32)
        out[..., :f] = arr
        return out
    raise ValueError(f"Unsupported data shape: {arr.shape}")


def load_raw_data(data_path: str) -> np.ndarray:
    p = Path(data_path)
    ext = p.suffix.lower()
    if ext == ".npz":
        npz = np.load(str(p), allow_pickle=True)
        if "data" in npz:
            return _pack_to_model_shape(npz["data"])
        return _pack_to_model_shape(npz[npz.files[0]])
    if ext == ".h5":
        try:
            df = pd.read_hdf(str(p))
            return _pack_to_model_shape(df.values)
        except Exception:
            import h5py
            with h5py.File(str(p), "r") as f:
                key = list(f.keys())[0]
                return _pack_to_model_shape(np.array(f[key]))
    if ext == ".csv":
        df = pd.read_csv(str(p))
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            num = df.apply(pd.to_numeric, errors="coerce")
        return _pack_to_model_shape(num.values)
    raise ValueError(f"Unsupported data format: {data_path}")


class TrafficDataset(Dataset):
    def __init__(self, data_path: str, input_window: int, output_window: int, mode: str = "train"):
        raw_data = load_raw_data(data_path)
        self.mean = np.mean(raw_data[..., :3], axis=(0, 1), keepdims=True)
        self.std = np.std(raw_data[..., :3], axis=(0, 1), keepdims=True)
        raw_data[..., :3] = (raw_data[..., :3] - self.mean) / (self.std + 1e-5)

        self.data = torch.tensor(raw_data, dtype=torch.float32)
        self.input_window = int(input_window)
        self.output_window = int(output_window)

        total_len = len(raw_data) - self.input_window - self.output_window
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.1)

        if mode == "train":
            self.start_idx, self.end_idx = 0, train_len
        elif mode == "val":
            self.start_idx, self.end_idx = train_len, train_len + val_len
        else:
            self.start_idx, self.end_idx = train_len + val_len, total_len

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, index: int):
        idx = self.start_idx + index
        x = self.data[idx: idx + self.input_window]
        y = self.data[idx + self.input_window: idx + self.input_window + self.output_window]
        time_ind = torch.tensor((idx + self.input_window) % 288, dtype=torch.long)
        return x, y[..., 0], time_ind


def create_dataloaders(data_path: str, input_window: int, output_window: int, batch_size: int):
    train_ds = TrafficDataset(data_path, input_window, output_window, "train")
    val_ds = TrafficDataset(data_path, input_window, output_window, "val")
    test_ds = TrafficDataset(data_path, input_window, output_window, "test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_ds
