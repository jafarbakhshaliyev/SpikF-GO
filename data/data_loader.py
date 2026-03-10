import os
import datetime
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler



def _split_with_overlap(data: np.ndarray, train_ratio: float, val_ratio: float, seq_len: int):
    """
    Time split with overlap for val/test to allow past context:
      train: [0 : train_end)
      val  : [train_end - seq_len : val_end)
      test : [val_end - seq_len : T)
    """
    T = len(data)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_end = max(0, min(train_end, T))
    val_end = max(train_end, min(val_end, T))

    val_start = max(0, train_end - seq_len)
    test_start = max(0, val_end - seq_len)

    train_data = data[:train_end]
    val_data = data[val_start:val_end]
    test_data = data[test_start:]

    return train_data, val_data, test_data


def _fit_transform_splits(train_data, val_data, test_data, type_flag: str, scaler=None):
    if type_flag == "1":
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)
        return train_data, val_data, test_data, scaler
    else:
        return train_data, val_data, test_data, None


def _to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _clean_numeric_csv(df: pd.DataFrame) -> np.ndarray:
    """
    Keep only numeric columns, and drop common junk index columns.
    """
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    num_df = df.select_dtypes(include=[np.number])

    if num_df.shape[1] == 0:
        raise ValueError("No numeric columns found in CSV after cleaning. Check your file format.")

    num_df = num_df.dropna(axis=0, how="any")

    return num_df.values.astype(np.float32)



class _BaseTimeSeriesDataset(Dataset):

    def __init__(self, flag, seq_len, pre_len):
        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.seq_len = int(seq_len)
        self.pre_len = int(pre_len)
        self.scaler = None
        self.split = None  

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pre_len

        x = self.split[s_begin:s_end]
        y = self.split[s_end:r_end]
        return x, y

    def __len__(self):
        if self.split is None:
            return 0
        return max(0, len(self.split) - self.seq_len - self.pre_len)


class Dataset_Dhfm(_BaseTimeSeriesDataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio, scaler=None):
        super().__init__(flag, seq_len, pre_len)
        self.path = root_path

        load_data = np.load(root_path) 
        data = np.array(load_data).transpose() 
        data = _to_float32(data)

        train_data, val_data, test_data = _split_with_overlap(data, train_ratio, val_ratio, self.seq_len)
        train_data, val_data, test_data, self.scaler = _fit_transform_splits(train_data, val_data, test_data, type, scaler)

        if self.flag == "train":
            self.split = train_data
        elif self.flag == "val":
            self.split = val_data
        else:
            self.split = test_data


class Dataset_ECG(_BaseTimeSeriesDataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio, scaler=None):
        super().__init__(flag, seq_len, pre_len)
        self.path = root_path

        df = pd.read_csv(root_path)
        data = _clean_numeric_csv(df)  

        train_data, val_data, test_data = _split_with_overlap(data, train_ratio, val_ratio, self.seq_len)
        train_data, val_data, test_data, self.scaler = _fit_transform_splits(train_data, val_data, test_data, type, scaler)

        if self.flag == "train":
            self.split = train_data
        elif self.flag == "val":
            self.split = val_data
        else:
            self.split = test_data

class Dataset_Solar(_BaseTimeSeriesDataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio, scaler=None):
        super().__init__(flag, seq_len, pre_len)
        self.path = root_path

        files = os.listdir(root_path)
        solar_data = []
        time_data = None

        for file in files:
            full = os.path.join(root_path, file)
            if os.path.isdir(full):
                continue
            if file.startswith("DA_"):
                arr = pd.read_csv(full).values
                raw_time = arr[:, 0:1]
                if time_data is None:
                    time_data = raw_time
                raw_data = arr[:, 1:arr.shape[1]]
                raw_data = raw_data.transpose()
                solar_data.append(raw_data)

        if len(solar_data) == 0 or time_data is None:
            raise ValueError(f"No solar files found in {root_path} with prefix 'DA_'.")

        solar_data = np.array(solar_data).squeeze(1).transpose()   # (T, N)
        time_data = np.array(time_data)                             # (T, 1)
        out = np.concatenate((time_data, solar_data), axis=1)       # (T, 1+N)

        filtered = []
        for item in out:
            dt = datetime.datetime.strptime(item[0], "%m/%d/%y %H:%M")
            if 8 <= dt.hour <= 17:
                filtered.append(item[1:out.shape[1]-1])

        data = _to_float32(np.array(filtered))

        train_data, val_data, test_data = _split_with_overlap(data, train_ratio, val_ratio, self.seq_len)
        train_data, val_data, test_data, self.scaler = _fit_transform_splits(train_data, val_data, test_data, type, scaler)

        if self.flag == "train":
            self.split = train_data
        elif self.flag == "val":
            self.split = val_data
        else:
            self.split = test_data


class Dataset_Wiki(_BaseTimeSeriesDataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio, scaler=None):
        super().__init__(flag, seq_len, pre_len)
        self.path = root_path

        df = pd.read_csv(root_path)

        if df.shape[1] < 2:
            raise ValueError("Wiki CSV must have at least 2 columns (time + features).")

        df_feat = df.iloc[:, 1:]
        data = _clean_numeric_csv(df_feat)  

        train_data, val_data, test_data = _split_with_overlap(data, train_ratio, val_ratio, self.seq_len)
        train_data, val_data, test_data, self.scaler = _fit_transform_splits(train_data, val_data, test_data, type, scaler)

        if self.flag == "train":
            self.split = train_data
        elif self.flag == "val":
            self.split = val_data
        else:
            self.split = test_data



class Dataset_PEMS_BAY(_BaseTimeSeriesDataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio, scaler=None, fillna="ffill"):
        super().__init__(flag, seq_len, pre_len)
        self.path = root_path

        obj = pd.read_hdf(root_path)

        if isinstance(obj, pd.Series):
            df = obj.to_frame()
        elif isinstance(obj, pd.DataFrame):
            df = obj
        else:
            df = pd.DataFrame(obj)

        if fillna == "ffill":
            df = df.ffill()
            df = df.fillna(0.0)  
        elif fillna == "zero":
            df = df.fillna(0.0)
        elif fillna == "drop":
            df = df.dropna(axis=0, how="any")
        elif fillna is None:
            pass
        else:
            raise ValueError("fillna must be one of: 'ffill', 'zero', 'drop', or None")

        data = df.values.astype(np.float32)

        train_data, val_data, test_data = _split_with_overlap(data, train_ratio, val_ratio, self.seq_len)
        train_data, val_data, test_data, self.scaler = _fit_transform_splits(train_data, val_data, test_data, type, scaler)

        if self.flag == "train":
            self.split = train_data
        elif self.flag == "val":
            self.split = val_data
        else:
            self.split = test_data
