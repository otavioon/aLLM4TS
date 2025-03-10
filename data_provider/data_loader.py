import os
import numpy as np
import pandas as pd
import os
import torch
import glob
import re
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from tqdm import tqdm
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer

warnings.filterwarnings("ignore")

################################################################################
# MY CODE
################################################################################
from minerva.models.ssl.tfc import TFC_Model
from minerva.models.nets.tfc import TFC_Backbone
from minerva.models.nets.tnc import TSEncoder
import numpy as np
import pandas as pd
import os
import warnings
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import random
import tqdm


from minerva.models.ssl.tfc import TFC_Model
from minerva.models.nets.tfc import TFC_Backbone
import warnings
import warnings
import lightning as L
from torch.utils.data import DataLoader, ConcatDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from minerva.data.datasets.series_dataset import MultiModalSeriesCSVDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pathlib import Path


# class Dataset_pretrain(Dataset):
#     # X is the input data and Y is the input data shifted by stride (16)
#     def __init__(
#         self,
#         pt_data="ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness",
#         patch_len=16,  # Not used
#         stride=16,  # Will be used as the shift (y = x + stride)
#         root_path="/workspaces/HIAAC-KR-Dev-Container/workspace/aLLM4TS/dataset/",
#         flag="train",
#         size=[1024, 0, 1024],  # [seq_len, label_len, pred_len]
#         features="M",
#         data_path="ETTh1.csv",  # Not used
#         target="OT",
#         scale=True,
#         timeenc=1,
#         freq="h",
#         percent=100,
#         return_values="x_y_mark",
#         *args,
#         **kwargs
#     ):
#         assert return_values in ["x", "x_y", "x_y_mark"]
#         self.return_values = return_values
#         self.pt_data = pt_data
#         self.patch_len = patch_len
#         self.stride = stride
#         if size == None:
#             raise NotImplementedError
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]

#         assert flag in ["train", "test", "val"]
#         type_map = {"train": 0, "val": 1, "test": 2}
#         self.set_type = type_map[flag]  # 0: train, 1: val, 2: test

#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#         self.percent = percent

#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         print(f"Loading pretrain data from {self.root_path}")
#         self.scaler = StandardScaler()
#         pt_datasets = self.pt_data.split("_")

#         data_list = []
#         data_stamp_list = []
#         for pt_dataset in pt_datasets:
#             df_raw = pd.read_csv(
#                 os.path.join(self.root_path, f"{pt_dataset}.csv")
#             )
#             dataset_len = len(df_raw)
#             # Handle datasets
#             if "ETTh" in pt_dataset:
#                 border1s = [
#                     0,
#                     12 * 30 * 24 - self.seq_len,
#                     12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
#                 ]
#                 border2s = [
#                     12 * 30 * 24,
#                     12 * 30 * 24 + 4 * 30 * 24,
#                     12 * 30 * 24 + 8 * 30 * 24,
#                 ]
#                 border1 = border1s[self.set_type]
#                 border2 = border2s[self.set_type]
#             elif "ETTm" in pt_dataset:
#                 border1s = [
#                     0,
#                     12 * 30 * 24 * 4 - self.seq_len,
#                     12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
#                 ]
#                 border2s = [
#                     12 * 30 * 24 * 4,
#                     12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
#                     12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
#                 ]
#                 border1 = border1s[self.set_type]
#                 border2 = border2s[self.set_type]
#             else:
#                 num_train = int(dataset_len * 0.7)
#                 num_test = int(dataset_len * 0.2)
#                 num_vali = dataset_len - num_train - num_test
#                 border1s = [
#                     0,
#                     num_train - self.seq_len,
#                     dataset_len - num_test - self.seq_len,
#                 ]
#                 border2s = [num_train, num_train + num_vali, dataset_len]
#                 border1 = border1s[self.set_type]
#                 border2 = border2s[self.set_type]

#             # Hendle some options
#             if self.set_type == 0:
#                 border2 = (
#                     border2 - self.seq_len
#                 ) * self.percent // 100 + self.seq_len
#             if self.features == "M" or self.features == "MS":
#                 cols_data = df_raw.columns[1:]
#                 df_data = df_raw[cols_data]
#             elif self.features == "S":
#                 df_data = df_raw[[self.target]]

#             df_data = df_data.values

#             # Handle scaling. Scaling is done on each subset of the data,
#             # separately. Train data will be normalizedon train data,
#             # validation data will be normalized on validation data, etc.
#             if self.scale:
#                 train_data = df_data[border1s[0] : border2s[0]]
#                 self.scaler.fit(train_data)
#                 data = self.scaler.transform(df_data)
#             else:
#                 data = df_data

#             data = data[border1:border2]
#             data = data.reshape((len(data) * len(cols_data), 1))
#             df_stamp = df_raw[["date"]][border1:border2]
#             df_stamp["date"] = pd.to_datetime(df_stamp.date)
#             if self.timeenc == 0:
#                 df_stamp["month"] = df_stamp.date.apply(
#                     lambda row: row.month, 1
#                 )
#                 df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
#                 df_stamp["weekday"] = df_stamp.date.apply(
#                     lambda row: row.weekday(), 1
#                 )
#                 df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
#                 data_stamp = df_stamp.drop(["date"], axis=1).values
#             elif self.timeenc == 1:
#                 data_stamp = time_features(
#                     pd.to_datetime(df_stamp["date"].values), freq=self.freq
#                 )
#                 data_stamp = data_stamp.transpose(1, 0)

#             data_list.append(data)
#             df_stamp = np.array(
#                 [data_stamp for i in range(len(cols_data))]
#             ).reshape((len(data_stamp) * len(cols_data), 4))
#             data_stamp_list.append(df_stamp)

#         self.data = np.concatenate(data_list, axis=0)
#         self.data_stamp = np.concatenate(data_stamp_list, axis=0)

#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_begin + self.stride
#         r_end = s_end + self.stride
#         # print(s_begin, s_end, r_begin, r_end)

#         # print(f"Values: seq_len={self.seq_len}, patch_len={self.patch_len}, stride={self.stride}, s_begin={s_begin}, s_end={s_end}, r_begin={r_begin}, r_end={r_end}")
#         # print(f"Seq_x: from {s_begin} to {s_end}")
#         # print(f"Seq_y: from {r_begin} to {r_end}")

#         seq_x = self.data[s_begin:s_end].swapaxes(0, 1)
#         seq_y = self.data[r_begin:r_end].swapaxes(0, 1)

#         if self.return_values == "x":
#             return seq_x
#         elif self.return_values == "x_y":
#             return seq_x, seq_y
#         else:
#             seq_x_mark = self.data_stamp[s_begin:s_end]
#             seq_y_mark = self.data_stamp[r_begin:r_end]
#             return seq_x, seq_y, seq_x_mark, seq_y_mark

#     def __len__(self):
#         return len(self.data) - self.seq_len - self.patch_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


class Base_Har_Dataset(Dataset):
    def __init__(
        self,
        root_path: str,
        feature_prefixes=[
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        flag: str = "train",
        size=[1024, 0, 1024],  # [seq_len, label_len, pred_len]
        stride=16,
        return_values="x_y",
        val_split: float = 0.1,
        test_split: float = 0.2,
        seed: int = 42,
        scale=True,
        percent: int = 100,
        *args,
        **kwargs,
    ):
        assert return_values in ["x", "x_y"]
        assert flag in ["train", "test", "val"]

        # From arguments
        self.root_dir = Path(root_path)
        self.feature_prefixes = feature_prefixes
        self.flag = flag
        self.seq_len, self.label_len, self.pred_len = size[0], size[1], size[2]
        self.stride = stride
        self.return_values = return_values
        self.seed = seed
        self.scale = scale
        self.percent = percent

        # Split the data
        self.train_split = 1 - val_split - test_split
        self.val_split = val_split
        self.test_split = test_split
        assert (
            self.train_split + self.val_split + self.test_split == 1
        ), "Splits should sum to 1"

        # Read the data and set the data and x, y begin and end indices
        self.scaler = StandardScaler()
        self.data, self.effective_stride = self.__read_data__()

    def _get_dataset_data(self) -> np.ndarray:
        raise NotImplementedError

    def __read_data__(self):
        data = self._get_dataset_data()
        original_shape = data.shape
        effective_stride = self.stride * original_shape[1]
        # Maintin only first dimension and flatten the rest
        print(
            f"Data shape before flatten: {data.shape}. Effective stride: {effective_stride} ({self.stride}*{original_shape[1]})"
        )
        # Flatten each sample in F order
        data = np.ascontiguousarray(
            data.reshape(original_shape[0], -1, order="F")
        )
        print(
            f"Data shape after flatten (F-order): {data.shape}. Min: {data.min()}, Max: {data.max()}"
        )

        if self.scale:
            shape = data.shape
            data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(shape)
            # Reshape, do the scaling and reshape back
            print(
                f"Data shape after scaling: {data.shape}. Min: {data.min()}, Max: {data.max()}"
            )

        if self.percent < 100:
            # Take only the first self.percent samples
            data = data[: int(data.shape[0] * self.percent / 100)]
            print(f"Data shape after taking {self.percent}%: {data.shape}")

        # Now, we have data of shape (num_samples, num_features * num_timesteps)
        # Where the last dimension is the flattened version of the original data
        # interleaved (F-order)
        return (data, effective_stride)

    def __getitem__(self, index):
        data = self.data[index]
        seq_x = data[: -self.effective_stride]
        seq_y = data[self.effective_stride :]

        # 1024-264 = 760 zeros at the end
        pad_size = max(0, self.seq_len - len(seq_x))
        seq_x = np.pad(seq_x, (0, pad_size), mode="constant", constant_values=0)
        seq_y = np.pad(seq_y, (0, pad_size), mode="constant", constant_values=0)

        # Add a channel dimension at the beginning
        seq_x = seq_x.reshape(1, -1).T
        seq_y = seq_y.reshape(1, -1).T
        seq_mark_x = np.zeros((len(seq_x), 4))
        seq_mark_y = np.zeros((len(seq_y), 4))

        # print(f"DAGHAR: {seq_x.shape}, {seq_y.shape}, {seq_mark_x.shape}, {seq_mark_y.shape}")

        return seq_x, seq_y, seq_mark_x, seq_mark_y

    def __len__(self):
        return len(self.data)

    def inverse_transform(self, data):
        shape = data.shape
        data = data.reshape(-1, 1)
        data = self.scaler.inverse_transform(data)
        return data.reshape(shape)


class Dataset_ExtraSensory(Base_Har_Dataset):
    def __init__(self, *args, **kwargs):
        kwargs.pop("root_path", None)
        root_path = "/workspaces/HIAAC-KR-Dev-Container/some_datasets/ES_Raw/"
        super().__init__(
            root_path=root_path,
            *args,
            **kwargs,
        )

        print(f"ROOT PATH: {root_path}")
        print(f"ARGS: {args}")
        print(f"KWARGS: {kwargs}")

    def _get_dataset_data(self) -> np.ndarray:
        print(f"Loading ExtraSensory data from {self.root_dir}")
        rng = random.Random(self.seed)
        # List files and shuffle it using the seed
        files = sorted(list(self.root_dir.glob("es_full.*.csv")))
        rng.shuffle(files)

        # Split the files
        num_train = int(len(files) * self.train_split)
        num_val = int(len(files) * self.val_split)

        if self.flag == "train":
            # First num_train files
            files = files[:num_train]
        elif self.flag == "val":
            # Files from num_train to num_train + num_val
            files = files[num_train : num_train + num_val]
        else:
            # Files from num_train + num_val to the end (last num_test files)
            files = files[num_train + num_val :]

        datasets = []
        tqdm_bar = tqdm.tqdm(
            enumerate(files), total=len(files), desc="Reading files..."
        )
        for i, f in tqdm_bar:
            tqdm_bar.set_description(f"Reading {f.name}")
            dataset = MultiModalSeriesCSVDataset(
                f, self.feature_prefixes, label=None, features_as_channels=True
            )

            # This should return a numpy array of shape (num_samples, num_features, num_timesteps)
            # In this case, (num_samples, 6, 60)
            data = dataset[:].astype(np.float32)
            datasets.append(data)

        # Data is an array of shape (num_samples, num_features, num_timesteps)
        data = np.concatenate(datasets, axis=0)

        return data


class Dataset_DAGHAR(Base_Har_Dataset):
    def __init__(self, root_path, *args, **kwargs):
        kwargs.pop("root_path", None)
        root_path = "/workspaces/HIAAC-KR-Dev-Container/some_datasets/DAGHAR/standardized_view/"

        super().__init__(
            root_path=root_path,
            *args,
            **kwargs,
        )

        print(f"ROOT PATH: {root_path}")
        print(f"ARGS: {args}")
        print(f"KWARGS: {kwargs}")

    def _get_dataset_data(self) -> np.ndarray:
        print(f"Loading DAGHAR data from {self.root_dir}")
        if self.flag == "train":
            files = self.root_dir.rglob("train.csv")
        elif self.flag == "val":
            files = self.root_dir.rglob("validation.csv")
        else:
            files = self.root_dir.rglob("test.csv")
        files = sorted(list(files))
        print(f"Selected {len(files)} files for {self.flag}: {files}")

        datasets = []
        tqdm_bar = tqdm.tqdm(
            enumerate(files), total=len(files), desc="Reading files..."
        )
        for i, f in tqdm_bar:
            tqdm_bar.set_description(f"Reading {f.name}")
            dataset = MultiModalSeriesCSVDataset(
                f, self.feature_prefixes, label=None, features_as_channels=True
            )

            # This should return a numpy array of shape (num_samples, num_features, num_timesteps)
            # In this case, (num_samples, 6, 60)
            data = dataset[:].astype(np.float32)
            datasets.append(data)

        # Data is an array of shape (num_samples, num_features, num_timesteps)
        data = np.concatenate(datasets, axis=0)

        return data


# class Dataset_DAGHAR(Dataset):
#     def __init__(self, *args, **kwargs):
#         flag = kwargs.get("flag")

#         root_dir = Path("/workspaces/HIAAC-KR-Dev-Container/workspace/aLLM4TS/dataset/")
#         self.path = root_dir / f"daghar_{flag}.npz"
#         data = np.load(self.path)
#         print(f"Loaded DAGHAR data from {self.path}")
#         self.X = data["X"].astype(np.float32)
#         self.y = data["Y"].astype(np.float32)
#         self.marks = np.zeros((self.X.shape[1], 4))
#         print(f"DAGHAR LOADED: X shape: {self.X.shape}, y shape: {self.y.shape}, marks shape: {self.marks.shape}")

#     def __getitem__(self, index):
#         return self.X[index], self.y[index], self.marks, self.marks

#     def __len__(self):
#         return len(self.X)


class MyConcatDataset(Dataset):
    def __init__(self, datasets):
        self.concat_dataset = ConcatDataset(datasets)
        print(f"Concatenated dataset with {len(self.concat_dataset)} samples")

    def __getitem__(self, index):
        return self.concat_dataset[index]

    def __len__(self):
        return len(self.concat_dataset)


class Pretrain_allm4ts_ES_dataset(MyConcatDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            [
                Dataset_pretrain(*args, **kwargs),
                Dataset_ExtraSensory(*args, **kwargs),
            ]
        )


class Pretrain_allm4ts_DAGHAR_dataset(MyConcatDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            [Dataset_pretrain(*args, **kwargs), Dataset_DAGHAR(*args, **kwargs)]
        )


################################################################################


class Dataset_pretrain(Dataset):
    def __init__(
        self,
        configs,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        percent=100,
    ):

        self.configs = configs
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        if size == None:
            raise NotImplementedError
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.flag = flag
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.dataset_loc = None
        self.another_dataset = None

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        pt_datasets = self.configs.pt_data.split("_")

        data_list = []
        data_stamp_list = []

        for pt_dataset in pt_datasets:
            df_raw = pd.read_csv(
                os.path.join(self.root_path, f"{pt_dataset}.csv")
            )
            dataset_len = len(df_raw)
            if "ETTh" in pt_dataset:
                border1s = [
                    0,
                    12 * 30 * 24 - self.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
                ]
                border2s = [
                    12 * 30 * 24,
                    12 * 30 * 24 + 4 * 30 * 24,
                    12 * 30 * 24 + 8 * 30 * 24,
                ]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]
            elif "ETTm" in pt_dataset:
                border1s = [
                    0,
                    12 * 30 * 24 * 4 - self.seq_len,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
                ]
                border2s = [
                    12 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                    12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
                ]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]
            else:
                num_train = int(dataset_len * 0.7)
                num_test = int(dataset_len * 0.2)
                num_vali = dataset_len - num_train - num_test
                border1s = [
                    0,
                    num_train - self.seq_len,
                    dataset_len - num_test - self.seq_len,
                ]
                border2s = [num_train, num_train + num_vali, dataset_len]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]
            if self.set_type == 0:
                border2 = (
                    border2 - self.seq_len
                ) * self.percent // 100 + self.seq_len
            if self.features == "M" or self.features == "MS":
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == "S":
                df_data = df_raw[[self.target]]

            df_data = df_data.values

            if self.scale:
                train_data = df_data[border1s[0] : border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(df_data)
            else:
                data = df_data

            data = data[border1:border2]
            data = data.reshape((len(data) * len(cols_data), 1))
            df_stamp = df_raw[["date"]][border1:border2]
            df_stamp["date"] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp["month"] = df_stamp.date.apply(
                    lambda row: row.month, 1
                )
                df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp["weekday"] = df_stamp.date.apply(
                    lambda row: row.weekday(), 1
                )
                df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(["date"], axis=1).values
            elif self.timeenc == 1:
                data_stamp = time_features(
                    pd.to_datetime(df_stamp["date"].values), freq=self.freq
                )
                data_stamp = data_stamp.transpose(1, 0)

            data_list.append(data)
            df_stamp = np.array(
                [data_stamp for i in range(len(cols_data))]
            ).reshape((len(data_stamp) * len(cols_data), 4))
            data_stamp_list.append(df_stamp)

        self.data = np.concatenate(data_list, axis=0)
        self.data_stamp = np.concatenate(data_stamp_list, axis=0)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.stride
        r_end = s_end + self.stride

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # print(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.patch_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        configs,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        percent=100,
    ):

        self.configs = configs

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (
                border2 - self.seq_len
            ) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        configs,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        percent=100,
    ):

        self.configs = configs

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (
                border2 - self.seq_len
            ) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Weather(Dataset):
    def __init__(
        self,
        configs,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        percent=100,
    ):

        self.configs = configs

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (
                border2 - self.seq_len
            ) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Traffic(Dataset):
    def __init__(
        self,
        configs,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        percent=100,
    ):

        self.configs = configs

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (
                border2 - self.seq_len
            ) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Electricity(Dataset):
    def __init__(
        self,
        configs,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        percent=100,
    ):

        self.configs = configs

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (
                border2 - self.seq_len
            ) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        configs,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        percent=100,
    ):

        self.configs = configs

        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (
                border2 - self.seq_len
            ) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, "train.csv"))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(
            os.path.join(root_path, "test_label.csv")
        ).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(
                self.train[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "val":
            return np.float32(
                self.val[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index : index + self.win_size]
            ), np.float32(self.test_labels[index : index + self.win_size])
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(
            os.path.join(root_path, "MSL_test_label.npy")
        )
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(
                self.train[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "val":
            return np.float32(
                self.val[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index : index + self.win_size]
            ), np.float32(self.test_labels[index : index + self.win_size])
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(
            os.path.join(root_path, "SMAP_test_label.npy")
        )
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(
                self.train[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "val":
            return np.float32(
                self.val[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index : index + self.win_size]
            ), np.float32(self.test_labels[index : index + self.win_size])
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8) :]
        self.test_labels = np.load(
            os.path.join(root_path, "SMD_test_label.npy")
        )

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(
                self.train[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "val":
            return np.float32(
                self.val[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index : index + self.win_size]
            ), np.float32(self.test_labels[index : index + self.win_size])
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "swat_train2.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "swat2.csv"))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        self.val = test_data
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(
                self.train[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "val":
            return np.float32(
                self.val[index : index + self.win_size]
            ), np.float32(self.test_labels[0 : self.win_size])
        elif self.flag == "test":
            return np.float32(
                self.test[index : index + self.win_size]
            ), np.float32(self.test_labels[index : index + self.win_size])
        else:
            return np.float32(
                self.test[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            ), np.float32(
                self.test_labels[
                    index
                    // self.step
                    * self.win_size : index
                    // self.step
                    * self.win_size
                    + self.win_size
                ]
            )


class Dataset_M4(Dataset):
    def __init__(
        self,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=False,
        inverse=False,
        timeenc=0,
        freq="15min",
        seasonal_patterns="Yearly",
    ):

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        if self.flag == "train":
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(
                training=False, dataset_file=self.root_path
            )
        training_values = np.array(
            [
                v[~np.isnan(v)]
                for v in dataset.values[
                    dataset.groups == self.seasonal_patterns
                ]
            ]
        )
        self.ids = np.array(
            [i for i in dataset.ids[dataset.groups == self.seasonal_patterns]]
        )
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(
            low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
            high=len(sampled_timeseries),
            size=1,
        )[0]

        insample_window = sampled_timeseries[
            max(0, cut_point - self.seq_len) : cut_point
        ]
        insample[-len(insample_window) :, 0] = insample_window
        insample_mask[-len(insample_window) :, 0] = 1.0
        outsample_window = sampled_timeseries[
            cut_point
            - self.label_len : min(
                len(sampled_timeseries), cut_point + self.pred_len
            )
        ]
        outsample[: len(outsample_window), 0] = outsample_window
        outsample_mask[: len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len :]
            insample[i, -len(ts) :] = ts_last_window
            insample_mask[i, -len(ts) :] = 1.0
        return insample, insample_mask


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(
            root_path, file_list=file_list, flag=flag
        )
        self.all_IDs = (
            self.all_df.index.unique()
        )  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(
                os.path.join(root_path, "*")
            )  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_path, "*"))
            )
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [
            p for p in data_paths if os.path.isfile(p) and p.endswith(".ts")
        ]
        if len(input_paths) == 0:
            pattern = "*.ts"
            raise Exception(
                "No .ts files found using pattern: '{}'".format(pattern)
            )

        all_df, labels_df = self.load_single(
            input_paths[0]
        )  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(
            filepath,
            return_separate_X_and_y=True,
            replace_missing_vals_with="NaN",
        )
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(
            labels.cat.codes, dtype=np.int8
        )  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)
        ).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if (
            np.sum(horiz_diffs) > 0
        ):  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if (
            np.sum(vert_diffs) > 0
        ):  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat(
            (
                pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                .reset_index(drop=True)
                .set_index(pd.Series(lengths[row, 0] * [row]))
                for row in range(df.shape[0])
            ),
            axis=0,
        )

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if (
            self.root_path.count("EthanolConcentration") > 0
        ):  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(
                torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(
            torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)
        ), torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)


class DAGHAR(Dataset):
    def __init__(self, root_path, flag="train"):
        self.flag = flag.lower()
        self.root_path = Path(root_path)
        
        if self.flag == "train":
            self.file = self.root_path / "train.csv"
        elif self.flag == "val":
            self.file = self.root_path / "validation.csv"
        else:
            self.file = self.root_path / "test.csv"

        self.dataset = self._read_data()
        self.max_seq_len = 60
        self.feature_df = self.dataset.data.copy().T
        self.class_names = list(range(6))

    def _read_data(self):
        dataset = MultiModalSeriesCSVDataset(
            data_path=self.file,
            feature_prefixes=[
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
            ],
            label="standard activity code",
            features_as_channels=True,
            cast_to="float64"
        )
        return dataset
    
    def instance_norm(self, case):
        mean = case.mean(0, keepdim=True)
        case = case - mean
        stdev = torch.sqrt(
            torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        case /= stdev
        return case


    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = x.T
        y = np.array(y, dtype=np.int8)
        y = np.expand_dims(y, axis=0)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        # x = self.instance_norm(x)
        return x, y

    def __len__(self):
        return len(self.dataset)