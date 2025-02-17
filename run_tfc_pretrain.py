#!/usr/bin/env python
# coding: utf-8


import argparse
from typing import Literal
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
from data_provider.data_loader import (
    Dataset_pretrain,
    Pretrain_allm4ts_ES_dataset,
    Pretrain_allm4ts_DAGHAR_dataset,
)


warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class TFC_Dataset_Wrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, y, x_mark, y_mark = self.dataset[index]
        x = x.T
        y = y.T
        return x, y

    def __len__(self):
        return len(self.dataset)


class config:
    def __init__(self, pt_data, patch_len, stride):
        self.pt_data = pt_data
        self.patch_len = patch_len
        self.stride = stride


def train(
    checkpoints: str,
    model_name: str = "tfc-transformer-encoder",
    ckpt_version: str = "final",
    input_channels: int = 1,
    TS_length: int = 1024,
    batch_size: int = 128,
    num_workers: int = 0,
    data: Literal[
        "pretrain", "pretrain_allm4ts_es", "pretrain_allm4ts_daghar"
    ] = "pretrain",
    train_epochs: int = 100,
    accelerator: str = "gpu",
    devices: int = 1,
):
    assert data in [
        "pretrain",
        "pretrain_allm4ts_es",
        "pretrain_allm4ts_daghar",
    ]

    ############################################################################
    # DATA
    ############################################################################
    default_pretrain_args = {
        "configs": config(
            pt_data="ETTh1_ETTm1_ETTh2_ETTm2_weather_traffic_electricity_illness",
            patch_len=16,
            stride=16,
        ),
        "root_path": "/workspaces/HIAAC-KR-Dev-Container/workspace/aLLM4TS/dataset/",
        "size": [1024, 0, 1024],
        "features": "M",
        "data_path": "ETTh1.csv",
        "target": "OT",
        "scale": True,
        "timeenc": 1,
        "freq": "h",
        "percent": 100,
    }

    if data == "pretrain":
        train_dataset = Dataset_pretrain(flag="train", **default_pretrain_args)
        val_dataset = Dataset_pretrain(flag="val", **default_pretrain_args)
    elif data == "pretrain_allm4ts_es":
        train_dataset = Pretrain_allm4ts_ES_dataset(
            flag="train", **default_pretrain_args
        )
        val_dataset = Pretrain_allm4ts_ES_dataset(
            flag="val", **default_pretrain_args
        )
    else:
        train_dataset = Pretrain_allm4ts_DAGHAR_dataset(
            flag="train", **default_pretrain_args
        )
        val_dataset = Pretrain_allm4ts_DAGHAR_dataset(
            flag="val", **default_pretrain_args
        )
        
    train_dataset = TFC_Dataset_Wrapper(train_dataset)
    val_dataset = TFC_Dataset_Wrapper(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    model = TFC_Model(
        input_channels=input_channels,
        batch_size=batch_size,
        TS_length=TS_length,
        num_classes=None,
        batch_1_correction=True,
        backbone=TFC_Backbone(
            input_channels=input_channels,
            TS_length=TS_length,
            time_encoder=TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=TS_length, dim_feedforward=2 * 128, nhead=2
                ),
                num_layers=2,
            ),
            frequency_encoder=TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=TS_length, dim_feedforward=2 * 128, nhead=2
                ),
                num_layers=2,
            ),
        ),
    )

    callbacks = [
        ModelCheckpoint(
            filename=model_name + "-{epoch:02d}",
            every_n_epochs=1,
        ),
        EarlyStopping(
            patience=40,
            verbose=True,
            monitor="val_loss",
        ),
    ]

    logger = CSVLogger(
        save_dir=checkpoints,
        name=model_name,
        version=ckpt_version,
    )

    trainer = L.Trainer(
        max_epochs=train_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print("Training Done!")


def main():
    parser = argparse.ArgumentParser(description="Train TFC Model")
    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tfc-transformer-encoder",
        help="Model name",
    )
    parser.add_argument(
        "--ckpt_version", type=str, default="final", help="Checkpoint version"
    )
    parser.add_argument(
        "--input_channels", type=int, default=1, help="Number of input channels"
    )
    parser.add_argument(
        "--TS_length", type=int, default=1024, help="Time series length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers"
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["pretrain", "pretrain_allm4ts_es", "pretrain_allm4ts_daghar"],
        default="pretrain",
        help="Dataset type",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type"
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices"
    )

    args = parser.parse_args()

    train(
        checkpoints=args.checkpoints,
        model_name=args.model_name,
        ckpt_version=args.ckpt_version,
        input_channels=args.input_channels,
        TS_length=args.TS_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data=args.data,
        train_epochs=args.train_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
    )


if __name__ == "__main__":
    main()
