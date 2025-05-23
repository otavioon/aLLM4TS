import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import tqdm
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import LLM4TS_cls
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import math
from data_provider.uea import collate_fn
from pathlib import Path
import pickle

warnings.filterwarnings("ignore")


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {
            2: 5e-5,
            4: 1e-5,
            6: 5e-6,
            8: 1e-6,
            10: 5e-7,
            15: 1e-7,
            20: 5e-8,
        }
    elif args.lradj == "cosine":
        lr_adjust = {
            epoch: args.learning_rate
            / 2
            * (1 + math.cos(epoch / args.train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class Exp_Classification_KNN(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.ckpt_files = self._scan_for_checkpoints()
        self.n_neighbors = args.n_neighbors
        self.knn_results = []

    def _scan_for_checkpoints(self):
        base_model_dir = Path(os.path.join(self.args.pt_sft_base_dir, self.args.pt_sft_model))
        print(f"----> Looking for checkpoints in '{base_model_dir}'")       
        ckpt_files = list(sorted(base_model_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime))
        print(f"Found {len(ckpt_files)} checkpoints")
        return ckpt_files

    def _build_model(self, ckpt_file=None):
        # model input depends on data
        train_data, _ = self._get_data(flag="TRAIN")
        test_data, _ = self._get_data(flag="TEST")
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model_dict = {
            "LLM4TS_cls": LLM4TS_cls,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        if ckpt_file is None:
            print("No checkpoint file provided, using default model initialization.")
            return model

        print(f"Using checkpoint: {ckpt_file}")

        pt_model = torch.load(ckpt_file)  # type: ignore
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pt_model.items() if k in model_dict.keys()}
        if self.args.model == "LLM4TS_cls":
            state_dict.pop("revin_layer.affine_weight", None)
            state_dict.pop("revin_layer.affine_bias", None)
            state_dict.pop("in_layer.weight", None)
            state_dict.pop("in_layer.bias", None)
            state_dict.pop("out_layer.weight", None)
            state_dict.pop("out_layer.bias", None)
        else:
            state_dict.pop("revin_layer.affine_weight", None)
            state_dict.pop("revin_layer.affine_bias", None)
            state_dict.pop("out_layer.weight", None)
            state_dict.pop("out_layer.bias", None)

        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model = model.float().to(self.device)

        del pt_model
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def get_labels_and_projections(self, model, data):
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=128,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
        )

        all_projections, all_labels = [], []
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in tqdm.tqdm(
                enumerate(dataloader),
                total=len(dataloader),
            ):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                _, projections = model(
                    batch_x, padding_mask, None, None, return_projections=True
                )
                all_projections.append(projections)
                all_labels.append(label)

        projections = torch.cat(all_projections, 0).detach().cpu().numpy()
        labels = torch.cat(all_labels, 0).detach().cpu().numpy()
        
        return projections, labels

    def run_knn(self, model, knn_model, train_data, validation_data, test_data, path):
        # Obtain data projections and labels
        train_projections, train_labels = self.get_labels_and_projections(model, train_data)
        val_projections, val_labels = self.get_labels_and_projections(model, validation_data)
        test_projections, test_labels = self.get_labels_and_projections(model, test_data)
        
        # Describe the projections
        print(f"Train projections shape: {train_projections.shape} (min: {train_projections.min()}, mean: {train_projections.mean()}, max: {train_projections.max()})")
        print(f"Validation projections shape: {val_projections.shape} (min: {val_projections.min()}, mean: {val_projections.mean()}, max: {val_projections.max()})")
        print(f"Test projections shape: {test_projections.shape} (min: {test_projections.min()}, mean: {test_projections.mean()}, max: {test_projections.max()})")

        # Squeeze all labels to ensure they are 1D arrays
        train_labels = np.squeeze(train_labels)
        val_labels = np.squeeze(val_labels)
        test_labels = np.squeeze(test_labels)

        # Fit model on training data
        knn_model.fit(train_projections, train_labels)
        predictions_val = np.squeeze(knn_model.predict(val_projections))
        predictions_test = np.squeeze(knn_model.predict(test_projections))

        # Calculate accuracy
        val_acc = cal_accuracy(predictions_val, val_labels)
        test_acc = cal_accuracy(predictions_test, test_labels)
        
        return val_acc, test_acc

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        csv_file = os.path.join(
            self.args.checkpoints, setting, f"results_knn_{self.n_neighbors}.csv"
        )
        
        # insert a None checkpoint to the beginning of the list
        # this represents the case where we do not use any checkpoint (random initialization)
        self.ckpt_files.insert(0, None)

        for ckpt_file in tqdm.tqdm(self.ckpt_files, desc="Training with checkpoints"):
            self.model = self._build_model(ckpt_file)
            self.model = self.model.eval().float().to(self.device)
            if ckpt_file is None:
                path = os.path.join(
                    self.args.checkpoints,
                    setting,
                    "checkpoint_0",
                    f"neighbors_{self.n_neighbors}",
                )
            else:
                path = os.path.join(
                    self.args.checkpoints,
                    setting,
                    ckpt_file.stem,
                    f"neighbors_{self.n_neighbors}",
                )
            if not os.path.exists(path):
                os.makedirs(path)

            knn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            (val_acc, test_acc) = self.run_knn(
                self.model,
                knn_model,
                train_data,
                vali_data,
                test_data,
                path,
            )

            print(f"Checkpoint: {ckpt_file or 'checkpoint_0'}, val_accuracy: {val_acc}, test_accuracy: {test_acc}")
            print()

            with open(os.path.join(path, "results.txt"), "wt") as f:
                f.write(f"Checkpoint: {ckpt_file or 'checkpoint_0'}, val_accuracy: {val_acc}, test_accuracy: {test_acc}\n")

            self.knn_results.append(
                {
                    "checkpoint": ckpt_file or 'checkpoint_0',
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                    "neighbors": self.n_neighbors,
                }
            )

            df = pd.DataFrame(self.knn_results)
            df.to_csv(csv_file, index=False)
            print(f"Results saved to {csv_file}")
            
        df = pd.DataFrame(self.knn_results)
        print(f"Results saved to {csv_file}")
        print("\n---------------------------------------------------------")
        print(df.to_markdown())
        df.to_csv(csv_file, index=False)
        print("---------------------------------------------------------\n")
        print()
        print("Training completed.")

    def test(self, setting, test=0):
        pass
