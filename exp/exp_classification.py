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


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag="TRAIN")
        test_data, test_loader = self._get_data(flag="TEST")
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

        if self.args.pt_sft:

            if self.args.load_last:
                base_model_dir = os.path.join(
                    self.args.pt_sft_base_dir, self.args.pt_sft_model
                )
                print(f"Looking for checkpoints in '{base_model_dir}'")
                # Get the last modified file in base_model_dir ending with 'ckpt'
                ckpt_files = [
                    f for f in os.listdir(base_model_dir) if f.endswith(".ckpt")
                ]
                if not ckpt_files:
                    raise FileNotFoundError(
                        "No checkpoint files found in the directory."
                    )
                ckpt_file = max(
                    ckpt_files,
                    key=lambda f: os.path.getmtime(
                        os.path.join(base_model_dir, f)
                    ),
                )
                ckpt_file = os.path.join(base_model_dir, ckpt_file)

                print(
                    f"Loading model from checkpoint (last modified): {ckpt_file}"
                )
            else:
                ckpt_file = os.path.join(
                    self.args.pt_sft_base_dir,
                    self.args.pt_sft_model,
                    "checkpoint.pth",
                )
                print(f"Loading model from checkpoint: {ckpt_file}")

            pt_model = torch.load(ckpt_file)
            model_dict = model.state_dict()
            state_dict = {
                k: v for k, v in pt_model.items() if k in model_dict.keys()
            }
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

            del pt_model
        else:
            print(">> No pre-trained model (with aLLM4TS) loaded!")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds
        )  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def save_projections(self, model, data, path, name):
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=128,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len),
        )

        all_projections = []
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in tqdm.tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Saving projections: {name}",
            ):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                _, projections = model(
                    batch_x, padding_mask, None, None, return_projections=True
                )
                all_projections.append(projections)

        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"{name}.npy")

        np.save(
            save_path,
            torch.cat(all_projections, 0).detach().cpu().numpy(),
        )

        print(f"Projections saved to {save_path}")

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        # self.save_projections(
        #     self.model, train_data, path, "train_projections_before_training"
        # )
        # self.save_projections(
        #     self.model, vali_data, path, "val_projections_before_training"
        # )
        # self.save_projections(
        #     self.model, test_data, path, "test_projections_before_training"
        # )

        for epoch in range(self.args.train_epochs):
            print(f"Epoch: {epoch + 1}...")
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in tqdm.tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Training. Epoch: {epoch + 1}",
            ):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs, projections = self.model(
                    batch_x, padding_mask, None, None, return_projections=True
                )
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print(
                "Epoch: {} cost time: {}".format(
                    epoch + 1, time.time() - epoch_time
                )
            )
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(
                vali_data, vali_loader, criterion
            )
            test_loss, test_accuracy = self.vali(
                test_data, test_loader, criterion
            )

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    val_accuracy,
                    test_loss,
                    test_accuracy,
                )
            )
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        # self.save_projections(
        #     self.model, train_data, path, "train_projections_after_training"
        # )
        # self.save_projections(
        #     self.model, vali_data, path, "val_projections_after_training"
        # )
        # self.save_projections(
        #     self.model, test_data, path, "test_projections_after_training"
        # )

        return self.model

    def test(self, setting, test=0):
        for flag in ["VAL", "TEST"]:
            try:
                _, test_loader = self._get_data(flag=flag)
                if test:
                    print("loading model")
                    self.model.load_state_dict(
                        torch.load(
                            os.path.join("./checkpoints/" + setting, "checkpoint.pth")
                        )
                    )

                preds = []
                trues = []
                folder_path = f"./{flag}_results/" + setting + "/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                self.model.eval()
                with torch.no_grad():
                    for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                        batch_x = batch_x.float().to(self.device)
                        padding_mask = padding_mask.float().to(self.device)
                        label = label.to(self.device)

                        outputs = self.model(batch_x, padding_mask, None, None)

                        preds.append(outputs.detach())
                        trues.append(label)

                preds = torch.cat(preds, 0)
                trues = torch.cat(trues, 0)
                path = os.path.join(self.args.checkpoints, setting)
                if not os.path.exists(path):
                    os.makedirs(path)
                # save_path = os.path.join(path, f"{flag}_logits.npy")
                # np.save(save_path, preds.detach().cpu().numpy())
                # print(f"{flag} logits saved to {save_path}")
                
                print(f"{flag} shape:", preds.shape, trues.shape)

                probs = torch.nn.functional.softmax(
                    preds
                )  # (total_samples, num_classes) est. prob. for each class and sample
                predictions = (
                    torch.argmax(probs, dim=1).cpu().numpy()
                )  # (total_samples,) int class index for each sample
                trues = trues.flatten().cpu().numpy()
                accuracy = cal_accuracy(predictions, trues)

                # result save
                folder_path = "./results/" + setting + "/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                print(f"{flag}_accuracy: {accuracy}")
                file_name = f"result_classification_{flag}.txt"
                f = open(os.path.join(folder_path, file_name), "a")
                f.write(setting + "  \n")
                f.write(f"{flag}_accuracy: {accuracy}")
                f.write("\n")
                f.write("\n")
                f.close()

            except Exception as e:
                print(f"Error processing {flag} data: {e}")
                continue