from typing import Union, Optional, Dict, Tuple, List
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.func import functional_call

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from omegaconf import DictConfig

import pyat.utils.global_logger as glogger
import pyat.utils.global_config as gconfig
import pyat.utils.global_writer as gwriter
from pyat.model.base_classes import BaseModel
from pyat.utils.data import SessionDataset, session_dataset_collate_fn


class BaseTrainer:
    def __init__(self, model: BaseModel, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.silent = gconfig.silent
        self.best_model_snapshot = None
        self.best_result = None
        self.user_params = None

    def fit(
        self,
        train_data: Dict,
        val_data: Optional[Dict],
        test_data: Optional[Dict],
        cfg: DictConfig,
    ):
        # Dict is the type of a preprocessed data object
        assert all(
            k in cfg for k in ["n_epochs", "lr", "batch_size", "val_freq"]
        ), "Missing key(s) in cfg"
        self.best_model_snapshot = None
        self.best_result = None

        n_epochs = cfg.n_epochs
        lr = cfg.lr
        batch_size = cfg.batch_size
        val_freq = cfg.val_freq

        self.model.train()
        train_dataset = SessionDataset(train_data)
        val_dataset = SessionDataset(val_data)
        self.user_params = {
            k: nn.Parameter(
                torch.zeros(train_dataset.num_users, *p.detach().shape).to(self.device),
                requires_grad=True,
            )
            for k, p in self.model.get_user_params().items()
        }
        for k, p in self.user_params.items():
            nn.init.xavier_normal_(p)
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=max(1, batch_size // 16),
            collate_fn=session_dataset_collate_fn,
            shuffle=True,
            num_workers=0,
        )
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(
            list(self.user_params.values())
            + list(self.model.get_item_params().values()),
            lr=lr,
        )

        for ep in range(1, n_epochs + 1):
            glogger.logger.info(f"Epoch {ep} vanilla training")

            ep_losses, ep_preds, ep_labels = [], [], []
            for sessions in tqdm(train_dataloader, disable=self.silent):
                user_no_list, item_nos_list, labels = self._generate_train_batch(
                    sessions
                )
                preds = torch.concat(
                    [
                        functional_call(
                            self.model,
                            parameter_and_buffer_dicts={
                                k: p[user_no] for k, p in self.user_params.items()
                            },
                            args=(item_nos,),
                        )
                        for user_no, item_nos in zip(user_no_list, item_nos_list)
                    ]
                )
                loss = loss_func(preds, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ep_losses.append(loss.item())
                ep_preds.extend(preds.detach().cpu().numpy().tolist())
                ep_labels.extend(labels.detach().cpu().numpy().tolist())

            ep_loss = sum(ep_losses) / len(ep_losses)
            gwriter.writer.add_scalar("Loss/train", ep_loss, ep)
            ep_preds_np, ep_labels_np = (
                np.array(ep_preds),
                np.array(ep_labels, dtype=int),
            )
            bin_preds_np = np.where(ep_preds_np > 0.5, 1, 0)
            ep_acc = accuracy_score(ep_labels_np, bin_preds_np)
            ep_auc = roc_auc_score(ep_labels_np, ep_preds_np)
            gwriter.writer.add_scalar("Acc/train", ep_acc, ep)
            gwriter.writer.add_scalar("Auc/train", ep_auc, ep)
            glogger.logger.exp(
                f"[train] Epoch {ep}: acc={ep_acc:.4f}, auc={ep_auc:.4f}, loss={ep_loss:.4f} "
            )

            # validation per `val_freq` epochs
            if (val_dataset is not None) and (ep % val_freq == 0):
                glogger.logger.info(f"Epoch {ep}: validating")
                val_result = self._validation(val_dataset)
                glogger.logger.exp(
                    f'[val]Epoch {ep}: acc={val_result["acc"]:.4f}, '
                    f'auc={val_result["auc"]:.4f}, '
                    f'loss={val_result["loss"]:.4f}'
                )
                gwriter.writer.add_scalar("Loss/val", val_result["loss"], ep)
                gwriter.writer.add_scalar("Acc/val", val_result["acc"], ep)
                gwriter.writer.add_scalar("Auc/val", val_result["auc"], ep)

                if (self.best_model_snapshot is None) or all(
                    val_result[k] >= self.best_result[k] for k in ("acc", "auc")
                ):
                    self.best_model_snapshot = self._model_snapshot()
                    self.best_result = val_result

            # TODO test per `test_freq` epochs with the best model selected by validation

    def _validation(self, val_dataset: SessionDataset):
        val_dataloader = data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=session_dataset_collate_fn,
        )
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for sessions in tqdm(val_dataloader, disable=self.silent):
                session = sessions[0]
                user_no = session["user_no"]
                item_nos = list(session["all_logs"].keys())
                labels = [session["all_logs"][i] for i in item_nos]
                item_nos = torch.tensor(item_nos).long().to(self.device)
                preds = functional_call(
                    self.model,
                    {k: p[user_no] for k, p in self.user_params.items()},
                    args=(item_nos,),
                )
                preds = preds.detach().cpu().numpy().tolist()
                all_labels.extend(labels)
                all_preds.extend(preds)
        self.model.train()
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_bin_preds = np.where(all_preds > 0.5, 1, 0)
        acc = accuracy_score(all_labels, all_bin_preds)
        auc = roc_auc_score(all_labels, all_preds)
        loss = nn.BCELoss()(
            torch.from_numpy(all_preds).double().to(self.device),
            torch.from_numpy(all_labels).double().to(self.device),
        )
        return {"acc": acc, "auc": auc, "loss": loss}

    def _model_snapshot(self):
        snapshot = {"config": gconfig.config, "model": self.model.state_dict()}
        return snapshot

    def get_best_model(self):
        return self.best_model_snapshot["model"]

    def save_model(self, save_file_path):
        if self.best_model_snapshot is not None:
            saved_obj = self.best_model_snapshot
        else:
            saved_obj = self._model_snapshot()
        torch.save(saved_obj, save_file_path)

    def _generate_train_batch(
        self, sessions, sample_size=16
    ) -> Tuple[List[int], List[torch.LongTensor], torch.FloatTensor]:
        user_no_list, item_nos_list, labels = [], [], []
        for sess in sessions:
            user_no_list.append(sess["user_no"])
            samples = (
                random.sample(list(sess["all_logs"].keys()), sample_size)
                if sample_size < len(sess["all_logs"])
                else list(sess["all_logs"].keys())
            )
            labels.extend([sess["all_logs"][i] for i in samples])
            item_nos_list.append(torch.LongTensor(samples).to(self.device))
        return user_no_list, item_nos_list, torch.FloatTensor(labels).to(self.device)
