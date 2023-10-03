import os
import math
import copy
import typing

import torch
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from pyat.utils.data import SessionDataset, session_dataset_collate_fn
import pyat.utils.global_logger as glogger
import pyat.utils.global_config as gconfig
import pyat.utils.global_writer as gwriter
from pyat.evaluator import CognitiveDiagnosisEvaluator
from pyat.model.base_classes import MetaModel
from pyat.strategy import BaseStrategy, build_strategy


class MetaTrainer:

    def __init__(self, model: MetaModel, device):
        self.model = model
        self.train_strategy: BaseStrategy = build_strategy('RandomStrategy', {})
        self.device = device
        self.model.to(device)
        self.silent = gconfig.silent
        self.best_model_snapshot = None
        self.best_result = None

    def fit(self, train_data: typing.Dict, val_data: typing.Optional[typing.Dict], test_data: typing.Optional[typing.Dict], cfg):

        def generate_batch(batch):
            # initialize session
            sessions = []
            for session in batch:
                session['unselected'] = list(session['all_logs'].keys())
                session['selected'] = []
                session['base_model'] = self.model
                session['item_data'] = train_data['item_data']
                session['meta_grads'] = []
                session['orig_grads'] = []
                sessions.append(session)
            return sessions

        # check args and set up configurations
        needed_keys = ['n_epochs', 'lr', 'meta_lr', 'batch_size', 'val_freq', 'test_length', 'each_step']
        assert all(k in cfg for k in needed_keys), \
            f'Missing key(s) in cfg: {[k for k in needed_keys if k not in cfg]}'

        n_epochs = cfg.n_epochs
        lr = cfg.lr  # for item calibration
        meta_lr = cfg.meta_lr  # for meta train
        batch_size = cfg.batch_size
        val_freq = cfg.val_freq
        test_freq = cfg.test_freq
        test_length = cfg.test_length
        n_wait_epochs = cfg.n_wait_epochs
        n_item_epochs = cfg.n_item_epochs
        each_step = cfg.each_step

        train_dataset = SessionDataset(train_data)
        train_dataloader = data.DataLoader(train_dataset, collate_fn=session_dataset_collate_fn,
                                           batch_size=batch_size, shuffle=True, num_workers=0)

        meta_params = list(self.model.get_meta_params().values())
        vanilla_optimizer = optim.Adam(self.model.base_model.parameters(), lr=lr)
        if len(meta_params) > 0:
            meta_optimizer = optim.Adam(meta_params, lr=meta_lr)
        else:
            glogger.logger.warn('Meta-training with no meta-parameters')
            meta_optimizer = None

        evaluator = CognitiveDiagnosisEvaluator(self.model, device=self.device)

        # meta-training and validation
        self.model.train()
        best_ep = 0
        for ep in range(1, n_epochs + 1):
            glogger.logger.info(f'Epoch {ep} meta training')
            # for debug
            """
            if self.model.base_model.__class__.__name__ == 'ItemResponseTheoryModel':
                if self.model.__class__.__name__ == 'ModelAgnosticMetaLearning':
                    glogger.logger.debug(f'meta-learned theta={self.model.hyper_model.base_params["user_params"].item()}')
                if self.model.__class__.__name__ == 'AmortizedBayesianMetaLearning':
                    glogger.logger.debug(f'meta-learned mean={self.model.hyper_model.mean["user_params"].item()}')
                    glogger.logger.debug(f'meta-learned std={math.exp(self.model.hyper_model.log_std["user_params"].item())}')
            """

            # train and record
            ep_meta_losses = []  # for train loss record
            ep_preds, ep_labels = [], []  # for train acc and auc record
            for batch in tqdm(train_dataloader, disable=self.silent):
                sessions = generate_batch(batch)
                # Run inner loops to get adapted parameters
                batch_adapted_hyper_models = []
                batch_meta_loss = []
                for session in sessions:
                    adapted_hyper_model, meta_loss, preds, labels = self._adapt_and_evaluate_on_session(
                        session, test_length, each_step)
                    batch_meta_loss.append(meta_loss)
                    batch_adapted_hyper_models.append(adapted_hyper_model)
                    ep_preds.extend(preds.detach().cpu().numpy().tolist())
                    ep_labels.extend(labels.detach().cpu().numpy().tolist())
                meta_loss = sum(batch_meta_loss) / float(len(batch_meta_loss))
                ep_meta_losses.append(meta_loss.item())
                vanilla_optimizer.zero_grad()
                if meta_optimizer is not None:
                    meta_optimizer.zero_grad()
                meta_loss.backward()
                if ep <= n_item_epochs:
                    vanilla_optimizer.step()
                if meta_optimizer is not None:
                    meta_optimizer.step()

            # track training with logger and writer
            ep_meta_loss = sum(ep_meta_losses) / len(ep_meta_losses)
            gwriter.writer.add_scalar('Loss/train', ep_meta_loss, ep)
            ep_preds_np, ep_labels_np = np.array(ep_preds), np.array(ep_labels, dtype=int)
            bin_preds_np = np.where(ep_preds_np > 0.5, 1, 0)
            ep_acc = accuracy_score(ep_labels_np, bin_preds_np)
            ep_auc = roc_auc_score(ep_labels_np, ep_preds_np)
            gwriter.writer.add_scalar('Acc/train', ep_acc, ep)
            gwriter.writer.add_scalar('Auc/train', ep_auc, ep)
            glogger.logger.exp(f'[train] Epoch {ep}: acc={ep_acc:.4f}, auc={ep_auc:.4f}, meta loss={ep_meta_loss:.4f}')

            # validation per `val_freq` epochs
            if (val_data is not None) and (ep % val_freq == 0):
                glogger.logger.info(f'Epoch {ep}: meta validation')
                evaluator_cfg = {
                    'train_size': test_length,
                    'test_ratio': 0.25,
                    'random_seed': 0
                }
                val_result = evaluator.evaluate(val_data, evaluator_cfg)
                glogger.logger.exp(f'[val]Epoch {ep}: acc={val_result["acc"]:.4f}, '
                                   f'auc={val_result["auc"]:.4f}, '
                                   f'loss={val_result["loss"]:.4f}')
                gwriter.writer.add_scalar('Loss/val', val_result['loss'], ep)
                gwriter.writer.add_scalar('Acc/val', val_result['acc'], ep)
                gwriter.writer.add_scalar('Auc/val', val_result['auc'], ep)
                # save best model
                if (self.best_model_snapshot is None) or \
                        all(val_result[k] >= self.best_result[k] for k in ('acc', 'auc')):
                    self.best_model_snapshot = self._model_snapshot()
                    self.best_result = val_result
                    model_save_file = os.path.join(gconfig.config.output_dir,
                        f'{gconfig.config.meta_model.name}_{gconfig.config.base_model.name}_'
                        f'{gconfig.config.dataset_name}_{gconfig.config.dataset_seed}.pt')
                    self.save_model(model_save_file)
                if any(val_result[k] > self.best_result[k] for k in ('acc', 'auc')) or \
                        (val_result['loss'] < self.best_result['loss']):
                    best_ep = ep
                elif ep - best_ep > n_wait_epochs:
                    glogger.logger.info('Early Stopped')
                    break

            # test per `test_freq` epochs, with the best model selected by validation
            if (test_data is not None) and (self.best_model_snapshot is not None) and (ep % test_freq == 0):
                glogger.logger.info(f'Epoch {ep}: meta test')
                best_model = copy.deepcopy(self.model).to(self.device)
                best_model.load_state_dict(self.best_model_snapshot['model'])
                test_evaluator = CognitiveDiagnosisEvaluator(best_model, device=self.device)
                for length in [3, 5, 10, 1000]:
                    test_results = []
                    evaluator_cfg = {
                        'train_size': length,
                        'test_ratio': 0.25,
                        'random_seed': 0
                    }
                    for _ in range(5):
                        test_result = test_evaluator.evaluate(test_data, evaluator_cfg)
                        test_results.append(test_result)
                    acc, auc, loss = [sum(d[k] for d in test_results) / len(test_results)
                                      for k in ('acc', 'auc', 'loss')]
                    glogger.logger.exp(f'[test]Epoch {ep}: acc={acc:.4f}, '
                                       f'auc={auc:.4f}, '
                                       f'loss={loss:.4f}')
                    gwriter.writer.add_scalar('Loss/test', loss, ep)
                    gwriter.writer.add_scalar('Acc/test', acc, ep)
                    gwriter.writer.add_scalar('Auc/test', auc, ep)

    def _adapt_and_evaluate_on_session(self, session: typing.Dict, test_length: int, each_step: bool) \
            -> typing.Tuple[typing.Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:

        def _adapt_current_session(adapted_hyper_model_):
            # compute model output and loss
            item_nos = session['selected'][-1:] if self.model.update_policy == 'last' else session['selected']
            labels = torch.tensor([session['all_logs'][i] for i in item_nos]).float().to(self.device)
            item_nos = torch.tensor(item_nos).long().to(self.device)
            return self.model.adaptation(item_nos, labels, adapted_hyper_model=adapted_hyper_model_)

        adapted_hyper_model = dict(self.model.hyper_model.named_parameters())
        for i in range(test_length):
            # select item
            selection = self.train_strategy.select_item(session)
            # update session
            session['unselected'].remove(selection)
            session['selected'].append(selection)
            if each_step:
                adapted_hyper_model = _adapt_current_session(adapted_hyper_model)

        if not each_step:
            adapted_hyper_model = _adapt_current_session(adapted_hyper_model)

        val_item_nos, val_labels = zip(*(session['all_logs'].items()))
        val_item_nos = torch.LongTensor(val_item_nos).to(self.device)
        val_labels = torch.tensor(val_labels).float().to(self.device)
        meta_loss, preds = self.model.validation(val_item_nos, val_labels, adapted_hyper_model,
                                                 return_prediction=True)
        return adapted_hyper_model, meta_loss, preds, val_labels

    def _model_snapshot(self):
        snapshot = {
            'config': gconfig.config,
            'model': self.model.state_dict()
        }
        return snapshot

    def get_best_model(self):
        return self.best_model_snapshot['model']

    def save_model(self, save_file_path):
        if self.best_model_snapshot is not None:
            saved_obj = self.best_model_snapshot
        else:
            saved_obj = self._model_snapshot()
        torch.save(saved_obj, save_file_path)
