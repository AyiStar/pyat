import typing

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import pyat.utils.global_logger as glogger
import pyat.utils.global_config as gconfig
from pyat.utils.data import SessionDataset
from pyat.model import BaseModel
from pyat.strategy import BaseStrategy
from pyat.stopping_criterion import BaseStoppingCriterion


class AdaptiveTestingEvaluator(object):

    def __init__(self, model: BaseModel, strategy: BaseStrategy, stopping_criterion: BaseStoppingCriterion, device):

        self.model = model
        self.strategy = strategy
        self.stopping_criterion = stopping_criterion
        self.device = device

    def evaluate(self, data: typing.Dict, cfg: typing.Dict) \
            -> typing.Union[typing.Dict, typing.List[typing.Dict]]:

        self.model.eval()

        silent = gconfig.silent

        def _init_session(sess: typing.Dict) -> typing.Dict:
            sess['unselected'] = list(sess['all_logs'].keys())
            sess['selected'] = []
            sess['base_model'] = self.model
            sess['item_data'] = data['item_data']
            return sess

        dataset = SessionDataset(data)
        all_preds = {}
        all_labels = {}

        for j in tqdm(range(len(dataset)), disable=silent):
            session = _init_session(dataset[j])
            # start session
            self.model.init_user_params()
            step = 0

            while True:
                item_nos = list(session['all_logs'].keys())
                labels = [session['all_logs'][i] for i in item_nos]
                item_nos = torch.tensor(item_nos).long().to(self.device)
                preds = self.model(item_nos).detach().cpu().numpy().tolist()
                all_preds.setdefault(step, [])
                all_labels.setdefault(step, [])
                all_preds[step].extend(preds)
                all_labels[step].extend(labels)
                if not self.stopping_criterion.stop(session):
                    selection = self.strategy.select_item(session)
                    session['unselected'].remove(selection)
                    session['selected'].append(selection)
                    self.model.update_user_params(session)
                    step = step + 1
                else:
                    break

        # get results
        results = {}
        for i in all_preds:
            preds, labels = np.array(all_preds[i]), np.array(all_labels[i], dtype=int)
            bin_preds = np.where(preds > 0.5, 1, 0)
            acc = accuracy_score(labels, bin_preds)
            auc = roc_auc_score(labels, preds)
            rmse = np.sqrt(np.mean((preds - labels) ** 2))
            loss = nn.BCELoss()(torch.from_numpy(preds).double().to(self.device),
                                torch.from_numpy(labels).double().to(self.device))
            results[i] = {'acc': acc, 'auc': auc, 'rmse': rmse, 'loss': loss}

        self.model.train()

        # TODO return log like cd_evaluator
        return results

    def run_session(self, session: typing.Dict) -> None:
        # select item
        selection = self.strategy.select_item(session)
        # update session
        session['unselected'].remove(selection)
        session['selected'].append(selection)
        # update model
        self.model.update_user_params(session)
