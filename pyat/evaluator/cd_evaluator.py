import copy
import typing
import random

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import pyat.utils.global_logger as glogger
import pyat.utils.global_config as gconfig
from pyat.utils.data import SessionDataset
from pyat.model import BaseModel, MetaModel


class CognitiveDiagnosisEvaluator(object):

    def __init__(self, model: BaseModel, device):
        self.model = model
        self.device = device

    def evaluate(self, data: typing.Dict, cfg: typing.Dict) \
            -> typing.Union[typing.Dict, typing.List[typing.Dict]]:

        self.model.eval()

        train_rand = random.Random(cfg.get('random_seed', None))
        test_rand = random.Random(cfg.get('random_seed', None))
        noise_rand = random.Random(cfg.get('random_seed', None))

        silent = gconfig.silent
        train_size = cfg['train_size']
        test_ratio = cfg['test_ratio']
        noise_size = cfg.get('noise_size', 0)
        dataset = SessionDataset(data)
        # things to log
        all_trainset = {}
        all_testset = {}
        all_preds = {}
        all_labels = {}
        all_models = {}
        all_users = []
        extended_preds = []
        extended_labels = []

        user_params = {k: nn.Parameter(torch.zeros(len(dataset), *p.detach().shape).to(self.device),
                       requires_grad=True)
                       for k, p in self.model.get_user_params().items()}
        for k, p in user_params.items():
            nn.init.xavier_normal_(p)

        for j in tqdm(range(len(dataset)), disable=silent):
            session = dataset[j]
            all_items = sorted(list(session['all_logs'].keys()))
            test_size = int(len(all_items) * test_ratio)
            test_items = test_rand.sample(all_items, test_size)
            candidate_items = sorted(list(set(all_items) - set(test_items)))
            train_items = train_rand.sample(candidate_items, min(train_size, len(candidate_items)))
            session['selected'] = train_items
            session['unselected'] = list(set(all_items).difference(set(session['selected'])))
            session['item_data'] = data['item_data']
            # add noise if set
            noise_items = noise_rand.sample(train_items, noise_size)
            for item_no in noise_items:
                session['all_logs'][item_no] = 1 - session['all_logs'][item_no]
            # start session
            self.model.init_user_params(params={k: p[j].data for k, p in user_params.items()})
            # print(self.model.get_user_params())
            self.model.update_user_params(session)
            # print(self.model.get_user_params())
            item_nos = list(session['all_logs'].keys())
            labels = [session['all_logs'][i] for i in item_nos]
            item_nos_t = torch.tensor(item_nos).long().to(self.device)
            preds = self.model(item_nos_t).detach().cpu().numpy().tolist()

            pred_dict = {i: p for i, p in zip(item_nos, preds)}
            label_dict = {i: p for i, p in zip(item_nos, labels)}
            extended_preds.extend([pred_dict[i] for i in test_items])
            extended_labels.extend([label_dict[i] for i in test_items])

            user_no = session['user_no']
            all_users.append(user_no)
            all_trainset[user_no] = session['selected'].copy()
            all_testset[user_no] = test_items.copy()
            all_preds[user_no] = pred_dict
            all_labels[user_no] = label_dict
            if isinstance(self.model, MetaModel):
                all_models[user_no] = {k: v.detach() for k, v in self.model.adaptive_hyper_model.items()}
            else:
                all_models[user_no] = {k: v.detach() for k, v in self.model.state_dict().items()}

        log = {
            user_no: {
                'labels': all_labels[user_no],
                'predictions': all_preds[user_no],
                'train_set': all_trainset[user_no],
                'test_set': all_testset[user_no],
                'model': all_models[user_no]
            }
            for user_no in all_users
        }

        # get results
        preds, labels = np.array(extended_preds), np.array(extended_labels, dtype=int)
        bin_preds = np.where(preds > 0.5, 1, 0)
        acc = accuracy_score(labels, bin_preds)
        auc = roc_auc_score(labels, preds)
        rmse = np.sqrt(np.mean((preds - labels) ** 2))
        loss = nn.BCELoss()(torch.from_numpy(preds).double().to(self.device),
                            torch.from_numpy(labels).double().to(self.device))
        results = {'acc': acc, 'auc': auc, 'rmse': rmse, 'loss': loss, 'log': log}

        self.model.train()

        return results
