from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq

from ..base_classes import BaseModel
import pyat.utils.global_logger as global_logger
from ..build import BASE_MODEL_REGISTRY


@BASE_MODEL_REGISTRY.register()
class ItemResponseTheoryModel(BaseModel, nn.Module):

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        # set up vanilla params
        self.user_params = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        self.item_params = nn.Embedding(cfg['num_items'], 1)
        self.device = self.cfg['device']
        # update policy
        self.update_policy = self.cfg.get('update_policy', None)
        self.update_lr = self.cfg.get('update_lr', None)
        self.update_max_loop = self.cfg.get('update_max_loop', None)
        # init or load pretrained
        self._init()

    def _init(self):
        nn.init.normal_(self.user_params)
        nn.init.xavier_normal_(self.item_params.weight)
        # below for optimization
        self._item_params_npy = None
        self._min_user_param = -10
        self._max_user_param = 10

    def forward(self, item_nos: torch.LongTensor) -> torch.Tensor:
        user_params = self.user_params
        item_params = self.item_params(item_nos)
        return torch.sigmoid(user_params - item_params).squeeze(-1)

    def init_user_params(self, params=None):
        if params is not None:
            with torch.no_grad():
                for k, v in params.items():
                    getattr(self, k).data = v
        else:
            nn.init.normal_(self.user_params)
            # for optimization
        self._item_params_npy = self.item_params.weight.detach().cpu().numpy()

    def get_user_params(self) -> Dict[str, nn.Parameter]:
        return {k: v for k, v in self.named_parameters() if
                k.startswith('user_params')}

    def get_item_params(self) -> Dict:
        return {k: v for k, v in self.named_parameters() if
                k.startswith('item_params')}

    def update_user_params(self, session):
        if self.update_policy is not None and self.update_lr is not None and self.update_max_loop is not None:
            super().update_user_params(session)
        else:
            self._update_user_params_optim(session)

    def _update_user_params_optim(self, session):
        item_nos = session['selected']
        labels = [session['all_logs'][i] for i in item_nos]

        new_user_param = 0.
        try:
            new_user_param = brentq(lambda x: self._deriv_likelihood(x, item_nos, labels),
                                    self._min_user_param, self._max_user_param)
        except ValueError:
            if all(label == 1 for label in labels):
                new_user_param = self._max_user_param
            elif all(label == 0 for label in labels):
                new_user_param = self._min_user_param
            else:
                f_a = self._deriv_likelihood(self._min_user_param, item_nos, labels)
                f_b = self._deriv_likelihood(self._max_user_param, item_nos, labels)
                global_logger.logger.debug(f'May have same sign: f(a)={f_a:.4f}, f(b)={f_b:.4f}')
                if f_a > 0 and f_b > 0:
                    new_user_param = self._max_user_param
                elif f_a < 0 and f_b < 0:
                    new_user_param = self._min_user_param
                else:
                    global_logger.logger.warn(f'Some error occurs during updating user param, use average user param')

        with torch.no_grad():
            self.user_params.data.copy_(torch.tensor(new_user_param))

    def _deriv_likelihood(self, user_param, item_nos, labels):
        item_params = self._item_params_npy[item_nos].reshape(-1)
        return sum(y - (1.0 / (1.0 + np.exp(-(user_param - item_param)))) for y, item_param in zip(labels, item_params))
