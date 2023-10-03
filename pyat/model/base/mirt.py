from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_classes import BaseModel
import pyat.utils.global_logger as global_logger
from ..build import BASE_MODEL_REGISTRY


@BASE_MODEL_REGISTRY.register()
class MultidimensionalItemResponseTheoryModel(BaseModel):

    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.hidden_dim = cfg['hidden_dim']
        # set up vanilla params
        self.user_params = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.item_a = nn.Embedding(cfg['num_items'], self.hidden_dim)
        self.item_b = nn.Embedding(cfg['num_items'], 1)

        self.device = self.cfg['device']
        # update policy
        self.update_policy = self.cfg['update_policy']
        self.update_lr = self.cfg['update_lr']
        self.update_max_loop = self.cfg['update_max_loop']
        # init or load pretrained
        self._init()

    def _init(self):
        nn.init.normal_(self.user_params)
        nn.init.xavier_normal_(self.item_a.weight)
        nn.init.xavier_normal_(self.item_b.weight)

    def forward(self, item_nos: torch.LongTensor) -> torch.Tensor:
        if item_nos.dim() == 0:
            item_nos = item_nos.unsqueeze(0)
        theta = self.user_params.unsqueeze(0)
        a = self.item_a(item_nos)
        b = self.item_b(item_nos)
        output = torch.sigmoid(torch.sum(torch.multiply(a, theta), keepdim=True, dim=-1) - b).squeeze()
        return output

    def init_user_params(self, params=None):
        if params is not None:
            with torch.no_grad():
                for k, v in params.items():
                    getattr(self, k).data = v
        else:
            nn.init.normal_(self.user_params)

    def get_user_params(self) -> Dict[str, nn.Parameter]:
        return {k: v for k, v in self.named_parameters() if
                k.startswith('user_params')}

    def get_item_params(self) -> Dict:
        return {k: v for k, v in self.named_parameters() if
                not k.startswith('user_params')}

    def update_user_params(self, session, **kwargs):
        super().update_user_params(session)
