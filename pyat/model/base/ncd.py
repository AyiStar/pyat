from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyat.utils import make_knowledge_embs
import pyat.utils.global_logger as glogger
from pyat.model.base_classes import BaseModel
from pyat.model.build import BASE_MODEL_REGISTRY


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


@BASE_MODEL_REGISTRY.register()
class NeuralCognitiveDiagnosisModel(BaseModel):

    def __init__(self, cfg: Dict):
        nn.Module.__init__(self)
        self.cfg = cfg
        # set up network structure parameters
        self.num_knowledge = cfg['num_knowledge']
        self.prednet_len_1 = cfg['prednet_len_1']
        self.prednet_len_2 = cfg['prednet_len_2']
        self.dropout_rate = cfg['dropout_rate']
        # set up network structure
        self.user_params = nn.Parameter(torch.zeros(self.num_knowledge), requires_grad=True)
        self.knowledge_difficulty = nn.Embedding(cfg['num_items'], self.num_knowledge)
        self.item_difficulty = nn.Embedding(cfg['num_items'], 1)
        if self.prednet_len_2 is not None:
            self.net = nn.Sequential(
                PosLinear(self.num_knowledge, self.prednet_len_1),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                PosLinear(self.prednet_len_1, self.prednet_len_2),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                PosLinear(self.prednet_len_2, 1)
            )
        else:
            self.net = nn.Sequential(
                PosLinear(self.num_knowledge, self.prednet_len_1),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                PosLinear(self.prednet_len_1, 1)
            )

        self.item_data = self.cfg['item_data']
        self.update_policy = cfg['update_policy']
        assert self.update_policy in ('last', 'all')
        self.update_max_loop = cfg['update_max_loop']
        self.update_lr = cfg['update_lr']
        self.device = self.cfg['device']

        # cache all the knowledge embeddings
        self.knowledge_embs = make_knowledge_embs(self.item_data, list(range(len(self.item_data)))
                                                  , self.num_knowledge).to(self.device)
        # init or load pretrained
        self._init()

    def _init(self):
        nn.init.normal_(self.user_params)
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, item_nos: torch.LongTensor) -> torch.Tensor:
        knowledge_embs = self.knowledge_embs[item_nos, :]
        stu_emb = torch.sigmoid(self.user_params)
        k_difficulty = torch.sigmoid(self.knowledge_difficulty(item_nos))
        i_difficulty = torch.sigmoid(self.item_difficulty(item_nos))  # * 10
        input_x = i_difficulty * (stu_emb - k_difficulty) * knowledge_embs
        output = torch.sigmoid(self.net(input_x)).squeeze(-1)
        return output

    # def apply_clipper(self):
    #     clipper = NoneNegClipper()
    #     self.net.apply(clipper)

    def init_user_params(self, params=None):
        if params is not None:
            with torch.no_grad():
                for k, v in params.items():
                    getattr(self, k).data = v
        else:
            nn.init.normal_(self.user_params)

    def update_user_params(self, session, **kwargs):
        super().update_user_params(session)

    def get_user_params(self) -> Dict:
        return {k: v for k, v in self.named_parameters() if
                k.startswith('user_params')}

    def get_item_params(self) -> Dict:
        return {k: v for k, v in self.named_parameters() if
                not k.startswith('user_params')}


"""
class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            module.weight.data = torch.clamp(w, min=0.).detach()
"""