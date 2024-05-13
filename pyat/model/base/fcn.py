from typing import Dict

import torch
import torch.nn as nn

from pyat.model.base_classes import BaseModel
from pyat.model.build import BASE_MODEL_REGISTRY


@BASE_MODEL_REGISTRY.register()
class FullyConnectedNetworkModel(BaseModel):
    def __init__(self, cfg: Dict):
        nn.Module.__init__(self)
        self.cfg = cfg
        # set up network structure parameters
        self.user_dim = cfg["user_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        # set up network structure
        self.user_params = nn.Parameter(torch.zeros(self.user_dim), requires_grad=True)
        self.net = nn.Sequential(
            nn.Linear(self.user_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, cfg["num_items"]),
        )
        self.update_policy = cfg.get("update_policy", "all")
        assert self.update_policy in ("last", "all")
        self.update_max_loop = cfg.get("update_max_loop", 20)
        self.update_lr = cfg.get("update_lr", 0.0005)
        self.device = self.cfg["device"]

        # init or load pretrained
        self._init()

    def _init(self):
        nn.init.normal_(self.user_params)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, item_nos: torch.LongTensor) -> torch.Tensor:
        user_params = self.user_params
        output = self.net(user_params)
        output = output[item_nos]
        output = torch.sigmoid(output)
        return output

    def init_user_params(self, params=None):
        if params is not None:
            with torch.no_grad():
                for k, v in params.items():
                    getattr(self, k).data = v
        else:
            nn.init.normal_(self.user_params)

    def update_user_params(self, session):
        super().update_user_params(session)

    def get_user_params(self) -> Dict:
        return {k: v for k, v in self.named_parameters() if k.startswith("user_params")}

    def get_item_params(self) -> Dict:
        return {k: v for k, v in self.named_parameters() if k.startswith("net")}
