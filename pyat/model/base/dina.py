from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyat.utils import make_knowledge_embs

from ..base_classes import BaseModel
import pyat.utils.global_logger as global_logger
from ..build import BASE_MODEL_REGISTRY


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


@BASE_MODEL_REGISTRY.register()
class DeterministicInputNoisyAndGateModel(BaseModel):
    def __init__(self, cfg: Dict):
        nn.Module.__init__(self)
        self.cfg = cfg
        # set up network structure parameters
        self.num_knowledge = cfg["num_knowledge"]
        self.max_slip = cfg["max_slip"]
        self.max_guess = cfg["max_guess"]
        # set up network structure
        self.user_params = nn.Parameter(
            torch.zeros(self.num_knowledge), requires_grad=True
        )
        self.item_guess = nn.Embedding(cfg["num_items"], 1)
        self.item_slip = nn.Embedding(cfg["num_items"], 1)
        self.sign = StraightThroughEstimator()

        self.item_data = self.cfg["item_data"]
        self.update_policy = cfg["update_policy"]
        assert self.update_policy in ("last", "all")
        self.update_max_loop = cfg["update_max_loop"]
        self.update_lr = cfg["update_lr"]
        self.device = self.cfg["device"]

        # cache all the knowledge embeddings
        self.knowledge_embs = make_knowledge_embs(
            self.item_data, list(range(len(self.item_data))), self.num_knowledge
        ).to(self.device)
        # init or load pretrained
        self._init()

    def _init(self):
        nn.init.normal_(self.user_params)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, item_nos: torch.LongTensor) -> torch.Tensor:
        knowledge_embs = self.knowledge_embs[item_nos, :]
        stu_emb = self.sign(self.user_params)
        slip = torch.squeeze(torch.sigmoid(self.item_slip(item_nos)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.item_guess(item_nos)) * self.max_guess)
        mask_theta = (knowledge_embs == 0) + (knowledge_embs == 1) * stu_emb
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        output = torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)
        return output

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
        return {k: v for k, v in self.named_parameters() if k.startswith("user_params")}

    def get_item_params(self) -> Dict:
        return {
            k: v for k, v in self.named_parameters() if not k.startswith("user_params")
        }
