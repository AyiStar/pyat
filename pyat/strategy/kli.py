from typing import Dict, List

import torch

# import vegas
import numpy as np

from .base import BaseStrategy
from pyat.model.base.irt import ItemResponseTheoryModel
from .build import STRATEGY_REGISTRY


@STRATEGY_REGISTRY.register()
class KullbackLeiblerInformationStrategy(BaseStrategy):
    def __init__(self, cfg: Dict):
        super().__init__()

    def select_item(self, session: Dict) -> int:
        unselected = list(session["unselected"])
        selection = unselected[
            np.argmax(
                self._KL_information_batch(
                    unselected, session, k=len(session["selected"])
                )
            )
        ]
        return selection

    @staticmethod
    def _KL_divergence_batch(
        beta: torch.Tensor, theta: torch.Tensor, theta0: torch.Tensor
    ):
        """calculate KL(theta||theta0)"""
        f_theta = torch.sigmoid(theta - beta)
        f_theta0 = torch.sigmoid(theta0 - beta)
        return f_theta0 * (
            torch.log(0.0001 + f_theta0) - torch.log(0.0001 + f_theta)
        ) + (1.0001 - f_theta0) * (
            torch.log(1.0001 - f_theta0) - torch.log(1.0001 - f_theta)
        )

    @staticmethod
    def _KL_information_batch(
        item_nos: List[int], session: Dict, k: int, n: int = 50
    ) -> List:
        model = session["base_model"]
        item_nos_input = torch.tensor(item_nos).long().to(model.device)
        with torch.no_grad():
            if isinstance(model, ItemResponseTheoryModel):
                beta = model.item_params(item_nos_input).reshape(-1)
                theta = model.user_params.item()
            else:
                raise NotImplementedError
            r = max(model._max_user_param - model._min_user_param, 1)
            delta = r / (k + 1) ** 0.5
            dt = (2 * delta) / n
            theta_ = torch.arange(start=theta - delta, end=theta + delta, step=dt).to(
                model.device
            )
            theta_ = theta_.unsqueeze(dim=-1).expand(
                theta_.size(0), item_nos_input.size(0)
            )
            beta = beta.unsqueeze(dim=0).expand_as(theta_)
            theta = torch.full_like(theta_, theta).to(model.device)
            KL = KullbackLeiblerInformationStrategy._KL_divergence_batch(
                beta, theta, theta_
            )
            KI = (KL * dt).sum(dim=0).cpu().numpy().tolist()
            assert len(KI) == len(item_nos)
            return KI
