from typing import Dict, List

import torch
import numpy as np

from .base import BaseStrategy
from .build import STRATEGY_REGISTRY


@STRATEGY_REGISTRY.register()
class MaximumEntropyStrategy(BaseStrategy):
    def __init__(self, cfg: Dict):
        super().__init__()

    def select_item(self, session: Dict) -> int:
        unselected = list(session["unselected"])
        selection = unselected[np.argmax(self._entropy_batch(unselected, session))]
        return selection

    @staticmethod
    def _entropy_batch(item_nos: List, session) -> List:
        model = session["base_model"]
        user_no = session["user_no"]
        item_nos = torch.tensor(item_nos).long().to(model.device)
        user_nos = torch.tensor([user_no]).long().expand_as(item_nos).to(model.device)

        with torch.no_grad():
            preds = model(user_nos, item_nos)
            ents = -(
                preds * torch.log(preds + 1e-7)
                + (1 - preds) * torch.log(1.0000001 - preds)
            )
            return ents.cpu().numpy().tolist()
