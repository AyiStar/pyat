from typing import Dict, List

import torch
import numpy as np

from .base import BaseStrategy
from pyat.model.base.irt import ItemResponseTheoryModel
from .build import STRATEGY_REGISTRY


@STRATEGY_REGISTRY.register()
class MaximumFisherInformationStrategy(BaseStrategy):

    def __init__(self, cfg: Dict):
        super().__init__()

    def select_item(self, session: Dict) -> int:
        unselected = list(session['unselected'])
        selection = unselected[np.argmax(self._fisher_information_batch(unselected, session))]
        return selection

    @staticmethod
    def _fisher_information_batch(item_nos: List, session) -> List:
        model = session['base_model']
        item_nos = torch.tensor(item_nos).long().to(model.device)

        with torch.no_grad():
            if isinstance(model, ItemResponseTheoryModel):
                f_theta = model.forward(item_nos)
                return (f_theta * (1 - f_theta)).squeeze(-1).cpu().numpy().tolist()
            else:
                raise NotImplementedError
