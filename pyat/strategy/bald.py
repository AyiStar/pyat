from typing import Dict, List

import torch
import numpy as np

from .base import BaseStrategy
from scipy.special import xlogy
from pyat.model.meta.abml import AmortizedBayesianMetaLearning
from .build import STRATEGY_REGISTRY


@STRATEGY_REGISTRY.register()
class BayesianActiveLearningByDisagreementStrategy(BaseStrategy):

    def __init__(self, cfg: Dict):
        super().__init__()
        self.num_samples = cfg['num_samples']

    def select_item(self, session: Dict) -> int:
        unselected = list(session['unselected'])
        selection = unselected[np.argmax(self._score_batch(unselected, session))]
        return selection

    def _score_batch(self, item_nos: List, session) -> List:
        model = session['base_model']
        item_nos = torch.tensor(item_nos).long().to(model.device)

        with torch.no_grad():
            if isinstance(model, AmortizedBayesianMetaLearning):
                sampled_predictions = [model.prediction(item_nos, adapted_hyper_model=model.adaptive_hyper_model)
                                       for _ in range(self.num_samples)]
                item_preds = torch.transpose(torch.stack(sampled_predictions), dim0=0, dim1=1)
                item_preds = torch.stack([item_preds, 1 - item_preds], dim=2).cpu().numpy()
                # item_preds: [num_items, num_samples, 2]
                expected_entropy = np.mean(-np.sum(xlogy(item_preds, item_preds), axis=-1), axis=1)
                expected_p = np.mean(item_preds, axis=1)
                entropy_expected_p = -np.sum(xlogy(expected_p, expected_p), axis=-1)
                bald_acq = entropy_expected_p - expected_entropy
                return bald_acq
            else:
                raise NotImplementedError
