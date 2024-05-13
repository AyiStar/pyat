from typing import Dict, List, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from .base import BaseStrategy
from pyat.model.base.irt import ItemResponseTheoryModel
from ._utils import get_grad_embeddings
from .build import STRATEGY_REGISTRY


@STRATEGY_REGISTRY.register()
class MAATStrategy(BaseStrategy):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.item_data = cfg["item_data"]
        self.k = cfg["k"]
        self.emb_policy = cfg["emb_policy"]
        assert self.emb_policy in ("pseudo", "expected")
        # init knowledge weights
        # for simplicity, use ratio of questions instead
        self.knowledge_weights = {}
        for item_no, item_attrs in enumerate(self.item_data):
            for knowledge_id in item_attrs["knowledge"]:
                self.knowledge_weights.setdefault(knowledge_id, 0)
                self.knowledge_weights[knowledge_id] += 1
        norm_sum = sum(self.knowledge_weights.values())
        for knowledge_id in self.knowledge_weights:
            self.knowledge_weights[knowledge_id] = float(
                self.knowledge_weights[knowledge_id]
            ) / float(norm_sum)

    def select_item(self, session: Dict) -> int:
        unselected = list(session["unselected"])
        # if isinstance(session['base_model'], VanillaIRTModel):
        #     selection = max(list(unselected), key=lambda item_no: self._expected_model_change(item_no, session))
        # else:
        #     selection = unselected[np.argmax(self._expected_model_change_batch(unselected, session))]
        informativeness_scores = self._expected_model_change_batch(unselected, session)
        informativeness_candidates = [
            unselected[i] for i in np.argsort(informativeness_scores)
        ][-self.k :]
        selected_knowledge = set()
        for i in session["selected"]:
            selected_knowledge = selected_knowledge.union(
                set(self.item_data[i]["knowledge"])
            )
        diversity_scores = []
        for item_no in informativeness_candidates:
            knowledge_ids = self.item_data[item_no]["knowledge"]
            diversity_scores.append(
                sum(
                    [
                        self.knowledge_weights[k]
                        for k in knowledge_ids
                        if k not in selected_knowledge
                    ]
                )
            )
        selection = informativeness_candidates[np.argmax(diversity_scores)]
        return selection

    def _expected_model_change_batch(self, item_nos: List[int], session: Dict):
        model = session["base_model"]
        user_no = session["user_no"]
        # get pseudo label
        with torch.no_grad():
            item_nos_input = torch.tensor(item_nos).long().to(model.device)
            user_nos_input = (
                torch.tensor([user_no])
                .long()
                .expand_as(item_nos_input)
                .to(model.device)
            )
            preds = model(item_nos_input).cpu().numpy()
            if self.emb_policy == "pseudo":
                preds = np.where(preds > 0.5, 1, 0)
            else:
                pass

        # if getattr(model, 'update_policy', 'all') == 'last':
        #     all_item_nos = [[item_no] for item_no in item_nos]
        # else:
        #     all_item_nos = [session['selected'] + [item_no] for item_no in item_nos]

        pseudo_labels = [1] * len(item_nos)
        pos_grads = get_grad_embeddings(model, item_nos, pseudo_labels)
        # pos_grads = get_model_changes(model, user_no, item_nos, pseudo_labels, session)
        # grads_norm_pos = np.linalg.norm(pos_grads, axis=1)

        pseudo_labels = [0] * len(item_nos)
        neg_grads = get_grad_embeddings(model, item_nos, pseudo_labels)
        # neg_grads = get_model_changes(model, user_no, item_nos, pseudo_labels, session)
        # grads_norm_neg = np.linalg.norm(neg_grads, axis=1)

        # expected_grads_norm = np.linalg.norm(preds * pos_grads + (1 - preds) * neg_grads, axis=1)
        expected_grads_norm = preds * np.linalg.norm(pos_grads, axis=1) + (
            1 - preds
        ) * np.linalg.norm(neg_grads, axis=1)
        assert len(expected_grads_norm) == len(item_nos)
        return expected_grads_norm

    @staticmethod
    def _expected_model_change(item_no, session):
        model = session["base_model"]
        user_no = session["user_no"]
        if isinstance(model, ItemResponseTheoryModel):
            # get pseudo label
            pred = model.forward(torch.tensor([item_no]).long().to(model.device)).item()
            # pseudo_label = int(pred > 0.5)
            # retrain for positive labels
            tmp_session = {
                "user_no": user_no,
                "all_logs": {item_no: 1, **session["all_logs"]},
                "selected": session["selected"] + [item_no],
            }
            tmp_model = deepcopy(model)
            tmp_model.update_user_params(tmp_session)
            expected_model_change_pos = abs(
                model.user_params.item() - tmp_model.user_params.item()
            )
            # retrain for negative labels
            tmp_session = {
                "user_no": user_no,
                "all_logs": {item_no: 0, **session["all_logs"]},
                "selected": session["selected"] + [item_no],
            }
            tmp_model = deepcopy(model)
            tmp_model.update_user_params(tmp_session)
            expected_model_change_neg = abs(
                model.user_params.item() - tmp_model.user_params.item()
            )
            del tmp_model
            expected_model_change = (
                pred * expected_model_change_pos
                + (1 - pred) * expected_model_change_neg
            )
            return expected_model_change
        else:
            raise NotImplementedError
