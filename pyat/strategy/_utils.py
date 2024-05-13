from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
from torch.func import functional_call
import numpy as np

from pyat.model.base.irt import ItemResponseTheoryModel
from pyat.model.base.ncd import NeuralCognitiveDiagnosisModel


def get_grad_embeddings(
    model, item_nos: List[int], pseudo_labels: List[int]
) -> np.ndarray:
    n = len(item_nos)
    loss_fn = nn.BCELoss(reduction="sum")
    pseudo_labels_tensor = torch.tensor(pseudo_labels).float().to(model.device)
    item_nos_tensor = torch.tensor(item_nos).long().to(model.device)
    if isinstance(model, (ItemResponseTheoryModel, NeuralCognitiveDiagnosisModel)):
        batched_user_params = (
            model.user_params.data.detach()
            .clone()
            .unsqueeze(0)
            .expand(n, -1)
            .requires_grad_()
            .to(model.device)
        )
        model.requires_grad_(False)
        preds = functional_call(
            model,
            parameter_and_buffer_dicts={"user_params": batched_user_params},
            args=(item_nos_tensor,),
        )
        loss = loss_fn(preds, pseudo_labels_tensor.clone().detach())
        grads = torch.autograd.grad(loss, batched_user_params)[0]
        model.requires_grad_(True)
        assert grads.size() == batched_user_params.size(), f"Wrong size {grads.size()}"
        return grads.cpu().numpy()
    else:
        raise NotImplementedError
