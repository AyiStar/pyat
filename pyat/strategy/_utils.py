from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np

from pyat.model.base.irt import ItemResponseTheoryModel


def get_grad_embeddings(model, user_no: int, item_nos: List[int],
                        pseudo_labels: List[int], use_meta_grad=True, session=None) -> np.ndarray:
    n = len(item_nos)  # len(unselected)
    loss_fn = nn.BCELoss(reduction='sum')
    pseudo_labels = torch.tensor(pseudo_labels).float().to(model.device)
    if isinstance(model, ItemResponseTheoryModel):
        user_params = torch.tensor([[model.user_params.weight[user_no].item()] for _ in range(n)]).requires_grad_().to(model.device)
        item_params = model.item_params(torch.tensor(item_nos).long().to(model.device)).detach().clone().requires_grad_()
        preds = torch.sigmoid(user_params - item_params).squeeze(-1)
        loss = loss_fn(preds, pseudo_labels.clone().detach().float().to(model.device))
        grads = torch.autograd.grad(loss, user_params)[0]
        assert grads.size() == (n, 1), f'Wrong size {grads.size()}'
        return grads.cpu().numpy()
    elif isinstance(model, (FCNModel, NCDModel)):
        item_nos_in = torch.tensor(item_nos).long().to(model.device)
        net = deepcopy(model.net)
        user_params = model.base_params.weight[user_no].detach().clone().unsqueeze(0).expand(n, -1).requires_grad_().to(
            model.device)
        output = net(user_params)
        output = output[torch.arange(output.size(0)).long().to(model.device), item_nos_in]
        preds = torch.sigmoid(output)
        loss = loss_fn(preds, torch.tensor(pseudo_labels).float().to(model.device))
        grads = torch.autograd.grad(loss, user_params)[0]
        assert grads.size() == (n, model.user_dim), f'Wrong size {grads.size()}'
        return grads.cpu().numpy()
    elif isinstance(model, (MetaIRTModel, MetaFCNModel)):
        user_nos_in = torch.tensor([session['user_no']]).long().expand(len(item_nos)).to(model.device)  # useless indeed
        item_nos_in = torch.tensor(item_nos).long().to(model.device)
        param_dict = {k: deepcopy(v) for k, v in model.named_parameters() if k not in model.get_meta_params()}
        model.zero_grad()
        for k in model.get_user_params():
            param_dict[k] = param_dict[k].expand(n, -1).requires_grad_().to(model.device)
        preds = model.forward(user_nos_in, item_nos_in, state_dict=param_dict, use_meta_params=False)
        loss = loss_fn(preds, pseudo_labels)
        params = {k: v for k, v in param_dict.items() if k in sorted(model.get_user_params().keys())}
        grads = torch.autograd.grad(loss, list(params.values()), create_graph=False)
        grad_embs = {}
        if model.mode == 'meta-sgd':
            with torch.no_grad():
                for (k, v), grad in zip(params.items(), grads):
                        grad_embs[k] = (model.get_meta_lrs(prefix=False)[k] * grad
                                        if use_meta_grad else grad.detach.clone())
        elif model.mode == 'dynamic-meta-sgd':
            with torch.no_grad():
                if len(session['meta_grads']) == 0:
                    dynamic_lrs = {k: torch.zeros_like(v).to(model.device) for k, v in model.meta_params.items()}
                else:
                    meta_rnn_input = torch.stack(session['orig_grads'], dim=0).unsqueeze(dim=0)
                    meta_rnn_output, _ = model.meta_rnn(meta_rnn_input)  # (h0, c0 default to zero)
                    dynamic_lrs = model.pack_meta_params(meta_rnn_output[0, -1, :])
                for (k, v), grad in zip(params.items(), grads):
                    static_lr = model.get_meta_lrs(prefix=False)[k]
                    dynamic_lr = dynamic_lrs[k]
                    grad_embs[k] = (((static_lr + model.gamma * dynamic_lr) * grad).detach().clone()
                                    if use_meta_grad else grad.detach().clone())
        tensors = []
        for key in sorted(grad_embs.keys()):
            tensors.append(grad_embs[key])
        return torch.cat(tensors, dim=1).cpu().numpy()
    else:
        raise NotImplementedError


def get_model_changes(model, user_no: int, item_nos: List[int],
                      labels: List[int], session) -> np.ndarray:
    assert isinstance(model, (ItemResponseTheoryModel, FCNModel))
    model_changes = []
    pseudo_session = {
        k: deepcopy(v) for k, v in session.items() if k in ('user_no', 'selected', 'all_logs')
    }
    pseudo_model = deepcopy(model)
    for item_no, label in zip(item_nos, labels):
        pseudo_session['selected'].append(item_no)
        pseudo_session['all_logs'][item_no] = label
        pseudo_model.update_user_params(pseudo_session)
        if isinstance(model, ItemResponseTheoryModel):
            model_change = np.array([pseudo_model.user_params.weight[user_no].item()
                                     - model.user_params.weight[user_no].item()])
        else:
            raise NotImplementedError
        pseudo_session['selected'].remove(item_no)
        model_changes.append(model_change)
    return np.array(model_changes)
