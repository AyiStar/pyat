import os
import errno
from typing import List, Dict, Union, Tuple

from omegaconf import DictConfig
import torch
import torch.nn as nn


def mkdir_for_file(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def make_knowledge_embs(
        item_data: List[Dict],
        item_nos: Union[List[int], torch.LongTensor],
        num_knowledge: int) -> torch.Tensor:
    embs = []
    for item_no in item_nos:
        emb = [1 if i in item_data[item_no]['knowledge'] else 0 for i in range(num_knowledge)]
        embs.append(torch.tensor(emb).float())
    return torch.stack(embs, dim=0)


def get_dataset_info(data_obj):
    return {
        'num_users': data_obj['meta_data']['num_users'],
        'num_items': data_obj['meta_data']['num_items'],
        'num_knowledge': data_obj['meta_data']['num_knowledge'],
        'num_logs': data_obj['meta_data']['num_logs'],
        'process_args': data_obj['process_args']
    }


def get_adapt_term(static_lr, dynamic_lr, orig_grad, act_dyna=None, act_total=None):
    if act_dyna is not None:
        dynamic_lr = act_dyna(dynamic_lr)
    if act_total is not None:
        return act_total(static_lr + dynamic_lr)
    else:
        return static_lr + dynamic_lr


def get_tags(cfg: DictConfig) -> Tuple[List[str], List[str]]:
    model_tags = [
        cfg.meta_model.name if 'meta_model' in cfg else "base",
        cfg.base_model.name,
        cfg.dataset_name,
        str(cfg.dataset_seed)
    ]
    exp_tags = model_tags[:]
    if 'evaluator' in cfg:
        if cfg.evaluator.name == 'cd':
            exp_tags.append(str(cfg.evaluator.train_size))
            exp_tags.append(str(cfg.evaluator.random_seed))
    return model_tags, exp_tags
