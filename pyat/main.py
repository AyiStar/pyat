import errno
import pickle
import logging
import os.path
import random
import tempfile
import itertools
from datetime import datetime
from typing import List, Dict, Tuple

import torch
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from prettytable import PrettyTable

from pyat.utils.data import split_data_by_user, split_data_by_log
from pyat.model import build_base_model, build_meta_model, BaseModel, MetaModel
from pyat.strategy import build_strategy, BaseStrategy
from pyat.trainer import MetaTrainer, BaseTrainer
from pyat.stopping_criterion import build_stopping_criterion
import pyat.utils.global_logger as global_logger
import pyat.utils.global_writer as global_writer
import pyat.utils.global_config as global_config
from pyat.evaluator import AdaptiveTestingEvaluator, CognitiveDiagnosisEvaluator


def check_config(cfg: DictConfig):
    # TODO
    assert 'base_model' in cfg

    if 'evaluator' in cfg:
        if cfg.evaluator.name == 'at':
            assert 'strategy' in cfg
            assert 'stopping_criterion' in cfg
        if 'trainer' not in cfg:
            assert cfg.get('load_model_dir', None)
    if 'trainer' in cfg:
        if 'meta_model' in cfg:
            assert cfg.trainer.type == 'meta'
        else:
            assert cfg.trainer.type == 'base'
    if 'trainer' not in cfg:
        assert 'load_model_dir' in cfg


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


def show_dataset_info(dataset_name: str,
                      train_dataobj: Dict,
                      test_dataobj: Dict,
                      logger: logging.Logger):
    info_table = PrettyTable()
    info_table.add_column(dataset_name,
                          ["#Users",
                           "#Items",
                           "#Knolwedge",
                           "#Logs",
                           "#Avg.Logs",
                           "#Min.Logs",
                           "#Max.Logs"])
    info_table.add_column("Train",
                          [train_dataobj['meta_data']['num_users'],
                           train_dataobj['meta_data']['num_items'],
                           train_dataobj['meta_data']['num_knowledge'],
                           train_dataobj['meta_data']['num_logs'],
                           sum(list(map(len, train_dataobj['user_data']))) / len(train_dataobj['user_data']),
                           min(list(map(len, train_dataobj['user_data']))),
                           max(list(map(len, train_dataobj['user_data'])))])
    info_table.add_column('Test',
                          [test_dataobj['meta_data']['num_users'],
                           test_dataobj['meta_data']['num_items'],
                           test_dataobj['meta_data']['num_knowledge'],
                           test_dataobj['meta_data']['num_logs'],
                           sum(list(map(len, test_dataobj['user_data']))) / len(test_dataobj['user_data']),
                           min(list(map(len, test_dataobj['user_data']))),
                           max(list(map(len, test_dataobj['user_data'])))])
    info_table.float_format = '.2'
    logger.info('\n' + info_table.get_string())


def main(cfg: DictConfig):
    """ Main function for train and evaluation """


    """ Basic Configuration """
    check_config(cfg)

    base_model_cfg = cfg.base_model
    meta_model_cfg = cfg.get('meta_model', None)
    trainer_cfg = cfg.get('trainer', None)
    strategy_cfg = cfg.get('strategy', None)
    stopping_criterion_cfg = cfg.get('stopping_criterion', None)
    evaluator_cfg = cfg.get('evaluator', None)

    dataset_dir = cfg.dataset_dir
    dataset_name = cfg.dataset_name
    dataset_seed = cfg.dataset_seed

    use_meta_model = (meta_model_cfg is not None)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_tags, exp_tags = get_tags(cfg)

    # set output
    tmp_dir = None
    if cfg.output_dir:
        cfg.output_dir = os.path.join(cfg.output_dir, f'EXP@{datetime.now().strftime("%Y%m%d-%H-%M-%S")}@{"_".join(exp_tags)}')
        if not os.path.exists(cfg.output_dir):
            try:
                os.makedirs(cfg.output_dir)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
    else:
        tmp_dir = tempfile.TemporaryDirectory()
        cfg.output_dir = tmp_dir.name
    log_file_path = os.path.join(cfg.output_dir, 'log.txt')
    cfg.logger.file_path = log_file_path

    global_config.init_config(cfg)
    global_config.silent = cfg.silent
    global_logger.init_logger(cfg.logger)
    global_writer.init_writer(log_dir=cfg.output_dir)
    global_logger.logger.info(OmegaConf.to_yaml(cfg))

    # seed every randomness
    seed_all = cfg.get('seed_all', 0)
    random.seed(seed_all)
    np.random.seed(seed_all)
    torch.manual_seed(seed_all)


    """ Load Data """
    train_data_path = os.path.join(dataset_dir, dataset_name, f'{dataset_name}_train_seed_{dataset_seed}.pkl')
    with open(train_data_path, 'rb') as f:
        all_train_data = pickle.load(f)
        if not use_meta_model:
            train_data, val_data = split_data_by_log(all_train_data, split_ratio=0.8)
        else:
            train_data, val_data = split_data_by_user(all_train_data, split_ratio=0.75)
    test_data_path = os.path.join(dataset_dir, dataset_name, f'{dataset_name}_test_seed_{dataset_seed}.pkl')
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    show_dataset_info(cfg.dataset_name, all_train_data, test_data, global_logger.logger)


    """ Create Model and Strategy """
    load_model_dir = cfg.load_model_dir
    assert (load_model_dir is None) or (os.path.exists(load_model_dir)), f'Invalid load_model_dir: {load_model_dir}'
    with open_dict(cfg):
        cfg.device = device
        base_model_cfg.device = device
        base_model_cfg.num_users = train_data['meta_data']['num_users']
        base_model_cfg.num_items = train_data['meta_data']['num_items']
        base_model_cfg.num_knowledge = train_data['meta_data']['num_knowledge']
        base_model_cfg.item_data = train_data['item_data']
        if meta_model_cfg is not None:
            meta_model_cfg.device = device
        if strategy_cfg is not None:
            strategy_cfg.num_users = train_data['meta_data']['num_users']
            strategy_cfg.num_items = train_data['meta_data']['num_items']
            strategy_cfg.num_knowledge = train_data['meta_data']['num_knowledge']
            strategy_cfg.item_data = train_data['item_data']

    model: BaseModel = build_base_model(base_model_cfg.class_name, base_model_cfg)
    if use_meta_model:
        model: MetaModel = build_meta_model(meta_model_cfg.class_name, model, meta_model_cfg)
    model.to(device)


    """ Load Pretrained Model """
    if load_model_dir is not None:
        global_logger.logger.exp('Loading model')
        load_model_path = os.path.join(load_model_dir, f'{"_".join(model_tags)}.pt')
        saved_snapshot = torch.load(load_model_path, map_location=device)
        model.load_state_dict(saved_snapshot['model'])


    """ Train and Save Model"""
    if trainer_cfg is not None:
        global_logger.logger.exp('Training model')
        if use_meta_model:
            trainer = MetaTrainer(model, device=device)
        else:
            trainer = BaseTrainer(model, device=device)
        trainer.fit(train_data, val_data, test_data, trainer_cfg)
        model.load_state_dict(trainer.get_best_model())
        # save best model
        save_model_dir = os.path.join(cfg.output_dir, '_'.join(model_tags) + '.pt')
        trainer.save_model(save_model_dir)


    """ Test """
    if evaluator_cfg is not None:
        global_logger.logger.exp('Testing')
        if evaluator_cfg.name == 'at':
            strategy = build_strategy(strategy_cfg.class_name, strategy_cfg)
            stopping_criterion = build_stopping_criterion(stopping_criterion_cfg.class_name, stopping_criterion_cfg)
            evaluator = AdaptiveTestingEvaluator(model, strategy, stopping_criterion, device=device)
        else:
            evaluator = CognitiveDiagnosisEvaluator(model, device=device)

        test_result = evaluator.evaluate(test_data, evaluator_cfg)

        # save test result
        with open(os.path.join(cfg.output_dir, f'test_result@{"_".join(exp_tags)}.txt'), 'w') as f:
            if evaluator_cfg.name == 'at':
                for i in sorted(test_result.keys()):
                    f.write(f'Test step {i}: acc={test_result[i]["acc"]:.4f}, auc={test_result[i]["auc"]:.4f}, '
                            f'rmse={test_result[i]["rmse"]:.4f}\n')
                    global_logger.logger.exp(f'Test step {i}: acc={test_result[i]["acc"]:.4f}, '
                                             f'auc={test_result[i]["auc"]:.4f}, rmse={test_result[i]["rmse"]:.4f}')
            else:
                f.write(f'acc={test_result["acc"]:.4f}, auc={test_result["auc"]:.4f}, rmse={test_result["rmse"]:.4f}\n')
                global_logger.logger.exp(f'acc={test_result["acc"]:.4f}, auc={test_result["auc"]:.4f}, rmse={test_result["rmse"]:.4f}')

        # save test log
        if 'log' in test_result:
            torch.save(test_result['log'], os.path.join(cfg.output_dir, f'test_log@{"_".join(exp_tags)}.pt'))

        # save cfg
        torch.save(cfg, os.path.join(cfg.output_dir, f'cfg@{"_".join(exp_tags)}.pt'))
        with open(os.path.join(cfg.output_dir, f'cfg@{"_".join(exp_tags)}.txt'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    # free the temporary directory
    if tmp_dir is not None:
        tmp_dir.cleanup()
