import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

import pyat.utils.global_logger as glogger
from ..base_classes import BaseModel, MetaModel
from ..hyper import NormalVariationalNet, IdentityNet, StandardNormalNet
from ..build import META_MODEL_REGISTRY
from .._utils import zero_grad, clone_state_dict, kl_divergence_gaussians

StateDictType = typing.Dict[str, torch.Tensor]


@META_MODEL_REGISTRY.register()
class AmortizedBayesianMetaLearning(MetaModel):

    def __init__(self, base_model: BaseModel, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.base_model = base_model

        self.hyper_model = NormalVariationalNet(base_model)
        # for ablation study
        self.no_bayes = cfg.get('no_bayes', False)
        self.fix_prior = cfg.get('fix_prior', False)
        assert not (self.no_bayes and self.fix_prior), 'Cannot remove Bayes and fix prior at the same time'
        if self.no_bayes:
            self.hyper_model = IdentityNet(base_model)
        if self.fix_prior:
            self.hyper_model = StandardNormalNet(base_model)

        self.adaptive_hyper_model: typing.Optional[StateDictType] = None  # used for adaptive testing

        self.inner_num_updates = self.cfg['inner_num_updates']
        self.inner_sgd = self.cfg['inner_sgd']
        self.inner_learn_lr = self.cfg['inner_learn_lr']
        self.num_samples = self.cfg['num_samples']
        self.kl_weight = self.cfg['kl_weight']
        self.update_policy = self.cfg['update_policy']
        self.device = self.cfg['device']

        if self.inner_learn_lr:
            self.inner_lrs = nn.Parameter(torch.Tensor([self.cfg['inner_lr']] * self.inner_num_updates).to(self.device),
                                          requires_grad=True)
        else:
            self.inner_lrs = torch.Tensor([self.cfg['inner_lr']] * self.inner_num_updates).to(self.device)

    def adaptation(self, item_nos: torch.LongTensor, labels: torch.Tensor,
                   adapted_hyper_model: StateDictType) -> StateDictType:

        def inner_update(item_nos_input, labels_input):
            q_params = list(adapted_hyper_model.values())
            grads_accum = [0] * len(adapted_hyper_model)
            for _ in range(self.num_samples):
                fast_weights = functional_call(self.hyper_model, adapted_hyper_model, args=())
                preds = functional_call(self.base_model, fast_weights, args=(item_nos_input,))
                bce_loss = nn.BCELoss()(preds, labels_input)
                kl_loss = kl_divergence_gaussians(p=q_params, q=p_params)
                loss = bce_loss + self.kl_weight * kl_loss
                zero_grad(adapted_hyper_model)
                grads = torch.autograd.grad(loss, q_params, retain_graph=True)
                for i, g in enumerate(grads):
                    grads_accum[i] = grads_accum[i] + (g / self.num_samples)
            for (key, val), grad in zip(adapted_hyper_model.items(), grads_accum):
                adapted_hyper_model[key] = val - self.inner_lrs[inner_epoch] * grad

        p_params = [p.clone() for p in adapted_hyper_model.values()]
        for inner_epoch in range(self.inner_num_updates):
            if self.inner_sgd:
                for i in range(item_nos.size(0)):
                    inner_update(item_nos[i], labels[i])
            else:
                inner_update(item_nos, labels)

        return adapted_hyper_model

    def prediction(self, item_nos: torch.LongTensor,
                   adapted_hyper_model: StateDictType
                   ) -> torch.Tensor:
        # generate task-specific parameter
        base_model_params = functional_call(self.hyper_model, adapted_hyper_model, args=())
        logits = functional_call(self.base_model, base_model_params, args=(item_nos,))
        return logits

    def validation(self, item_nos: torch.LongTensor, labels: torch.Tensor,
                   adapted_hyper_model: StateDictType, return_prediction: bool = False
                   ) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
        sampled_preds = [self.prediction(item_nos=item_nos, adapted_hyper_model=adapted_hyper_model)
                         for _ in range(self.num_samples)]
        loss = sum([nn.BCELoss()(preds, labels) for preds in sampled_preds]) / self.num_samples
        if return_prediction:
            preds = sum(sampled_preds) / self.num_samples
            return loss, preds
        else:
            return loss

    def init_user_params(self, params=None) -> None:
        self.adaptive_hyper_model = clone_state_dict(self.hyper_model)

    def get_user_params(self) -> StateDictType:
        with torch.no_grad():
            return functional_call(self.hyper_model,
                                   self.adaptive_hyper_model if self.adaptive_hyper_model is not None else {},
                                   args=())

    def get_item_params(self) -> typing.Dict:
        return self.base_model.get_item_params()

    def get_meta_params(self) -> typing.Dict:
        if self.fix_prior:
            return {}
        else:
            return {k: v for k, v in self.named_parameters() if
                    k.startswith('hyper_model')}

    def update_user_params(self, session: typing.Dict, **kwargs) -> None:
        item_nos = session['selected'][-1:] if self.update_policy == 'last' else session['selected']
        labels = torch.tensor([session['all_logs'][i] for i in item_nos]).float().to(self.device)
        item_nos = torch.tensor(item_nos).long().to(self.device)
        self.adaptive_hyper_model = self.adaptation(item_nos, labels, adapted_hyper_model=self.adaptive_hyper_model)

    def forward(self, item_nos: torch.LongTensor):
        return sum([self.prediction(item_nos, adapted_hyper_model=self.adaptive_hyper_model)
                    for _ in range(self.num_samples)]) / self.num_samples
