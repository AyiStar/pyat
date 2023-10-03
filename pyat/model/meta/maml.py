import typing

import torch
import torch.nn as nn
from torch.nn.utils.stateless import functional_call

from ..base_classes import BaseModel, MetaModel
from ..hyper import IdentityNet
from ..build import META_MODEL_REGISTRY
from .._utils import zero_grad, clone_state_dict

StateDictType = typing.Dict[str, torch.Tensor]

@META_MODEL_REGISTRY.register()
class ModelAgnosticMetaLearning(MetaModel):

    def __init__(self, base_model: BaseModel, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.base_model = base_model
        self.hyper_model = IdentityNet(base_model)
        self._adaptive_hyper_model: typing.Optional[StateDictType] = None  # used for adaptive testing

        self.num_inner_updates = self.cfg['inner_num_updates']
        self.inner_lr = self.cfg['inner_lr']
        self.update_policy = self.cfg['update_policy']
        self.device = self.cfg['device']

    def adaptation(self, item_nos: torch.LongTensor, labels: torch.Tensor,
                   adapted_hyper_model: StateDictType
                   ) -> StateDictType:
        for inner_epoch in range(self.num_inner_updates):
            fast_weights = functional_call(self.hyper_model, adapted_hyper_model, args=())
            preds = functional_call(self.base_model, fast_weights, args=(item_nos,))
            loss = nn.BCELoss()(preds, labels)
            zero_grad(adapted_hyper_model)
            grads = torch.autograd.grad(loss, list(adapted_hyper_model.values()))
            for (key, val), grad in zip(adapted_hyper_model.items(), grads):
                adapted_hyper_model[key] = val - self.inner_lr * grad
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
        preds = self.prediction(item_nos=item_nos, adapted_hyper_model=adapted_hyper_model)
        loss = nn.BCELoss()(preds, labels)
        return (loss, preds) if return_prediction else loss

    def init_user_params(self, params=None) -> None:
        self._adaptive_hyper_model = clone_state_dict(self.hyper_model)

    def get_user_params(self) -> StateDictType:
        with torch.no_grad():
            return functional_call(self.hyper_model,
                                   self._adaptive_hyper_model if self._adaptive_hyper_model is not None else {},
                                   args=())

    def get_item_params(self) -> typing.Dict:
        return self.base_model.get_item_params()

    def update_user_params(self, session: typing.Dict, **kwargs) -> None:
        if self.update_policy == 'last':
            item_nos = session['selected'][-1:]
        else:
            item_nos = session['selected']
        labels = torch.tensor([session['all_logs'][i] for i in item_nos]).float().to(self.device)
        item_nos = torch.tensor(item_nos).long().to(self.device)
        self._adaptive_hyper_model = self.adaptation(item_nos, labels, adapted_hyper_model=self._adaptive_hyper_model)

    def forward(self, item_nos: torch.LongTensor):
        base_model_params = functional_call(self.hyper_model, self._adaptive_hyper_model, args=())
        return functional_call(self.base_model, base_model_params, args=(item_nos,))
