from abc import ABC, abstractmethod
import typing

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def init_user_params(self, params=None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_user_params(self) -> typing.Dict:
        raise NotImplementedError()

    @abstractmethod
    def get_item_params(self) -> typing.Dict:
        raise NotImplementedError()

    @abstractmethod
    def update_user_params(self, session: typing.Dict) -> None:
        # default implementation
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(
            list(self.get_user_params().values()), lr=self.update_lr
        )

        if self.update_policy == "last":
            item_nos = session["selected"][-1:]
        else:
            item_nos = session["selected"]
        labels = [session["all_logs"][i] for i in item_nos]

        for i in range(self.update_max_loop):
            # Stochastic Gradient Descent
            for item_no, label in zip(item_nos, labels):
                item_no_t = torch.LongTensor([item_no]).squeeze().to(self.device)
                label_t = torch.FloatTensor([label]).squeeze().to(self.device)
                preds = self.forward(item_no_t)
                loss = loss_fn(preds, label_t)
                self.zero_grad()
                loss.backward()
                optimizer.step()

            """
            # Batch Gradient Descent
            item_nos_t = torch.tensor(item_nos).long().to(self.device)
            labels_t = torch.tensor(labels).float().to(self.device)
            preds = self.forward(item_nos_t)
            loss = loss_fn(preds, labels_t)
            self.zero_grad()
            loss.backward()
            optimizer.step()
            """

    def predict_user_performance(self, item_nos: torch.LongTensor) -> torch.Tensor:
        with torch.no_grad():
            prediction = self.forward(item_nos)
            return prediction


class MetaModel(BaseModel, ABC):
    @property
    def base_model(self):
        return self._base_model

    @base_model.setter
    def base_model(self, value):
        self._base_model = value

    @property
    def hyper_model(self):
        return self._hyper_model

    @hyper_model.setter
    def hyper_model(self, value):
        self._hyper_model = value

    @abstractmethod
    def adaptation(
        self,
        item_nos: torch.LongTensor,
        labels: torch.Tensor,
        adapted_hyper_model: typing.Dict[str, torch.Tensor],
    ) -> typing.Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def get_meta_params(self) -> typing.Dict:
        raise NotImplementedError()

    @abstractmethod
    def prediction(
        self,
        item_nos: torch.LongTensor,
        adapted_hyper_model: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def validation(
        self,
        item_nos: torch.LongTensor,
        labels: torch.Tensor,
        adapted_hyper_model: typing.Dict[str, torch.Tensor],
        return_prediction: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError()
