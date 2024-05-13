from abc import ABC, abstractmethod
from typing import Dict


class BaseStoppingCriterion(ABC):
    @abstractmethod
    def __init__(self, cfg):
        raise NotImplementedError

    @abstractmethod
    def stop(self, session: Dict) -> bool:
        raise NotImplementedError()
