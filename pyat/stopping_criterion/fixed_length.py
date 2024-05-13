from typing import Dict

from .base import BaseStoppingCriterion
from .build import STOPPING_CRITERION_REGISTRY


@STOPPING_CRITERION_REGISTRY.register()
class FixedLengthStoppingCriterion(BaseStoppingCriterion):
    def __init__(self, cfg):
        self.length = cfg["length"]

    def stop(self, session: Dict) -> bool:
        return len(session["selected"]) >= self.length
