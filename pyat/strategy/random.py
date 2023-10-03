import random
from typing import Dict


from .base import BaseStrategy
from .build import STRATEGY_REGISTRY


@STRATEGY_REGISTRY.register()
class RandomStrategy(BaseStrategy):

    def __init__(self, cfg: Dict):
        super().__init__()
        self.rand = random.Random(cfg.get('seed', None))

    def select_item(self, session: Dict) -> int:
        unselected = session['unselected']
        selection = self.rand.choice(unselected)
        return selection
