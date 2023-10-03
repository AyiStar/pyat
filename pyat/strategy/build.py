from pyat.utils.registry import Registry
from .base import BaseStrategy

STRATEGY_REGISTRY = Registry("STRATEGY")
STRATEGY_REGISTRY.__doc__ = """
Registry for selection strategies.
The registered object will be called with `obj(cfg)`
and expected to return a `BaseStrategy` object.
"""


def build_strategy(name, cfg) -> BaseStrategy:
    strategy = STRATEGY_REGISTRY.get(name)(cfg)
    return strategy
