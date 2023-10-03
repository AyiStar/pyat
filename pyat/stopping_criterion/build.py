from pyat.utils.registry import Registry
from .base import BaseStoppingCriterion

STOPPING_CRITERION_REGISTRY = Registry("STOPPING_CRITERION")  # noqa F401 isort:skip
STOPPING_CRITERION_REGISTRY.__doc__ = """
Registry for stopping criterions.
The registered object will be called with `obj(cfg)`
and expected to return a `BaseStoppingCriterion` object.
"""


def build_stopping_criterion(name: str, cfg) -> BaseStoppingCriterion:
    criterion = STOPPING_CRITERION_REGISTRY.get(name)(cfg)
    return criterion
