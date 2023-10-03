from pyat.utils.registry import Registry
from .base_classes import BaseModel

BASE_MODEL_REGISTRY = Registry("BASE_MODEL")  # noqa F401 isort:skip
BASE_MODEL_REGISTRY.__doc__ = """
Registry for base models.
The registered object will be called with `obj(cfg)`
and expected to return a `BaseModel` object.
"""


META_MODEL_REGISTRY = Registry("META_MODEL")  # noqa F401 isort:skip
META_MODEL_REGISTRY.__doc__ = """
Registry for meta models.
The registered object will be called with `obj(cfg)`
and expected to return a `MetaModel` object.
"""


def build_base_model(name, cfg):
    model = BASE_MODEL_REGISTRY.get(name)(cfg)
    return model


def build_meta_model(name, base_model: BaseModel, cfg):
    model = META_MODEL_REGISTRY.get(name)(base_model, cfg)
    return model
