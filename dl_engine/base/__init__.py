"""__init__.py
Abstract classes.
Do not import any concrete classes here.
"""


from typing import Any

from pydantic import BaseModel

from .base_loss import BaseLoss
from .base_layer import BaseNetwork
from .base_trainer import BaseTrainer
from .base_dataloader import BaseDataloader
from .base_callback import BaseCallback


class BaseParams(BaseModel):
    name: str = ""
    params: dict[str, Any] = {}


def get_object(params: BaseParams, object_list: dict[str, Any]):
    object_class = object_list[params.name]
    return object_class(object_class.Params(**params.params))


def get_default_params_of(name: str, object_list: dict[str, Any]) -> BaseModel:
    object_class = object_list[name]
    return object_class.Params()

