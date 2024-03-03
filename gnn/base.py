"""base.py

Author : Yusuke Kitamura
Create Date : 2024-02-28 22:27:18
Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
"""

import keras
from pydantic import BaseModel


class MetrixMixIn:
    def update_metrics(self) -> None:
        pass

    def get_metrics(self) -> dict[str, float]:
        return dict()

    def reset_metrics(self) -> None:
        pass


class BaseParams(BaseModel):
    name: str = ""
    params: dict = {}


def get_object(params: BaseParams, object_list: dict):
    object_class = object_list[params.name]
    return object_class(object_class.Params(**params.params))


def get_object_default_params(name, object_list: dict) -> BaseModel:
    object_class = object_list[name]
    return object_class.Params()
