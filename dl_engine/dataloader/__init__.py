"""dataloader.py
"""

from pydantic import BaseModel

from ..base import BaseParams, get_default_params_of, BaseDataloader
from .sandbox import MnistDataloader


class DataloaderParams(BaseParams):
    pass


dataloader_list: dict[str, type[BaseDataloader]] = {"mnist": MnistDataloader}


def get_dataloader(params: DataloaderParams, is_train: bool) -> BaseDataloader:
    """ """
    dataloader_class = dataloader_list[params.name]
    return dataloader_class(dataloader_class.Params(**params.params), is_train)


def get_default_dataloader_params(name: str) -> BaseModel:
    """ """
    return get_default_params_of(name, dataloader_list)
