"""base.py

Author : Yusuke Kitamura
Create Date : 2024-03-02 16:53:40
"""

from typing import Any

from pydantic import BaseModel


class BaseDataloader:
    class Params(BaseModel):
        batch_size: int = 1

    def __init__(self, params: Params):
        self.kparams = params

    @property
    def steps_per_epoch(self):
        raise NotImplementedError

    def get_next(self) -> dict[str, Any]:
        raise NotImplementedError
