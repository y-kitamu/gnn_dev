"""base.py
"""

from typing import Any

from pydantic import BaseModel


class BaseDataloader:
    class Params(BaseModel):
        batch_size: int = 1

    def __init__(self, params: Params):
        self.params = params

    @property
    def steps_per_epoch(self) -> int:
        raise NotImplementedError

    @property
    def output_keys(self) -> list[str]:
        raise NotImplementedError

    @property
    def ouput_shape(self) -> list[int]:
        raise NotImplementedError

    def get_next(self) -> dict[str, Any]:
        raise NotImplementedError
