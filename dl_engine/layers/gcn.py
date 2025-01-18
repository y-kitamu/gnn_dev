"""gcn.py

Author : Yusuke Kitamura
Create Date : 2024-02-28 22:15:26
"""

from pydantic import BaseModel

from .base import BaseNetwork


class GCN(BaseNetwork):
    class Params(BaseModel):
        pass
