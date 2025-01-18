"""gcn.py
"""

from pydantic import BaseModel

from ..base import BaseNetwork


class GCN(BaseNetwork):
    class Params(BaseModel):
        pass
