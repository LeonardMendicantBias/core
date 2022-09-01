# %%

import h5py
import numpy as np
from collections import Counter
from dataclasses import MISSING, dataclass, InitVar, field
from typing import Dict

@dataclass
class Data:
    a: int = 1
    b: int = 2

    # def __post_init__(self):
    def total(self):
        return self.a+self.b


if __name__ == '__main__':
    data = Data()
    print(data.total)


# %%
