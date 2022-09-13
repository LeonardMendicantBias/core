# %%

from dataclasses import dataclass
from typing import List
import numpy as np
import inspect


@dataclass
class Parent:
    abc: str

    def __post_init__(self):
        print('parent post init')

    
@dataclass
class Child(Parent):
    de: str

    def __post_init__(self):
        print('child post init')


if __name__ == '__main__':
    child = Child(abc='abc', de='de')
    print(child)
    # inspect.getsource(child.__init__)

# %%
