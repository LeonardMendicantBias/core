# %%

from functools import cached_property
import h5py
import numpy as np


class Parent:
    
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @cached_property
    def total(self):
        return self.a + self.b + self.c
    
    def abc(self):
        return self.total
    
if __name__ == '__main__':
    kw = {
        'a': 1,
        'b': 2,
    }

    parent = Parent(**kw, c=3)
    print(parent.abc())

# %%
