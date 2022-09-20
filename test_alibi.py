# %%

import math
import torch


def get_slopes(h):
    start = (2**(-2**-(math.log2(h)-3)))
    ratio = start
    return [start*ratio**i for i in range(h)]


def fill_with_neg_inf(t):
    return t.float().fill_(float("-inf")).type_as(t)

if __name__ == '__main__':
    # triu = torch.triu(fill_with_neg_inf(torch.zeros([5, 5])), 1)

    # print(triu)

    print(
        get_slopes(8)
    )
