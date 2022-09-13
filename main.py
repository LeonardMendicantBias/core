# %%

import models


if __name__ == '__main__':
    task = models.NLPTask.from_config(
        100, (2, 3),
        [50, 50],
        [(3, 4), (4, 5)]
    )

# %%
