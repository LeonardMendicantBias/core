# %%

import network_library
import models
from models import NLPSimulation, ADDTask


if __name__ == '__main__':
    simulation = NLPSimulation.from_task_network_cls(
        task=ADDTask.from_config(
            100, (2, 3),
            [50, 50],
            [(3, 4), (4, 5)]
        ),
        network_cls=network_library.Transformer,
        network_name="Transformer"
    )
    simulation.start()
    del simulation
    
    # task=ADDTask.from_config(
    #     100, (2, 3),
    #     [50, 50],
    #     [(3, 4), (4, 5)]
    # )

    # for sample in task.train_ds:
    #     print(sample)
    #     break
    # del task


# %%
