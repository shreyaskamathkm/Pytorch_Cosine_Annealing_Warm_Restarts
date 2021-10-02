import torch.optim as optim
from src.cosineScheduler import CosineAnnealingWarmupRestarts
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == '__main__':

    # Initialization
    epochs = 25
    steps_per_epoch = 3029

    # Models
    model = torch.nn.Conv2d(16, 16, 3, 1, 1)
    model1 = torch.nn.Conv2d(16, 16, 3, 1, 1)

    # Model parameters
    params = list()
    params.append({'params': model.parameters(), 'lr': 0.01 / 10})
    params.append({'params': model1.parameters(), 'lr': 0.05})

    # Optimizer
    optimizer = torch.optim.Adam(params)

    lrs = [l['lr'] for l in optimizer.param_groups]
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=steps_per_epoch * 8,
                                              cycle_mult=1.05,
                                              max_lr=lrs,
                                              min_lr=[1e-5, 0.007],
                                              warmup_steps=0,
                                              gamma=0.5)

    lrs1 = []
    lrs2 = []
    for e in range(0, epochs):
        x = 0
        for _ in range(0, steps_per_epoch):
            scheduler.step()
            lrs1.append(optimizer.param_groups[0]['lr'])
            lrs2.append(optimizer.param_groups[1]['lr'])
            print(x, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
            x += 1

    plt.figure()
    plt.plot(np.arange(len(lrs1)), lrs1)
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(lrs2)), lrs2)
    plt.show()
