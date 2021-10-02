
# Cosine Annealing with Warmup for PyTorch

Generally, during semantic segmentation with a pretrained backbone, the backbone and the decoder have different learning rates. Encoder usually employs 10x lower learning rate when compare to decoder. To adapt to this condition, this repository provides a cosine annealing with warmup scheduler adapted from  [katsura-jp](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup) . The original repo overwrites this condition and sets the same learning rate.

## Arguments

- optimizer (Optimizer): Wrapped optimizer.
- first_cycle_steps (int): First cycle step size.
- cycle_mult(float): Cycle steps magnification. Default: 1.
- max_lr (float or List): First cycle's max learning rate. Default: 0.1.
- min_lr (float or List): Min learning rate. Default: 0.001.
- warmup_steps(int): Linear warmup step size. Default: 0.
- gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
- last_epoch (int): The index of last epoch. Default: -1.

## Example
Run `test.py` to see the visual learning rate plot. 
```
>> from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
>>
>> model = ...
>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
>> lrs = [l['lr'] for l in optimizer.param_groups]
>> scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=200,
                                          cycle_mult=1.0,
                                          max_lr=lrs,
                                          min_lr=[1e-5, 0.007],
                                          warmup_steps=50,
                                          gamma=1.0)
>> for epoch in range(n_epoch):
>>     train()
>>     valid()
>>     scheduler.step()
```
