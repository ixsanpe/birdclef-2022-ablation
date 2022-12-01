from modules import SelectSplitData, RejoinSplitData 
import torch 
import numpy as np

bs = 32
N = 64
split = 4
dim = 4
input = [[np.arange(N+i) for i in range(dim)]]*bs
input = torch.tensor(input)
input_dict = {'x': input, 'lens': torch.tensor([N]*bs)}
print(f'{input.shape=}')

ssd = SelectSplitData(duration=N, sr=1, n_splits=split, offset=0.)
temp = ssd(input_dict)
print(f'{temp["x"].shape=}')

rsd = RejoinSplitData(duration=N//split, sr=1, n_splits=split, offset=0.)
rejoined = rsd(temp["x"])
print(f'{rejoined.shape=}')
print(rejoined)