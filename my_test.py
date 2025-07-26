import numpy as np
import torch

all_time_actions = torch.zeros([600, 630, 6])
all_actions = all_time_actions[[1], 1:31]
print(all_actions.shape)
