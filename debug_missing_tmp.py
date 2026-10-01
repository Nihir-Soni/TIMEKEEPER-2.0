import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(1, 1)

model = MyModel()
state = {'b.weight': torch.randn(1,1)}
missing, unexpected = model.load_state_dict(state, strict=False)
print('Missing:', missing)
print('Unexpected:', unexpected)
