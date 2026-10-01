import torch, torch.nn as nn
from Colorization.arch.ddcolor_arch import DDColor
model = DDColor(encoder_name='convnext-t')
for k, v in model.state_dict().items():
    if 'refine_net' in k:
        print(k)
