import torch
from torch.nn.modules.module import _IncompatibleKeys
missing = ["a"]
unexpected = ["b"]
print(_IncompatibleKeys(missing, unexpected))
