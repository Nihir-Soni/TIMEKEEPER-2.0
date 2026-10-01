import sys, torch
import torch.nn as nn
project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
sys.path.insert(0, project_root)

from Colorization.arch.unet import UnetBlockWide, NormType, CustomPixelShuffle_ICNR

blk = UnetBlockWide(10, 10, 10, norm_type=NormType.Spectral)
print("UnetBlockWide keys:")
for k, v in blk.state_dict().items():
    print(k)

print("\nCustomPixelShuffle_ICNR keys:")
shuf = CustomPixelShuffle_ICNR(10, 10, norm_type=NormType.Spectral, use_bn=False)
for k, v in shuf.state_dict().items():
    print(k)
