# Verify the exact UnetBlockWide n_out values from checkpoint shapes
import torch
project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
import sys; sys.path.insert(0, project_root)
from Colorization.colorize import find_checkpoint
ckpt_path = find_checkpoint('tiny')
raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
state = raw.get('params', raw) if isinstance(raw, dict) else raw
state = {k.replace('module.', ''): v for k, v in state.items()}

for i in range(3):
    shuf_w = f'decoder.layers.{i}.shuf.conv.0.weight_orig'
    conv_w = f'decoder.layers.{i}.conv.0.weight_orig'
    bn_w   = f'decoder.layers.{i}.bn.weight'
    if shuf_w in state:
        shuf_shape = tuple(state[shuf_w].shape)
        conv_shape = tuple(state[conv_w].shape)
        bn_ch = tuple(state[bn_w].shape)
        shuf_out = shuf_shape[0] // 4  # after pixel_shuffle scale=2
        print(f'layers[{i}]: shuf({shuf_shape[1]}->{shuf_out}) + skip_bn({bn_ch[0]}) -> conv({conv_shape[1]}->{conv_shape[0]})')
        # n_out = shuf_out*2 (since UnetBlockWide: up_out=n_out//2, x_out=n_out//2)
        print(f'  -> UnetBlockWide(up_in={shuf_shape[1]}, x_in={bn_ch[0]}, n_out={shuf_out*2})')

print()
# last_shuf
ls = 'decoder.last_shuf.conv.0.weight_orig'
if ls in state:
    s = tuple(state[ls].shape)
    print(f'last_shuf: ({s[1]}->{s[0]//16}) with scale=4 (pixel_shuffle 4)')

print()
# color_decoder input_proj  
for i in range(3):
    k = f'decoder.color_decoder.input_proj.{i}.weight'
    if k in state:
        print(f'input_proj[{i}]: {tuple(state[k].shape)}')
