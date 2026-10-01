import sys, torch
project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
sys.path.insert(0, project_root)
# Force fresh import by removing cached modules
for k in list(sys.modules.keys()):
    if 'Colorization' in k or 'colorize' in k or 'ddcolor' in k or 'unet' in k or 'convnext' in k:
        del sys.modules[k]

from Colorization.colorize import find_checkpoint
from Colorization.arch.ddcolor_arch import DDColor
from Colorization.arch.unet import NormType

ckpt_path = find_checkpoint('tiny')
raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
state = raw.get('params', raw) if isinstance(raw, dict) else raw
state = {k.replace('module.', ''): v for k, v in state.items()}

# Build with Spectral norm to match checkpoint
model = DDColor(encoder_name='convnext-t', num_input_channels=3, input_size=(512,512),
                nf=512, num_output_channels=2, last_norm='Spectral', do_normalize=False,
                num_queries=100, dec_layers=9, num_scales=3)

missing, unexpected = model.load_state_dict(state, strict=False)
print(f'Missing keys:    {len(missing)}')
print(f'Unexpected keys: {len(unexpected)}')
if missing:
    print(f'  Missing[:10]: {missing[:10]}')
if unexpected:
    print(f'  Unexpected[:10]: {unexpected[:10]}')
if not missing and not unexpected:
    print('PERFECT MATCH!')
