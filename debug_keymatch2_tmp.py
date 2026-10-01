import sys, torch
project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
sys.path.insert(0, project_root)
for mod in list(sys.modules.keys()):
    if 'Colorization' in mod:
        del sys.modules[mod]

from Colorization.colorize import find_checkpoint, _preprocess_for_ddcolor, _postprocess_ddcolor
from Colorization.arch.ddcolor_arch import DDColor
import cv2, numpy as np

ckpt_path = find_checkpoint('tiny')
raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
state = raw.get('params', raw) if isinstance(raw, dict) else raw
state = {k.replace('module.', ''): v for k, v in state.items()}

model = DDColor(encoder_name='convnext-t', num_input_channels=3, input_size=(512,512),
                nf=512, num_output_channels=2, last_norm='Weight', do_normalize=False,
                num_queries=100, dec_layers=9, num_scales=3)

missing, unexpected = model.load_state_dict(state, strict=False)
print(f'Missing keys:    {len(missing)}')
print(f'Unexpected keys: {len(unexpected)}')
if missing:
    print(f'  First missing: {missing[:5]}')
if unexpected:
    print(f'  First unexpected: {unexpected[:5]}')
if not missing and not unexpected:
    print('PERFECT MATCH: All 442 keys loaded correctly!')
