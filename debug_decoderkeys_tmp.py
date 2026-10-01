import sys, torch
project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
sys.path.insert(0, project_root)

from Colorization.colorize import find_checkpoint
ckpt_path = find_checkpoint('tiny')
raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
state = raw.get('params', raw) if isinstance(raw, dict) else raw
state = {k.replace('module.', ''): v for k, v in state.items()}

# Print only decoder keys with shapes
for k, v in sorted(state.items()):
    if k.startswith('decoder'):
        print(f'{k}: {tuple(v.shape)}')
