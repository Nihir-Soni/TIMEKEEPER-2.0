import sys, torch
project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
sys.path.insert(0, project_root)

from Colorization.colorize import find_checkpoint
ckpt_path = find_checkpoint('tiny')
raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
state = raw.get('params', raw) if isinstance(raw, dict) else raw
state = {k.replace('module.', ''): v for k, v in state.items()}

# Print all unique key patterns with shapes
for k, v in sorted(state.items()):
    print(f'{k}: {tuple(v.shape)}')
