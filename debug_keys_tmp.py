import sys, torch
project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
sys.path.insert(0, project_root)
for mod in list(sys.modules.keys()):
    if 'Colorization' in mod:
        del sys.modules[mod]

from Colorization.colorize import find_checkpoint
ckpt_path = find_checkpoint('tiny')
raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
state = raw.get('params', raw) if isinstance(raw, dict) else raw
state = {k.replace('module.', ''): v for k, v in state.items()}

from Colorization.arch.ddcolor_arch import DDColor
model = DDColor(encoder_name='convnext-t', num_input_channels=3, input_size=(512,512),
                nf=512, num_output_channels=2, last_norm='Weight', do_normalize=False,
                num_queries=100, dec_layers=9, num_scales=3)

missing, unexpected = model.load_state_dict(state, strict=False)

print('=== MISSING (model expects, ckpt lacks) - first 20:')
for k in missing[:20]:
    print(f'  {k}')

print()
print('=== UNEXPECTED (ckpt has, model lacks) - first 20:')
for k in unexpected[:20]:
    print(f'  {k}')

# Group by prefix
def prefixes(keys, depth=3):
    from collections import Counter
    c = Counter()
    for k in keys:
        parts = k.split('.')
        c['.'.join(parts[:depth])] += 1
    return c.most_common(15)

print()
print('=== MISSING grouped by prefix:')
for p, n in prefixes(missing):
    print(f'  {p}: {n}')

print()
print('=== UNEXPECTED grouped by prefix:')
for p, n in prefixes(unexpected):
    print(f'  {p}: {n}')
