import os, sys, numpy as np, torch, cv2
project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
sys.path.insert(0, project_root)
os.chdir(project_root)

# Reload modules fresh (clear any cached imports)
for mod in list(sys.modules.keys()):
    if 'Colorization' in mod:
        del sys.modules[mod]

from Colorization.colorize import find_checkpoint, _load_state_dict, _preprocess_for_ddcolor, _postprocess_ddcolor
from Colorization.arch.ddcolor_arch import DDColor

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
if missing or unexpected:
    print(f'  First missing:    {missing[:3]}')
    print(f'  First unexpected: {unexpected[:3]}')
else:
    print('All keys matched perfectly!')

model.eval()
restored_dir = os.path.join(project_root, 'output', 'stage_1_restore_output', 'restored_image')
debug_out = os.path.join(project_root, 'output', 'debug_ddcolor_fixed')
os.makedirs(debug_out, exist_ok=True)

for fname in sorted(os.listdir(restored_dir)):
    if not fname.lower().endswith(('.png','.jpg','.jpeg')):
        continue
    bgr = cv2.imread(os.path.join(restored_dir, fname))
    if bgr is None:
        continue
    tensor, l_full, orig_hw = _preprocess_for_ddcolor(bgr, 512)
    with torch.no_grad():
        ab_pred = model(tensor)
    colorized = _postprocess_ddcolor(ab_pred, l_full, orig_hw)
    b,g,r = colorized[:,:,0].mean(), colorized[:,:,1].mean(), colorized[:,:,2].mean()
    print(f'{fname}: min={colorized.min()} max={colorized.max()} mean={colorized.mean():.1f}  B={b:.1f} G={g:.1f} R={r:.1f}')
    dst = os.path.join(debug_out, os.path.splitext(fname)[0]+'_ddcolor_fixed.png')
    cv2.imwrite(dst, colorized)
    print(f'  -> {dst}')
print('Done.')
