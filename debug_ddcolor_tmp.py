import os, sys, logging, numpy as np, torch, cv2

project_root = r'c:\Bringing-Old-Photos-Back-to-Life-master'
sys.path.insert(0, project_root)
os.chdir(project_root)

logging.basicConfig(level=logging.DEBUG, format='%(name)s %(levelname)s: %(message)s')

from Colorization.colorize import find_checkpoint, _load_state_dict, _preprocess_for_ddcolor, _postprocess_ddcolor

ckpt_path = find_checkpoint('tiny')
print(f'Checkpoint: {ckpt_path}')

raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
if isinstance(raw, dict):
    print(f'Top-level keys: {list(raw.keys())}')
    state = raw.get('params', raw.get('state_dict', raw.get('model', raw)))
else:
    state = raw
state = {k.replace('module.', ''): v for k, v in state.items()}
print(f'Weight tensors: {len(state)}')
print(f'First 5 keys: {list(state.keys())[:5]}')
print(f'Last  5 keys: {list(state.keys())[-5:]}')

from Colorization.arch.ddcolor_arch import DDColor
model = DDColor(encoder_name='convnext-t', num_input_channels=3, input_size=(512,512),
                nf=512, num_output_channels=2, last_norm='Weight', do_normalize=False,
                num_queries=100, dec_layers=9, num_scales=3)

missing, unexpected = model.load_state_dict(state, strict=False)
print(f'Missing keys ({len(missing)}): {missing[:10]}')
print(f'Unexpected keys ({len(unexpected)}): {unexpected[:10]}')

model_params = dict(model.named_parameters())
model_buffers = dict(model.named_buffers())
all_model = {**model_params, **model_buffers}
mismatches = [(k, tuple(all_model[k].shape), tuple(v.shape))
              for k,v in state.items() if k in all_model and tuple(all_model[k].shape)!=tuple(v.shape)]
print(f'Shape mismatches ({len(mismatches)}): {mismatches[:5]}')

model.eval()
device = torch.device('cpu')
model.to(device)

restored_dir = os.path.join(project_root, 'output', 'stage_1_restore_output', 'restored_image')
debug_out = os.path.join(project_root, 'output', 'debug_ddcolor_intermediates')
os.makedirs(debug_out, exist_ok=True)

for fname in sorted(os.listdir(restored_dir)):
    if not fname.lower().endswith(('.png','.jpg','.jpeg')):
        continue
    src = os.path.join(restored_dir, fname)
    bgr = cv2.imread(src)
    if bgr is None:
        print(f'SKIP {fname}: cannot read')
        continue
    print(f'\n--- {fname} ---')
    print(f'  Input  shape={bgr.shape} dtype={bgr.dtype} min={bgr.min()} max={bgr.max()} mean={bgr.mean():.2f}')
    tensor, l_full, orig_hw = _preprocess_for_ddcolor(bgr, 512)
    tensor = tensor.to(device)
    print(f'  Tensor shape={tuple(tensor.shape)} min={tensor.min():.4f} max={tensor.max():.4f} mean={tensor.mean():.4f}')
    with torch.no_grad():
        ab_pred = model(tensor)
    print(f'  ab_pred shape={tuple(ab_pred.shape)} min={ab_pred.min():.4f} max={ab_pred.max():.4f} mean={ab_pred.mean():.4f}')
    if ab_pred.abs().max().item() < 1e-4:
        print('  WARNING: ab_pred near-zero -> output will be gray!')
    colorized = _postprocess_ddcolor(ab_pred, l_full, orig_hw)
    print(f'  Output shape={colorized.shape} dtype={colorized.dtype} min={colorized.min()} max={colorized.max()} mean={colorized.mean():.2f}')
    b,g,r = colorized[:,:,0].mean(), colorized[:,:,1].mean(), colorized[:,:,2].mean()
    print(f'  Channel means B={b:.2f} G={g:.2f} R={r:.2f}')
    if abs(float(b)-float(g))<0.5 and abs(float(g)-float(r))<0.5:
        print('  WARNING: channels identical -> effectively grayscale!')
    dst = os.path.join(debug_out, os.path.splitext(fname)[0]+'_ddcolor.png')
    cv2.imwrite(dst, colorized)
    print(f'  Saved: {dst}')
print('\nDone. Check output/debug_ddcolor_intermediates/')
