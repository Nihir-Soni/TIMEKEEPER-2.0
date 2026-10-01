"""
Debug script: trace every step of DDColor inference to find where colors go wrong.
Run with: .\venv\Scripts\python.exe debug_colorize_live.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

# ── Load a test image (use whatever is in output/stage_1_restore_output/restored_image) ──
import glob
candidates = glob.glob("output/stage_1_restore_output/restored_image/*.png") + \
             glob.glob("output/stage_1_restore_output/restored_image/*.jpg")
if not candidates:
    # Fall back to any image in the project
    candidates = glob.glob("test_images/*.png") + glob.glob("test_images/*.jpg")
if not candidates:
    candidates = glob.glob("imgs/*.png") + glob.glob("imgs/*.jpg")

if not candidates:
    print("No test images found. Place a PNG in output/stage_1_restore_output/restored_image/")
    sys.exit(1)

src = candidates[0]
print(f"\n=== Using test image: {src} ===\n")
img_bgr = cv2.imread(src)
print(f"Input shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
print(f"Input BGR value range: min={img_bgr.min()}, max={img_bgr.max()}\n")

# ── Step 1: BGR → LAB ──────────────────────────────────────────────────────────
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
l_full = lab[:, :, 0]
print(f"L channel: min={l_full.min():.2f}, max={l_full.max():.2f}  (expected 0-100)")

# ── Step 2: Build input tensor ─────────────────────────────────────────────────
INPUT_SIZE = 512
l_norm = l_full / 100.0
l_resize = cv2.resize(l_norm, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
inp = np.stack([l_resize] * 3, axis=0)[None]   # 1,3,H,W
tensor = torch.from_numpy(inp).float()
print(f"\nInput tensor: shape={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}")

# ── Step 3: Load model ─────────────────────────────────────────────────────────
from Colorization.colorize import DDColorizer, find_checkpoint
ckpt = find_checkpoint("tiny")
print(f"\nCheckpoint: {ckpt}")

device = torch.device("cpu")
colorizer = DDColorizer.from_checkpoint(checkpoint_path=ckpt, model_size="tiny",
                                         input_size=512, device=device)

# ── Step 4: Raw model forward pass ─────────────────────────────────────────────
colorizer.model.eval()
with torch.no_grad():
    raw_out = colorizer.model(tensor.to(device))   # 1×2×H'×W'

print(f"\nRaw model output: shape={raw_out.shape}")
print(f"  min={raw_out.min().item():.4f}, max={raw_out.max().item():.4f}")
print(f"  mean={raw_out.mean().item():.4f}, std={raw_out.std().item():.4f}")

# After tanh
tanh_out = torch.tanh(raw_out)
print(f"\nAfter tanh: min={tanh_out.min().item():.4f}, max={tanh_out.max().item():.4f}")

# ── Step 5: Scale to LAB ab range ──────────────────────────────────────────────
ab = tanh_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
ab_scaled = ab * 128.0
print(f"\nab (scaled): min={ab_scaled.min():.2f}, max={ab_scaled.max():.2f}  (expected -128 to 128)")

# ── Step 6: Resize ab back ─────────────────────────────────────────────────────
orig_h, orig_w = img_bgr.shape[:2]
ab_resize = cv2.resize(ab_scaled, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

# ── Step 7: Assemble LAB image ─────────────────────────────────────────────────
lab_out = np.stack([l_full, ab_resize[:, :, 0], ab_resize[:, :, 1]], axis=-1)
lab_out[:, :, 0] = np.clip(lab_out[:, :, 0], 0, 100)
lab_out[:, :, 1] = np.clip(lab_out[:, :, 1], -128, 127)
lab_out[:, :, 2] = np.clip(lab_out[:, :, 2], -128, 127)
print(f"\nLAB assembled: L=[{lab_out[:,:,0].min():.1f},{lab_out[:,:,0].max():.1f}],"
      f" a=[{lab_out[:,:,1].min():.1f},{lab_out[:,:,1].max():.1f}],"
      f" b=[{lab_out[:,:,2].min():.1f},{lab_out[:,:,2].max():.1f}]")

# ── Step 8: LAB → BGR ─────────────────────────────────────────────────────────
bgr_float = cv2.cvtColor(lab_out.astype(np.float32), cv2.COLOR_Lab2BGR)
print(f"\nAfter Lab2BGR (float): min={bgr_float.min():.4f}, max={bgr_float.max():.4f}")
bgr_out = np.clip(bgr_float * 255.0, 0, 255).astype(np.uint8)
print(f"After *255 + uint8: min={bgr_out.min()}, max={bgr_out.max()}")

# ── Save debug outputs ─────────────────────────────────────────────────────────
cv2.imwrite("debug_colorized_output.png", bgr_out)
print(f"\nSaved: debug_colorized_output.png")

# Also save gray input for comparison
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.imwrite("debug_gray_input.png", gray_bgr)
cv2.imwrite("debug_original_input.png", img_bgr)
print("Saved: debug_gray_input.png, debug_original_input.png")

# Also try: what if we use COLOR_Lab2RGB and then convert?
bgr_via_rgb = cv2.cvtColor(lab_out.astype(np.float32), cv2.COLOR_Lab2RGB)
bgr_via_rgb_uint8 = np.clip(bgr_via_rgb * 255.0, 0, 255).astype(np.uint8)
# convert RGB→BGR for OpenCV save
bgr_via_rgb_save = cv2.cvtColor(bgr_via_rgb_uint8, cv2.COLOR_RGB2BGR)
cv2.imwrite("debug_colorized_via_rgb.png", bgr_via_rgb_save)
print("Saved: debug_colorized_via_rgb.png (via Lab2RGB path)")

print("\n=== Done. Check the debug_*.png files ===")
