import torch
import cv2
import numpy as np
from pathlib import Path

from .model import UncertaintyUNet
from .config import UncertaintyConfig

class UncertaintyInferencer:
    def __init__(self, checkpoint_path, config_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = UncertaintyConfig(config_path)
        
        self.model = UncertaintyUNet(
            in_channels=self.config.get('model', 'in_channels'),
            out_channels=self.config.get('model', 'out_channels'),
            base_filters=self.config.get('model', 'base_filters')
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.image_size = self.config.get('dataset', 'image_size')

    def infer(self, damaged_bgr, restored_bgr):
        orig_h, orig_w = damaged_bgr.shape[:2]
        
        # Preprocess
        d_resized = cv2.resize(damaged_bgr, (self.image_size, self.image_size))
        r_resized = cv2.resize(restored_bgr, (self.image_size, self.image_size))
        
        d_rgb = cv2.cvtColor(d_resized, cv2.COLOR_BGR2RGB)
        r_rgb = cv2.cvtColor(r_resized, cv2.COLOR_BGR2RGB)
        
        d_tensor = torch.from_numpy(d_rgb).float().permute(2, 0, 1) / 255.0
        r_tensor = torch.from_numpy(r_rgb).float().permute(2, 0, 1) / 255.0
        
        input_tensor = torch.cat([d_tensor, r_tensor], dim=0).unsqueeze(0).to(self.device)
        
        # Infer
        with torch.no_grad():
            pred = self.model(input_tensor)
            
        pred_np = pred.squeeze().cpu().numpy()
        
        # Denormalize (we normalized target by 255.0 in training)
        pred_err = pred_np * 255.0
        
        # Resize back to original
        raw_uncertainty = cv2.resize(pred_err, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # Ensure no negative values (model uses ReLU, but cubic resize might overshoot slightly)
        raw_uncertainty = np.clip(raw_uncertainty, 0, None)
        
        return raw_uncertainty

def save_raw_uncertainty(raw_uncertainty, out_dir):
    out_dir = Path(out_dir)
    np.save(out_dir / "raw_uncertainty.npy", raw_uncertainty)
    
    vis_max = raw_uncertainty.max()
    if vis_max > 0:
        vis = (raw_uncertainty / vis_max) * 255.0
    else:
        vis = raw_uncertainty
        
    # Use COLORMAP_JET or similar for better visualization of uncertainty?
    # Proposal says "normalized for visualization". We will use grayscale for consistency with error_map.
    cv2.imwrite(str(out_dir / "raw_uncertainty.png"), vis.astype(np.uint8))
