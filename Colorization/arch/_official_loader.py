"""
Self-contained DDColor loader using the official arch files copied from
https://github.com/piddnad/DDColor (Apache 2.0 License).

Uses ddcolor_arch_official.py / unet_official.py etc. which are verbatim
copies of the official implementation, adapted only to remove basicsr imports.
"""
from __future__ import annotations
import sys, os

# Patch sys.path so the official arch files can do relative imports
_ARCH_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_official_model(checkpoint_path, model_size='tiny',
                         input_size=512, device=None):
    """Load DDColor using the official arch implementation."""
    import torch
    import importlib.util

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load the official arch file with patched imports ──────────────────────
    from Colorization.arch.ddcolor_arch import DDColor as _DDColor


    encoder_name = 'convnext-t' if model_size == 'tiny' else 'convnext-l'
    model = _DDColor(
        encoder_name=encoder_name,
        decoder_name='MultiScaleColorDecoder',
        num_input_channels=3,
        input_size=(input_size, input_size),
        nf=512,
        num_output_channels=2,
        last_norm='Spectral',
        do_normalize=False,
        num_queries=100,
        dec_layers=9,
        num_scales=3,
    )

    state = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    if isinstance(state, dict):
        for k in ('params', 'state_dict', 'model'):
            if k in state:
                state = state[k]
                break
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, device
