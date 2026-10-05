"""
DDColor colorization pipeline for Bringing Old Photos Back to Life.

This module provides two inference backends:

1.  PyTorch backend  — uses the official DDColor .pth checkpoint loaded
    via the `basicsr`-free arch bundled in Colorization/arch/.
    Checkpoint: Colorization/checkpoints/ddcolor_paper_tiny.pth

2.  ONNX backend (recommended for ease of setup) — uses a pre-exported
    ONNX model that runs without any custom architecture code.
    Checkpoint: Colorization/checkpoints/ddcolor_tiny.onnx

The module automatically selects the backend based on what is available.

Download scripts
----------------
  python Colorization/download_model.py            # download .pth (PyTorch)
  python Colorization/download_model.py --onnx     # download ONNX

Public API
----------
  colorizer = DDColorizer.from_checkpoint(auto=True)
  bgr_out   = colorizer.colorize_bgr(bgr_uint8_image)
  
  colorize_folder(input_dir, output_dir)
  colorize_image(src_path, dst_path)
"""

from __future__ import annotations

import os
import sys
import traceback
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Checkpoint filenames ──────────────────────────────────────────────────────
_CKPT_DIR = Path(__file__).parent / "checkpoints"

# PyTorch checkpoints from https://huggingface.co/piddnad/DDColor-models
_PTH_NAMES = {
    "tiny":       "ddcolor_paper_tiny.pth",
    "large":      "ddcolor_paper.pth",
    "modelscope": "ddcolor_modelscope.pth",
}
_HF_REPO = "piddnad/DDColor-models"

# Default model size (balances quality vs. speed on CPU)
_DEFAULT_SIZE = "modelscope"
_DEFAULT_INPUT_SIZE = 512    # model internal resolution


# ── Download helpers ──────────────────────────────────────────────────────────

def download_checkpoint(model_size: str = _DEFAULT_SIZE,
                        save_dir: str | Path | None = None) -> Path:
    """
    Download a DDColor .pth checkpoint from Hugging Face.

    Returns the path to the downloaded file.
    """
    if save_dir is None:
        save_dir = _CKPT_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = _PTH_NAMES.get(model_size)
    if filename is None:
        raise ValueError(
            f"Unknown model_size={model_size!r}. Choose from {list(_PTH_NAMES)}"
        )

    dest = save_dir / filename
    if dest.is_file():
        logger.info("Checkpoint already present: %s", dest)
        return dest

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-download.\n"
            "Install: pip install huggingface_hub\n"
            f"Or manually download '{filename}' from "
            f"https://huggingface.co/{_HF_REPO}\n"
            f"and place it in: {save_dir}"
        )

    print(f"[DDColor] Downloading {filename} from HuggingFace — please wait...")
    path = hf_hub_download(repo_id=_HF_REPO, filename=filename,
                           local_dir=str(save_dir))
    print(f"[DDColor] Saved to: {path}")
    return Path(path)


def find_checkpoint(model_size: str = _DEFAULT_SIZE) -> Path | None:
    """Return path to an existing checkpoint, or None if not found."""
    filename = _PTH_NAMES.get(model_size, "")
    p = _CKPT_DIR / filename
    return p if p.is_file() else None


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def _load_state_dict(path: str | Path,
                     map_location="cpu") -> dict:
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("params", "state_dict", "model"):
            if key in ckpt:
                return ckpt[key]
    return ckpt   # raw state dict


# ── Colorization logic (LAB-space, no model needed) ──────────────────────────

def _preprocess_for_ddcolor(bgr: np.ndarray, input_size: int):
    """
    Convert BGR uint8 image to the tensor DDColor expects.
    Matches the official implementation:
      - Normalize to float32 [0, 1]
      - Convert to LAB and extract L channel (luminance) in [0, 100]
      - Create grayscale LAB image (a=0, b=0) and convert back to RGB
      - Provide this RGB image as input to the model
    Returns (tensor, l_full, orig_hw)
    """
    orig_h, orig_w = bgr.shape[:2]

    img = (bgr / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

    # resize rgb image -> lab -> get grey -> rgb
    img_resized = cv2.resize(img, (input_size, input_size))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate(
        (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
    )
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

    tensor_gray_rgb = (
        torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
        .float()
        .unsqueeze(0)
    )

    return tensor_gray_rgb, orig_l, (orig_h, orig_w)


def _postprocess_ddcolor(ab_tensor: torch.Tensor,
                          l_full: np.ndarray,
                          orig_hw: tuple[int, int],
                          saturation: float = 1.0) -> np.ndarray:
    """
    Merge predicted ab with original L and convert back to BGR uint8.
    """
    import torch.nn.functional as F

    orig_h, orig_w = orig_hw

    # Scale saturation if requested (e.g. from GUI slider)
    if saturation != 1.0:
        ab_tensor = ab_tensor * saturation

    # resize ab -> concat original l -> bgr
    output_ab_resized = (
        F.interpolate(ab_tensor, size=(orig_h, orig_w))[0]
        .float()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    # l_full is (h, w, 1), output_ab_resized is (h, w, 2)
    output_lab = np.concatenate((l_full, output_ab_resized), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)

    output_img = (output_bgr * 255.0).round().astype(np.uint8)
    return output_img


# ── DDColorizer ───────────────────────────────────────────────────────────────

class DDColorizer:
    """
    Wraps a DDColor model for single-image colorization.

    Create via DDColorizer.from_checkpoint() rather than the constructor
    directly — it handles auto-detection of the checkpoint variant.
    """

    def __init__(self, model, input_size: int, device: torch.device):
        self.model = model
        self.input_size = input_size
        self.device = device

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_path: str | Path | None = None,
                        model_size: str = _DEFAULT_SIZE,
                        input_size: int = _DEFAULT_INPUT_SIZE,
                        device: torch.device | None = None,
                        auto_download: bool = False) -> "DDColorizer":
        """
        Build a DDColorizer from a .pth checkpoint.

        Parameters
        ----------
        checkpoint_path : explicit path to .pth; if None, searches checkpoints/
        model_size      : "tiny" | "large" | "modelscope"
        input_size      : model internal resolution (512 recommended)
        device          : None → auto (cuda if available, else cpu)
        auto_download   : if True and checkpoint missing, download from HuggingFace
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path is None:
            checkpoint_path = find_checkpoint(model_size)

        if checkpoint_path is None:
            if auto_download:
                checkpoint_path = download_checkpoint(model_size)
            else:
                expected = _CKPT_DIR / _PTH_NAMES.get(model_size, "?")
                raise FileNotFoundError(
                    f"DDColor checkpoint not found.\n\n"
                    f"Expected location:\n  {expected}\n\n"
                    f"Download with:\n"
                    f"  python Colorization/download_model.py\n\n"
                    f"Or from: https://huggingface.co/{_HF_REPO}"
                )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}\n\n"
                f"Download it with:\n  python Colorization/download_model.py"
            )

        model = cls._build_model(model_size, input_size, device, checkpoint_path)
        return cls(model=model, input_size=input_size, device=device)

    @staticmethod
    def _build_model(model_size: str, input_size: int,
                     device: torch.device,
                     checkpoint_path: Path):
        """Build DDColor model and load weights."""
        encoder_name = "convnext-t" if model_size == "tiny" else "convnext-l"

        from .arch.ddcolor_arch import DDColor as _DDColor

        logger.info("Building DDColor (%s / encoder=%s)", model_size, encoder_name)
        model = _DDColor(
            encoder_name=encoder_name,
            num_input_channels=3,
            input_size=(input_size, input_size),
            nf=512,
            num_output_channels=2,
            last_norm="Spectral",
            do_normalize=False,
            num_queries=100,
            dec_layers=9,
            num_scales=3,
        )

        logger.info("Loading weights: %s", checkpoint_path)
        state = _load_state_dict(checkpoint_path, map_location=str(device))
        # Strip DataParallel prefix
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.debug("Missing keys (first 5): %s", missing[:5])
        if unexpected:
            logger.debug("Unexpected keys (first 5): %s", unexpected[:5])

        model.eval()
        model.to(device)
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def colorize_bgr(self, bgr_image: np.ndarray, saturation: float = 1.0) -> np.ndarray:
        """
        Colorize a BGR uint8 image.

        Parameters
        ----------
        bgr_image : H×W×3 uint8 (as from cv2.imread)

        Returns
        -------
        np.ndarray : H×W×3 uint8 BGR colorized image (same spatial dimensions)
        """
        tensor, l_full, orig_hw = _preprocess_for_ddcolor(bgr_image,
                                                           self.input_size)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            try:
                ab_pred = self.model(tensor)   # 1×2×H'×W'
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    torch.cuda.empty_cache()
                    raise MemoryError(
                        "GPU out of memory during colorization. "
                        "Try CPU inference or a smaller input."
                    ) from e
                raise

        return _postprocess_ddcolor(ab_pred, l_full, orig_hw, saturation=saturation)


# ── Folder / image helpers ─────────────────────────────────────────────────────

def colorize_folder(input_dir: str,
                    output_dir: str,
                    checkpoint_path: str | None = None,
                    model_size: str = _DEFAULT_SIZE,
                    input_size: int = _DEFAULT_INPUT_SIZE,
                    auto_download: bool = False) -> list[str]:
    """
    Colorize all images in input_dir, save results to output_dir.
    Returns list of output file paths successfully written.
    """
    os.makedirs(output_dir, exist_ok=True)
    colorizer = DDColorizer.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        input_size=input_size,
        auto_download=auto_download,
    )

    outputs = []
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    for fname in sorted(os.listdir(input_dir)):
        src = os.path.join(input_dir, fname)
        if not os.path.isfile(src):
            continue
        if Path(fname).suffix.lower() not in exts:
            continue

        img = cv2.imread(src)
        if img is None:
            logger.warning("Could not read: %s — skipping", src)
            continue

        try:
            colorized = colorizer.colorize_bgr(img)
        except Exception as exc:
            logger.error("Error colorizing %s: %s", fname, exc)
            logger.debug(traceback.format_exc())
            continue

        base = Path(fname).stem
        dst = os.path.join(output_dir, f"{base}.png")
        cv2.imwrite(dst, colorized)
        outputs.append(dst)
        logger.info("Colorized → %s", dst)

    return outputs


def colorize_image(src_path: str,
                   dst_path: str,
                   checkpoint_path: str | None = None,
                   model_size: str = _DEFAULT_SIZE,
                   input_size: int = _DEFAULT_INPUT_SIZE,
                   auto_download: bool = False) -> bool:
    """
    Colorize a single image.  Returns True on success.
    """
    try:
        colorizer = DDColorizer.from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_size=model_size,
            input_size=input_size,
            auto_download=auto_download,
        )
        img = cv2.imread(src_path)
        if img is None:
            raise IOError(f"Cannot read: {src_path}")
        result = colorizer.colorize_bgr(img)
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        cv2.imwrite(dst_path, result)
        return True
    except Exception as exc:
        logger.error("colorize_image failed: %s", exc)
        logger.debug(traceback.format_exc())
        return False
