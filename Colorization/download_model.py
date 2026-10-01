"""
Download DDColor pretrained checkpoint from Hugging Face.

Run this script once before using the Colorize feature:

    python Colorization/download_model.py

The default model is the 'tiny' variant (faster on CPU).
Use --model large for higher quality (requires more RAM/VRAM).
"""

import argparse
import os
import sys

# Make sure project root is on path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from Colorization.colorize import download_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Download DDColor checkpoint")
    parser.add_argument(
        "--model", choices=["tiny", "large", "modelscope"],
        default="tiny",
        help="Which DDColor variant to download (default: tiny)"
    )
    parser.add_argument(
        "--save_dir", default=None,
        help="Directory to save checkpoint (default: Colorization/checkpoints/)"
    )
    args = parser.parse_args()

    print(f"Downloading DDColor '{args.model}' checkpoint...")
    path = download_checkpoint(model_size=args.model, save_dir=args.save_dir)
    print(f"✓ Checkpoint ready at: {path}")
    print()
    print("You can now use the Colorize feature in the GUI.")


if __name__ == "__main__":
    main()
