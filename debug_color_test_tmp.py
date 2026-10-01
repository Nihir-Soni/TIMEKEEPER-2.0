import os
import cv2
import numpy as np
from Colorization.colorize import DDColorizer
import matplotlib.pyplot as plt

image_path = r'test_images\old\d.png'

# Read the image and run the colorizer
print('Loading colorizer...')
colorizer = DDColorizer.from_checkpoint(model_size='tiny', device='cpu')

print('Processing image...')
img = cv2.imread(image_path)
out_img = colorizer.colorize_bgr(img)

cv2.imwrite('d_colorized.png', out_img)
print('Colorization done. Saved as d_colorized.png')

# Print basic stats to ensure it's not all black
print(f"Output shape: {out_img.shape}")
print(f"Mean value: {out_img.mean():.2f}")
print(f"Max value: {out_img.max()}")
print(f"Min value: {out_img.min()}")

