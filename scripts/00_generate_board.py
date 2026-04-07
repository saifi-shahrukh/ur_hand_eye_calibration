#!/usr/bin/env python3
"""Generate a printable ChArUco board image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, yaml
from src import CharucoDetector

with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)
c = cfg['charuco']
det = CharucoDetector(c['cols'], c['rows'], c['square_size_m'],
                      c['marker_size_m'], c['dictionary'])
img = det.generate_board_image(2500)
os.makedirs('results', exist_ok=True)
path = 'results/charuco_board.png'
cv2.imwrite(path, img)
print(f"\n💾 Board saved to {path}")
print(f"   Print at 100% scale, then measure square size with calipers")
print(f"   Update config.yaml if measured size differs from "
      f"{c['square_size_m']*1000:.0f}mm\n")