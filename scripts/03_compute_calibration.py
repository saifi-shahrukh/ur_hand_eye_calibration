#!/usr/bin/env python3
"""Compute calibration from saved samples."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, json, numpy as np
from datetime import datetime
from src import HandEyeCalibrator

parser = argparse.ArgumentParser()
parser.add_argument('--samples', default='results/samples/samples_latest.json')
args = parser.parse_args()

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

calibrator = HandEyeCalibrator(config)
if not calibrator.load_samples(args.samples):
    sys.exit(1)

# Load intrinsics

intr_file = config['intrinsics']['file']
if os.path.exists(intr_file):
    with open(intr_file) as f:
        idata = json.load(f)
    intr = idata.get('intrinsics', idata)
    cm = np.array([[intr['fx'],0,intr['cx']],
                    [0,intr['fy'],intr['cy']],[0,0,1]])
    dc = np.array(intr.get('distortion', [0]*5))
else:
    cm, dc = np.eye(3), np.zeros(5)

result = calibrator.compute()
if result:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibrator.save_result(result,
        f"results/calibration/calibration_{ts}.json", cm, dc)
    calibrator.save_result(result,
        "results/calibration/calibration_latest.json", cm, dc)