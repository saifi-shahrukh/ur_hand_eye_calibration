#!/usr/bin/env python3
"""
Camera intrinsic calibration.

Usage:
    python scripts/01_calibrate_intrinsics.py            # full calibration
    python scripts/01_calibrate_intrinsics.py --factory   # save factory intrinsics only
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, yaml, numpy as np
from src import CameraInterface, CharucoDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--factory', action='store_true',
                        help='Save factory intrinsics only (skip calibration)')
    args = parser.parse_args()

    with open('config/config.yaml') as f:
        cfg = yaml.safe_load(f)

    camera = CameraInterface(cfg['camera']['width'],
                             cfg['camera']['height'], cfg['camera']['fps'])
    if not camera.connect():
        return

    out = cfg['intrinsics']['file']
    if args.factory:
        camera.save_intrinsics(out, source='factory')
        camera.disconnect()
        return

    c = cfg['charuco']
    det = CharucoDetector(c['cols'], c['rows'], c['square_size_m'],
                          c['marker_size_m'], c['dictionary'])

    print("\n📷 Camera Intrinsic Calibration")
    print("   Hold charuco board at varied distances/angles")
    print("   'c' capture | 'q' finish (need ≥15 images)\n")

    all_obj, all_img = [], []
    cv2.namedWindow("Intrinsics", cv2.WINDOW_NORMAL)

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        detection = det.detect(frame, camera.camera_matrix,
                               camera.dist_coeffs, draw=True)
        if detection.success and detection.image is not None:
            disp = detection.image

        cv2.putText(disp, f"Captured: {len(all_obj)}  |  'c' capture  'q' quit",
                    (10, disp.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)
        cv2.imshow("Intrinsics", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and detection.success:
            obj_pts, img_pts = det.board.matchImagePoints(
                detection.corners, detection.ids)
            if obj_pts is not None and len(obj_pts) >= 6:
                all_obj.append(obj_pts)
                all_img.append(img_pts)
                print(f"  ✅ Captured {len(all_obj)} ({len(obj_pts)} pts)")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.disconnect()

    if len(all_obj) < 10:
        print(f"❌ Need ≥10 captures, got {len(all_obj)}")
        return

    print(f"\n🧮 Calibrating from {len(all_obj)} images...")
    ret, cm, dc, _, _ = cv2.calibrateCamera(
        all_obj, all_img, (cfg['camera']['width'], cfg['camera']['height']),
        None, None)
    print(f"   RMS reprojection error: {ret:.4f} px")
    print(f"   fx={cm[0,0]:.2f} fy={cm[1,1]:.2f} "
          f"cx={cm[0,2]:.2f} cy={cm[1,2]:.2f}")

    camera.camera_matrix = cm
    camera.dist_coeffs = dc.flatten()
    camera.save_intrinsics(out, source='calibrated')

if __name__ == "__main__":
    main()