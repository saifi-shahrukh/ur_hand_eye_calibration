#!/usr/bin/env python3
"""Collect hand-eye calibration samples with interactive gripper masking."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, yaml, json, numpy as np
from datetime import datetime
from src import (RobotInterface, CameraInterface, CharucoDetector,
                 HandEyeCalibrator)


def calibrate_mask(camera):
    """Interactive slider to set gripper mask."""
    print("\n🎭 GRIPPER MASK SETUP")
    print("Adjust sliders to hide gripper fingers, then press 'q'")
    cv2.namedWindow("Mask Setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask Setup", 960, 720)
    cv2.createTrackbar("Bottom %", "Mask Setup", 15, 50, lambda x: None)
    cv2.createTrackbar("Left %",   "Mask Setup", 0,  30, lambda x: None)
    cv2.createTrackbar("Right %",  "Mask Setup", 0,  30, lambda x: None)

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        h, w = frame.shape[:2]
        b = cv2.getTrackbarPos("Bottom %", "Mask Setup")
        l = cv2.getTrackbarPos("Left %",   "Mask Setup")
        r = cv2.getTrackbarPos("Right %",  "Mask Setup")

        overlay = frame.copy()
        if b: cv2.rectangle(overlay, (0,h-int(h*b/100)), (w,h), (0,0,255), -1)
        if l: cv2.rectangle(overlay, (0,0), (int(w*l/100),h), (0,0,255), -1)
        if r: cv2.rectangle(overlay, (w-int(w*r/100),0), (w,h), (0,0,255), -1)
        disp = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        cv2.putText(disp, "Red = masked. Press 'q' when done.", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Mask Setup", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("Mask Setup")
    return {'bottom_pct': b, 'left_pct': l, 'right_pct': r}


def build_mask(shape, params):
    h, w = shape[:2]
    m = np.ones((h, w), dtype=np.uint8) * 255
    bp = int(h * params.get('bottom_pct', 0) / 100)
    lp = int(w * params.get('left_pct', 0) / 100)
    rp = int(w * params.get('right_pct', 0) / 100)
    if bp: m[h-bp:, :] = 0
    if lp: m[:, :lp] = 0
    if rp: m[:, w-rp:] = 0
    return m


def main():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    robot = RobotInterface(config['robot']['ip'])
    if not robot.connect():
        return

    camera = CameraInterface(config['camera']['width'],
                             config['camera']['height'],
                             config['camera']['fps'])
    if not camera.connect():
        robot.disconnect(); return

    intr_file = config['intrinsics']['file']
    if os.path.exists(intr_file):
        camera.load_intrinsics(intr_file)

    ch = config['charuco']
    detector = CharucoDetector(ch['cols'], ch['rows'], ch['square_size_m'],
                               ch['marker_size_m'], ch['dictionary'])
    calibrator = HandEyeCalibrator(config)
    cfg = config['calibration']

    # ── Mask setup ────────────────────────────────────────
    mask_params = calibrate_mask(camera)
    print(f"✅ Mask set: bottom={mask_params['bottom_pct']}%, "
          f"left={mask_params['left_pct']}%, right={mask_params['right_pct']}%")
    os.makedirs('results/samples', exist_ok=True)
    with open('results/samples/gripper_mask.json', 'w') as f:
        json.dump(mask_params, f, indent=2)

    gmask = None

    print(f"\n{'='*60}")
    print(f"📊 HAND-EYE CALIBRATION - SAMPLE COLLECTION")
    print(f"{'='*60}")
    print(f"Target: {cfg['optimal_samples']} samples")
    print(f"Controls: 'c' capture | 'q' finish")
    print(f"{'='*60}")
    input("\nPress Enter when ready...")

    cv2.namedWindow("Collection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Collection", 960, 720)
    rejected = 0

    while len(calibrator.samples) < cfg['optimal_samples']:
        frame = camera.get_frame()
        if frame is None:
            continue

        if gmask is None:
            gmask = build_mask(frame.shape, mask_params)

        masked = cv2.bitwise_and(frame, frame, mask=gmask)
        tcp_pose = robot.get_tcp_pose()
        det = detector.detect(masked, camera.camera_matrix,
                              camera.dist_coeffs)

        disp = frame.copy()
        # draw mask line
        h, w = disp.shape[:2]
        bp = int(h * mask_params.get('bottom_pct', 0) / 100)
        if bp: cv2.line(disp, (0,h-bp), (w,h-bp), (0,0,200), 1)

        if det.success:
            if det.corners is not None:
                for c in det.corners:
                    cv2.circle(disp, tuple(c.flatten().astype(int)),
                               3, (0,255,0), -1)
            ok, msg = calibrator.check_sample_quality(det, tcp_pose)
            col = (0,255,0) if ok else (0,165,255)
            status = "✓ 'c' to capture" if ok else f"⚠ {msg}"
        else:
            status = "NO BOARD DETECTED"
            col = (0, 0, 255)
            ok = False

        cv2.rectangle(disp, (0,disp.shape[0]-55), (w,disp.shape[0]),
                      (0,0,0), -1)
        cv2.putText(disp, status, (10, disp.shape[0]-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(disp,
                    f"Samples: {len(calibrator.samples)}/{cfg['optimal_samples']}"
                    f"  Rejected: {rejected}",
                    (10, disp.shape[0]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 2)
        if det.success:
            cv2.putText(disp,
                        f"Reproj: {det.reproj_error:.3f}px  "
                        f"Dist: {det.distance*1000:.0f}mm  "
                        f"Angle: {det.viewing_angle:.1f}°",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,255), 2)

        cv2.imshow("Collection", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and det.success:
            if not calibrator.add_sample(det, tcp_pose, frame):
                rejected += 1
        elif key == ord('q'):
            if len(calibrator.samples) >= cfg['min_samples']:
                break
            print(f"  Need ≥{cfg['min_samples']} samples")

    cv2.destroyAllWindows()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibrator.save_samples(f"results/samples/samples_{ts}.json")
    calibrator.save_samples("results/samples/samples_latest.json")
    print(f"\n✅ Collection complete: {len(calibrator.samples)} samples")

    if len(calibrator.samples) >= cfg['min_samples']:
        result = calibrator.compute()
        if result:
            calibrator.save_result(
                result, f"results/calibration/calibration_{ts}.json",
                camera.camera_matrix, camera.dist_coeffs)
            calibrator.save_result(
                result, "results/calibration/calibration_latest.json",
                camera.camera_matrix, camera.dist_coeffs)

    robot.disconnect()
    camera.disconnect()

if __name__ == "__main__":
    main()