#!/usr/bin/env python3
"""
Verify calibration: board position in base should stay constant
across different robot poses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, yaml, json, numpy as np
from src import RobotInterface, CameraInterface, CharucoDetector
from src.transforms import camera_to_base


def main():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    calib_file = 'results/calibration/calibration_latest.json'
    if not os.path.exists(calib_file):
        print(f"❌ No calibration at {calib_file}"); return
    with open(calib_file) as f:
        calib = json.load(f)

    T_tcp_cam = np.array(calib['transformations']['T_tcp_cam'])

    print(f"\n{'='*60}")
    print(f"🔍 VERIFICATION")
    print(f"{'='*60}")
    print(f"Method: {calib['metadata']['method']}  "
          f"Acc: {calib['accuracy']['position_error_mm']:.3f}mm")
    print(f"Chain: p_base = T_base_tcp @ T_tcp_cam @ p_cam")
    print(f"  'r' record | 'q' quit\n")

    robot = RobotInterface(config['robot']['ip'])
    if not robot.connect(): return
    camera = CameraInterface(config['camera']['width'],
                             config['camera']['height'], config['camera']['fps'])
    if not camera.connect(): robot.disconnect(); return

    intr = calib['camera_intrinsics']
    camera.camera_matrix = np.array(intr['camera_matrix'])
    camera.dist_coeffs = np.array(intr['dist_coeffs'])

    ch = config['charuco']
    det = CharucoDetector(ch['cols'], ch['rows'], ch['square_size_m'],
                          ch['marker_size_m'], ch['dictionary'])

    # mask
    gmask = None
    mp = None
    mf = 'results/samples/gripper_mask.json'
    if os.path.exists(mf):
        with open(mf) as f: mp = json.load(f)

    cv2.namedWindow("Verify", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Verify", 960, 720)
    rec, live = [], []

    while True:
        frame = camera.get_frame()
        if frame is None: continue
        df = frame
        if mp and gmask is None:
            h, w = frame.shape[:2]
            gmask = np.ones((h,w), np.uint8)*255
            bp = int(h*mp.get('bottom_pct',0)/100)
            lp = int(w*mp.get('left_pct',0)/100)
            rp = int(w*mp.get('right_pct',0)/100)
            if bp: gmask[h-bp:,:]=0
            if lp: gmask[:,:lp]=0
            if rp: gmask[:,w-rp:]=0
        if gmask is not None:
            df = cv2.bitwise_and(frame, frame, mask=gmask)

        Tbt = robot.get_tcp_matrix()
        d = det.detect(df, camera.camera_matrix, camera.dist_coeffs)
        disp = frame.copy()

        if d.success:
            bb = camera_to_base(d.T_cam_board[:3,3], T_tcp_cam, Tbt)
            live.append(bb.copy())
            lm = np.linalg.norm(np.std(np.array(live[-50:]),0))*1000 if len(live)>1 else 0
            rm = np.linalg.norm(np.std(np.array(rec),0))*1000 if len(rec)>1 else 0
            if d.corners is not None:
                for c in d.corners:
                    cv2.circle(disp, tuple(c.flatten().astype(int)), 3, (0,255,0), -1)
            y = 25
            for t, col in [
                (f"Board Base: [{bb[0]:.4f},{bb[1]:.4f},{bb[2]:.4f}]m",(0,255,0)),
                (f"Live std: {lm:.2f}mm",(0,255,255)),
                (f"Rec: {len(rec)} | Rec std: {rm:.2f}mm",(255,200,0)),
                (f"Reproj: {d.reproj_error:.3f}px Dist: {d.distance*1000:.0f}mm",(200,200,200)),
            ]:
                cv2.putText(disp, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                y += 25
        else:
            cv2.putText(disp, "No board", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Verify", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and d.success:
            rec.append(bb.copy())
            print(f"  📌 #{len(rec)}: [{bb[0]:.4f},{bb[1]:.4f},{bb[2]:.4f}]")
        elif key == ord('q'): break

    cv2.destroyAllWindows(); robot.disconnect(); camera.disconnect()

    print(f"\n{'='*60}\n📊 RESULTS\n{'='*60}")
    if len(rec) >= 3:
        p = np.array(rec); mn = np.mean(p,0); sd = np.std(p,0)
        total = np.linalg.norm(sd)*1000
        print(f"Poses: {len(rec)}")
        print(f"Mean: [{mn[0]:.4f},{mn[1]:.4f},{mn[2]:.4f}]m")
        print(f"Std:  [{sd[0]*1000:.2f},{sd[1]*1000:.2f},{sd[2]*1000:.2f}]mm")
        print(f"Total: {total:.2f}mm")
        grade = ("🎯 EXCELLENT" if total<1 else "✅ GOOD" if total<3
                 else "⚠️ FAIR" if total<5 else "❌ POOR")
        print(f"\n{grade}")
    else:
        print("Need ≥3 recorded poses")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()