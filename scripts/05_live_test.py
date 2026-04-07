#!/usr/bin/env python3
"""Click on image to transform pixel → base frame coordinates."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, yaml, json, numpy as np
from src import RobotInterface, CameraInterface
from src.transforms import camera_to_base


def main():
    calib_file = 'results/calibration/calibration_latest.json'
    if not os.path.exists(calib_file):
        print("❌ No calibration"); return
    with open(calib_file) as f: calib = json.load(f)
    T_tcp_cam = np.array(calib['transformations']['T_tcp_cam'])
    print(f"✅ Loaded: {calib['metadata']['method']} "
          f"{calib['accuracy']['position_error_mm']:.3f}mm")

    with open('config/config.yaml') as f: config = yaml.safe_load(f)
    robot = RobotInterface(config['robot']['ip'])
    if not robot.connect(): return
    camera = CameraInterface(config['camera']['width'],
                             config['camera']['height'], config['camera']['fps'])
    if not camera.connect(): robot.disconnect(); return

    intr = calib['camera_intrinsics']
    camera.camera_matrix = np.array(intr['camera_matrix'])
    camera.dist_coeffs = np.array(intr['dist_coeffs'])
    fx, fy = camera.camera_matrix[0,0], camera.camera_matrix[1,1]
    cx, cy = camera.camera_matrix[0,2], camera.camera_matrix[1,2]

    click = [None]
    def on_mouse(ev,x,y,fl,p):
        if ev == cv2.EVENT_LBUTTONDOWN: click[0]=(x,y)

    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live", 960, 720)
    cv2.setMouseCallback("Live", on_mouse)
    depth = 0.40

    print("\nClick image | +/- depth | 'q' quit\n")

    while True:
        frame = camera.get_frame()
        if frame is None: continue
        disp = frame.copy()
        Tbt = robot.get_tcp_matrix()
        tcp = robot.get_tcp_pose()

        if click[0]:
            px, py = click[0]
            xc = (px-cx)*depth/fx; yc = (py-cy)*depth/fy
            pb = camera_to_base(np.array([xc,yc,depth]), T_tcp_cam, Tbt)
            cv2.drawMarker(disp, (px,py), (0,255,0), cv2.MARKER_CROSS, 20, 2)
            y = 25
            for t, col in [
                (f"Pixel: ({px},{py})",(0,255,0)),
                (f"Cam: [{xc:.4f},{yc:.4f},{depth:.4f}]m",(0,255,255)),
                (f"Base: [{pb[0]:.4f},{pb[1]:.4f},{pb[2]:.4f}]m",(255,255,0)),
                (f"Depth: {depth:.2f}m [+/-]",(200,200,200)),
            ]:
                cv2.putText(disp,t,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
                y+=28
        else:
            cv2.putText(disp,"Click to transform",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        cv2.putText(disp,f"TCP:[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}]",
                    (10,disp.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),1)
        cv2.imshow("Live", disp)
        key = cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key in (ord('+'),ord('=')): depth+=0.05
        elif key==ord('-'): depth=max(0.1,depth-0.05)

    cv2.destroyAllWindows(); robot.disconnect(); camera.disconnect()

if __name__=="__main__":
    main()