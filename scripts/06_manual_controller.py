#!/usr/bin/env python3
"""
Manual pick-and-place controller using calibrated hand-eye transform.

Controls:
    [a] Approach above detected object
    [t] Travel down to grip height
    [g] Grip (close)
    [o] Open gripper
    [l] Lift up
    [h] Home position
    [p] Print TCP pose
    [q] Quit
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, yaml, json, numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import rtde_receive, rtde_control

try:
    from robotiq_gripper import RobotiqGripper
    HAS_GRIPPER = True
except ImportError:
    HAS_GRIPPER = False
    print("⚠️  robotiq_gripper not found — gripper disabled")

# ── tunables ──────────────────────────────────────────────

GRASP_Z         = 0.005
APPROACH_Z      = 0.080
LIFT_Z          = 0.150
LOWER_HSV       = np.array([50, 100, 60])
UPPER_HSV       = np.array([80, 255, 255])
MIN_AREA        = 500
MOVE_V, MOVE_A  = 0.15, 0.3
SLOW_V, SLOW_A  = 0.05, 0.2
HOME_J          = np.deg2rad([45, -60, -120, -90, 90, 0])


class ManualController:
    def __init__(self):
        print("\n" + "="*60)
        print("🎯 CALIBRATED MANUAL CONTROLLER")
        print("="*60)
        self._load_config()
        self._load_calibration()
        self._connect_robot()
        self._connect_gripper()
        self._init_camera()
        self.last_target = None
        self.active_target = None
        cv2.namedWindow("Ctrl", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ctrl", 960, 720)
        print("\n✅ Ready!  [a]pproach [t]ravel [g]rip [o]pen [l]ift [h]ome [q]uit")

    def _load_config(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cf = os.path.join(base, 'config', 'config.yaml')
        if os.path.exists(cf):
            with open(cf) as f: self.cfg = yaml.safe_load(f)
            self.robot_ip = self.cfg['robot']['ip']
        else:
            self.robot_ip = "172.22.1.139"; self.cfg = None

    def _load_calibration(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in ['results/calibration/calibration_latest.json',
                   'calibration_latest.json']:
            fp = os.path.join(base, p)
            if os.path.exists(fp): break
        else:
            print("❌ No calibration file"); sys.exit(1)

        with open(fp) as f: data = json.load(f)
        self.T_tcp_cam = np.array(data['transformations']['T_tcp_cam'])

        intr = data['camera_intrinsics']
        if 'camera_matrix' in intr:
            cm = np.array(intr['camera_matrix'])
            self.fx, self.fy = float(cm[0][0]), float(cm[1][1])
            self.cx, self.cy = float(cm[0][2]), float(cm[1][2])
        else:
            self.fx = float(intr['fx']); self.fy = float(intr['fy'])
            self.cx = float(intr['cx']); self.cy = float(intr['cy'])

        self.rs_intr = rs.intrinsics()
        self.rs_intr.width, self.rs_intr.height = 640, 480
        self.rs_intr.ppx, self.rs_intr.ppy = self.cx, self.cy
        self.rs_intr.fx, self.rs_intr.fy = self.fx, self.fy
        self.rs_intr.model = rs.distortion.none
        self.rs_intr.coeffs = [0]*5

        t = self.T_tcp_cam[:3, 3]
        print(f"✅ Calibration loaded  T_tcp_cam trans:"
              f" [{t[0]:.4f},{t[1]:.4f},{t[2]:.4f}]m")

    def _connect_robot(self):
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        tcp = self.rtde_r.getActualTCPPose()
        print(f"✅ Robot [{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}]m")

    def _connect_gripper(self):
        self.gripper = None
        if not HAS_GRIPPER: return
        try:
            self.gripper = RobotiqGripper()
            self.gripper.connect(self.robot_ip, 63352)
            self.gripper.activate(); self.gripper.open()
            print("✅ Gripper ready")
        except Exception as e:
            print(f"⚠️  Gripper: {e}"); self.gripper = None

    def _init_camera(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        for _ in range(30): self.pipe.wait_for_frames()
        print("✅ Camera ready")

    # ── core transform ────────────────────────────────────

    def _cam_to_base(self, p_cam, T_base_tcp):
        """p_base = T_base_tcp @ T_tcp_cam @ p_cam"""
        T_base_cam = T_base_tcp @ self.T_tcp_cam
        return (T_base_cam @ np.append(p_cam, 1.0))[:3]

    def _get_T_base_tcp(self):
        tcp = self.rtde_r.getActualTCPPose()
        T = np.eye(4)
        T[:3,:3] = R.from_rotvec(tcp[3:]).as_matrix()
        T[:3,3] = tcp[:3]
        return T, tcp

    # ── detection ─────────────────────────────────────────

    def detect(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        cf = frames.get_color_frame(); df = frames.get_depth_frame()
        if not cf or not df: return None

        img = np.asanyarray(cf.get_data()); disp = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            cv2.imshow("Ctrl", disp); cv2.waitKey(1); return None
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < MIN_AREA: return None

        rect = cv2.minAreaRect(cnt)
        (cpx, cpy), (w, h), angle = rect

        depth_m = df.get_distance(int(cpx), int(cpy))
        if depth_m == 0:
            ds = []
            for dx in range(-5,6):
                for dy in range(-5,6):
                    d = df.get_distance(max(0,min(639,int(cpx)+dx)),
                                        max(0,min(479,int(cpy)+dy)))
                    if 0.1<d<1.0: ds.append(d)
            depth_m = float(np.median(ds)) if ds else 0
        if depth_m == 0: return None

        p_cam = rs.rs2_deproject_pixel_to_point(self.rs_intr, [cpx,cpy], depth_m)
        Tbt, tcp = self._get_T_base_tcp()
        p_base = self._cam_to_base(p_cam, Tbt)

        # Grasp orientation
        a_rad = np.deg2rad(angle)
        if w < h: a_rad += np.pi/2
        R_cam_obj = R.from_euler('z', a_rad).as_matrix()
        T_base_cam = Tbt @ self.T_tcp_cam
        R_base_obj = T_base_cam[:3,:3] @ R_cam_obj
        ox = R_base_obj[:,0].copy(); ox[2]=0
        n = np.linalg.norm(ox)
        ox = ox/n if n>1e-6 else np.array([1,0,0])
        z = np.array([0,0,-1.]); y = np.cross(z,ox); y/=np.linalg.norm(y)
        x = np.cross(y,z)
        rv = R.from_matrix(np.column_stack((x,y,z))).as_rotvec()

        self.last_target = [float(p_base[0]),float(p_base[1]),float(p_base[2]),
                            float(rv[0]),float(rv[1]),float(rv[2])]

        box = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(disp,[box],0,(0,255,0),2)
        cv2.circle(disp,(int(cpx),int(cpy)),5,(0,0,255),-1)
        y = 25
        for t, col in [
            (f"Base:[{p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}]m",(0,255,0)),
            (f"Depth:{depth_m:.3f}m Angle:{angle:.1f}°",(0,255,255)),
            (f"TCP:[{tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f}]m",(200,200,200)),
        ]:
            cv2.putText(disp,t,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2); y+=25
        cv2.imshow("Ctrl", disp); cv2.waitKey(1)
        return self.last_target

    # ── actions ───────────────────────────────────────────

    def approach(self):
        if not self.last_target:
            print("⚠️  No detection"); return
        self.active_target = list(self.last_target)
        self.active_target[2] = APPROACH_Z
        print(f"📐 Approach [{self.active_target[0]:.3f},"
              f"{self.active_target[1]:.3f},{self.active_target[2]:.3f}]")
        self.rtde_c.moveL(self.active_target, MOVE_V, MOVE_A)

    def descend(self):
        if not self.active_target:
            print("⚠️  Press 'a' first"); return
        gp = list(self.active_target); gp[2] = GRASP_Z
        print(f"👇 Descend to Z={GRASP_Z}")
        self.rtde_c.moveL(gp, SLOW_V, SLOW_A)
        self.active_target = gp

    def grip(self):
        if self.gripper: self.gripper.close(); time.sleep(0.5)
    def release(self):
        if self.gripper: self.gripper.open(); time.sleep(0.5)

    def lift(self):
        tcp = list(self.rtde_r.getActualTCPPose())
        tcp[2] = LIFT_Z
        self.rtde_c.moveL(tcp, MOVE_V, MOVE_A)

    def home(self):
        self.rtde_c.moveJ(HOME_J, 1.0, 0.5)
        self.active_target = None

    def run(self):
        while True:
            self.detect()
            key = cv2.waitKey(10)&0xFF
            if   key==ord('q'): break
            elif key==ord('a'): self.approach()
            elif key==ord('t'): self.descend()
            elif key==ord('g'): self.grip()
            elif key==ord('o'): self.release()
            elif key==ord('l'): self.lift()
            elif key==ord('h'): self.home()
            elif key==ord('p'):
                tcp = self.rtde_r.getActualTCPPose()
                print(f"TCP: [{tcp[0]:.4f},{tcp[1]:.4f},{tcp[2]:.4f}]m")

    def stop(self):
        try: self.pipe.stop()
        except: pass
        try: self.rtde_c.stopScript()
        except: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    c = ManualController()
    try: c.run()
    except KeyboardInterrupt: pass
    finally: c.stop()