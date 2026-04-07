"""Hand-eye calibration logic."""
import numpy as np
import cv2
import json
import os
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy.spatial.transform import Rotation as R

from .transforms import (
    pose_to_matrix, rotation_error, translation_error)


@dataclass
class CalibrationSample:
    id: str
    timestamp: str
    tcp_pose: List[float]
    T_base_tcp: List[List[float]]
    T_cam_board: List[List[float]]
    reproj_error_px: float
    viewing_angle_deg: float
    distance_m: float
    corner_count: int


@dataclass
class CalibrationResult:
    method: str
    T_tcp_cam: np.ndarray
    T_cam_tcp: np.ndarray
    position_error_mm: float
    rotation_error_deg: float
    consistency_mm: float
    num_samples: int


METHODS = [
    (cv2.CALIB_HAND_EYE_TSAI,       "TSAI"),
    (cv2.CALIB_HAND_EYE_PARK,       "PARK"),
    (cv2.CALIB_HAND_EYE_HORAUD,     "HORAUD"),
    (cv2.CALIB_HAND_EYE_ANDREFF,    "ANDREFF"),
    (cv2.CALIB_HAND_EYE_DANIILIDIS, "DANIILIDIS"),
]


class HandEyeCalibrator:
    def __init__(self, config: dict):
        self.config = config
        self.samples: List[CalibrationSample] = []
        os.makedirs("results/calibration", exist_ok=True)

    # ── quality gate ──────────────────────────────────────

    def check_sample_quality(self, detection, tcp_pose
                             ) -> Tuple[bool, str]:
        cfg = self.config['calibration']
        issues = []

        if detection.reproj_error > cfg['max_reproj_error_px']:
            issues.append(f"reproj={detection.reproj_error:.3f}px>"
                          f"{cfg['max_reproj_error_px']}px")
        if detection.viewing_angle < cfg['min_viewing_angle_deg']:
            issues.append(f"angle={detection.viewing_angle:.1f}°<"
                          f"{cfg['min_viewing_angle_deg']}°")
        elif detection.viewing_angle > cfg['max_viewing_angle_deg']:
            issues.append(f"angle={detection.viewing_angle:.1f}°>"
                          f"{cfg['max_viewing_angle_deg']}°")
        if detection.distance < cfg['min_distance_m']:
            issues.append(f"dist={detection.distance*1000:.0f}mm<"
                          f"{cfg['min_distance_m']*1000:.0f}mm")
        elif detection.distance > cfg['max_distance_m']:
            issues.append(f"dist={detection.distance*1000:.0f}mm>"
                          f"{cfg['max_distance_m']*1000:.0f}mm")
        if detection.corner_count < cfg.get('min_corners', 8):
            issues.append(f"corners={detection.corner_count}<"
                          f"{cfg.get('min_corners',8)}")

        if self.samples:
            cp = tcp_pose[:3]
            cr = R.from_rotvec(tcp_pose[3:])
            mpd = mrd = float('inf')
            for s in self.samples:
                pp = np.array(s.tcp_pose)
                mpd = min(mpd, np.linalg.norm(cp - pp[:3]))
                mrd = min(mrd, np.degrees(
                    (cr * R.from_rotvec(pp[3:]).inv()).magnitude()))
            if mpd < cfg['min_position_change_m']:
                issues.append(f"Δpos={mpd*1000:.0f}mm<"
                              f"{cfg['min_position_change_m']*1000:.0f}mm")
            if mrd < cfg['min_rotation_change_deg']:
                issues.append(f"Δrot={mrd:.1f}°<"
                              f"{cfg['min_rotation_change_deg']}°")

        return (True, "OK") if not issues else (False, " | ".join(issues))

    # ── add / save / load samples ─────────────────────────

    def add_sample(self, detection, tcp_pose, image=None) -> bool:
        ok, msg = self.check_sample_quality(detection, tcp_pose)
        if not ok:
            print(f"✗ Rejected: {msg}")
            return False
        sid = f"sample_{len(self.samples):03d}"
        self.samples.append(CalibrationSample(
            id=sid, timestamp=datetime.now().isoformat(),
            tcp_pose=tcp_pose.tolist(),
            T_base_tcp=pose_to_matrix(tcp_pose).tolist(),
            T_cam_board=detection.T_cam_board.tolist(),
            reproj_error_px=float(detection.reproj_error),
            viewing_angle_deg=float(detection.viewing_angle),
            distance_m=float(detection.distance),
            corner_count=int(detection.corner_count)))
        if image is not None:
            d = "results/calibration/images"
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(f"{d}/{sid}.png", image)
        print(f"✅ {sid}: reproj={detection.reproj_error:.3f}px, "
              f"dist={detection.distance*1000:.0f}mm, "
              f"angle={detection.viewing_angle:.1f}°, "
              f"corners={detection.corner_count}")
        return True

    def save_samples(self, filepath):
        data = {'metadata': {'timestamp': datetime.now().isoformat(),
                              'num_samples': len(self.samples),
                              'config': self.config},
                'samples': [asdict(s) for s in self.samples]}
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved {len(self.samples)} samples to {filepath}")

    def load_samples(self, filepath) -> bool:
        try:
            with open(filepath) as f:
                data = json.load(f)
            self.samples = [CalibrationSample(**s) for s in data['samples']]
            print(f"✅ Loaded {len(self.samples)} samples from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return False

    # ── compute ───────────────────────────────────────────

    def compute(self) -> Optional[CalibrationResult]:
        n = len(self.samples)
        mn = self.config['calibration']['min_samples']
        if n < mn:
            print(f"❌ Need {mn} samples, have {n}")
            return None

        print(f"\n{'='*60}")
        print(f"🧮 COMPUTING CALIBRATION ({n} samples)")
        print(f"{'='*60}")

        Rbt, tbt, Rcb, tcb = [], [], [], []
        for s in self.samples:
            B = np.array(s.T_base_tcp)
            C = np.array(s.T_cam_board)
            Rbt.append(B[:3, :3]); tbt.append(B[:3, 3:4])
            Rcb.append(C[:3, :3]); tcb.append(C[:3, 3:4])

        results = []
        for mid, mname in METHODS:
            try:
                Rtc, ttc = cv2.calibrateHandEye(
                    Rbt, tbt, Rcb, tcb, method=mid)
                err = self._error(Rtc, ttc, Rbt, tbt, Rcb, tcb)
                T = np.eye(4); T[:3, :3] = Rtc; T[:3, 3] = ttc.flatten()
                r = CalibrationResult(
                    mname, T, np.linalg.inv(T),
                    float(err['pos'] * 1000),
                    float(np.degrees(err['rot'])),
                    float(err['cons'] * 1000), n)
                results.append(r)
                print(f"  {mname:12s}: pos={r.position_error_mm:.3f}mm, "
                      f"rot={r.rotation_error_deg:.3f}°, "
                      f"cons={r.consistency_mm:.3f}mm")
            except Exception as e:
                print(f"  {mname:12s}: FAILED - {e}")

        if not results:
            print("❌ All methods failed"); return None
        best = min(results, key=lambda r: r.position_error_mm)
        print(f"\n✅ Best method: {best.method}")
        print(f"   Position error: {best.position_error_mm:.3f}mm")
        print(f"   Rotation error: {best.rotation_error_deg:.3f}°")
        return best

    def _error(self, Rtc, ttc, Rbt, tbt, Rcb, tcb):
        X = np.eye(4); X[:3, :3] = Rtc; X[:3, 3] = ttc.flatten()
        Xi = np.linalg.inv(X)
        pe, re = [], []
        n = len(Rbt)
        for i in range(n):
            Ti = np.eye(4); Ti[:3,:3]=Rbt[i]; Ti[:3,3]=tbt[i].flatten()
            Ci = np.eye(4); Ci[:3,:3]=Rcb[i]; Ci[:3,3]=tcb[i].flatten()
            for j in range(i+1, n):
                Tj = np.eye(4); Tj[:3,:3]=Rbt[j]; Tj[:3,3]=tbt[j].flatten()
                Cj = np.eye(4); Cj[:3,:3]=Rcb[j]; Cj[:3,3]=tcb[j].flatten()
                A = np.linalg.inv(Ti) @ Tj
                B = Ci @ np.linalg.inv(Cj)
                Ae = X @ B @ Xi
                pe.append(float(np.linalg.norm(A[:3,3]-Ae[:3,3])))
                Rd = A[:3,:3] @ Ae[:3,:3].T
                tr = np.clip(np.trace(Rd), -1., 3.)
                re.append(float(np.arccos(np.clip((tr-1)/2, -1., 1.))))
        return {'pos': float(np.median(pe)) if pe else 0,
                'rot': float(np.median(re)) if re else 0,
                'cons': float(np.std(pe)) if pe else 0}

    # ── save result ───────────────────────────────────────

    def save_result(self, result: CalibrationResult, filepath: str,
                    camera_matrix, dist_coeffs):
        trans = result.T_tcp_cam[:3, 3]
        quat = R.from_matrix(result.T_tcp_cam[:3, :3]).as_quat()
        euler = R.from_matrix(result.T_tcp_cam[:3, :3]).as_euler(
            'xyz', degrees=True)
        cfg = self.config['calibration']
        hit = bool(
            float(result.position_error_mm)
            <= float(cfg['target_position_error_mm'])
            and float(result.rotation_error_deg)
            <= float(cfg['target_rotation_error_deg']))

        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'method': str(result.method),
                'num_samples': int(result.num_samples),
                'target_achieved': hit},
            'accuracy': {
                'position_error_mm': float(result.position_error_mm),
                'rotation_error_deg': float(result.rotation_error_deg),
                'consistency_mm': float(result.consistency_mm)},
            'transformations': {
                'T_tcp_cam': result.T_tcp_cam.tolist(),
                'T_cam_tcp': result.T_cam_tcp.tolist(),
                'translation_m': trans.tolist(),
                'quaternion_xyzw': quat.tolist(),
                'euler_xyz_deg': euler.tolist()},
            'camera_intrinsics': {
                'camera_matrix': np.array(camera_matrix).tolist(),
                'dist_coeffs': np.array(dist_coeffs).tolist()}}

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved calibration to {filepath}")
        if hit:
            print(f"🎯 TARGET ACHIEVED: "
                  f"{result.position_error_mm:.3f}mm, "
                  f"{result.rotation_error_deg:.3f}°")
        else:
            print(f"⚠️  Target NOT achieved: "
                  f"{result.position_error_mm:.3f}mm, "
                  f"{result.rotation_error_deg:.3f}°")