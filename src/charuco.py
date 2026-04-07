"""ChArUco board detection and pose estimation."""
import numpy as np
import cv2
from typing import Optional
from dataclasses import dataclass


@dataclass
class CharucoDetection:
    success: bool
    T_cam_board: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    reproj_error: float = 999.0
    viewing_angle: float = 0.0
    distance: float = 0.0
    marker_count: int = 0
    corner_count: int = 0
    corners: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None


class CharucoDetector:
    def __init__(self, cols, rows, square_size, marker_size,
                 dictionary="DICT_5X5_250"):
        self.cols = cols
        self.rows = rows
        self.square_size = square_size
        self.marker_size = marker_size

        v = tuple(map(int, cv2.__version__.split('.')[:2]))
        self.new_api = v >= (4, 7)

        dict_id = getattr(cv2.aruco, dictionary)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.board = cv2.aruco.CharucoBoard(
            (cols, rows), square_size, marker_size, self.aruco_dict)

        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 5
        self.params.cornerRefinementMaxIterations = 30
        self.params.cornerRefinementMinAccuracy = 0.01
        self.params.adaptiveThreshWinSizeMin = 3
        self.params.adaptiveThreshWinSizeMax = 23
        self.params.adaptiveThreshWinSizeStep = 10

        if self.new_api:
            self.charuco_det = cv2.aruco.CharucoDetector(self.board)
            self.charuco_det.setDetectorParameters(self.params)

        print(f"✅ Charuco detector: {cols}x{rows}, "
              f"square={square_size*1000:.0f}mm, "
              f"marker={marker_size*1000:.0f}mm")
        print(f"   OpenCV {cv2.__version__} "
              f"({'new' if self.new_api else 'legacy'} API)")

    def detect(self, image, camera_matrix, dist_coeffs, draw=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        disp = image.copy() if draw else None
        if self.new_api:
            return self._new(gray, disp, camera_matrix, dist_coeffs, draw)
        return self._legacy(gray, disp, camera_matrix, dist_coeffs, draw)

    def generate_board_image(self, ppm=2000):
        w = int(self.cols * self.square_size * ppm)
        h = int(self.rows * self.square_size * ppm)
        return self.board.generateImage((w, h))

    # ── new API ───────────────────────────────────────────

    def _new(self, gray, disp, cm, dc, draw):
        cc, cid, mc, mid = self.charuco_det.detectBoard(gray)
        nm = 0 if mid is None else len(mid)
        nc = 0 if cc is None else len(cc)
        if nm < 4 or nc < 4:
            return CharucoDetection(False, marker_count=nm,
                                    corner_count=nc, image=disp)

        op, ip = self.board.matchImagePoints(cc, cid)
        if op is None or len(op) < 4:
            return CharucoDetection(False, marker_count=nm,
                                    corner_count=nc, image=disp)

        ok, rv, tv = cv2.solvePnP(op, ip, cm, dc)
        if not ok:
            return CharucoDetection(False, marker_count=nm,
                                    corner_count=nc, image=disp)

        return self._build(rv, tv, op, ip, cc, cid, mc, mid,
                           nm, nc, cm, dc, disp, draw)

    # ── legacy API ────────────────────────────────────────

    def _legacy(self, gray, disp, cm, dc, draw):
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.params)
        nm = 0 if ids is None else len(ids)
        if nm < 4:
            return CharucoDetection(False, marker_count=nm, image=disp)

        ret, cc, cid = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, self.board, cm, dc)
        nc = 0 if cc is None else len(cc)
        if not ret or nc < 4:
            return CharucoDetection(False, marker_count=nm,
                                    corner_count=nc, image=disp)

        ok, rv, tv = cv2.aruco.estimatePoseCharucoBoard(
            cc, cid, self.board, cm, dc, None, None)
        if not ok:
            return CharucoDetection(False, marker_count=nm,
                                    corner_count=nc, image=disp)

        op = self.board.getChessboardCorners()
        flat = cid.flatten()
        if max(flat) >= len(op):
            return CharucoDetection(False, marker_count=nm,
                                    corner_count=nc, image=disp)
        sel_op = op[flat]

        return self._build(rv, tv, sel_op, cc, cc, cid,
                           corners, ids, nm, nc, cm, dc, disp, draw)

    # ── shared builder ────────────────────────────────────

    def _build(self, rv, tv, op, ip, cc, cid, mc, mid,
               nm, nc, cm, dc, disp, draw):
        Rm, _ = cv2.Rodrigues(rv)
        T = np.eye(4)
        T[:3, :3] = Rm
        T[:3, 3] = tv.flatten()

        proj, _ = cv2.projectPoints(op, rv, tv, cm, dc)
        reproj = float(np.mean(np.linalg.norm(
            np.array(ip).reshape(-1, 2) - proj.reshape(-1, 2), axis=1)))

        bn = Rm @ np.array([0, 0, 1])
        angle = np.degrees(np.arccos(
            np.clip(np.abs(np.dot([0, 0, 1], bn)), 0, 1)))
        dist = float(np.linalg.norm(tv))

        if draw and disp is not None:
            cv2.aruco.drawDetectedMarkers(disp, mc, mid)
            if cc is not None:
                for c in cc:
                    pt = tuple(c.flatten().astype(int))
                    cv2.circle(disp, pt, 4, (0, 255, 0), -1)
            cv2.drawFrameAxes(disp, cm, dc, rv, tv,
                              self.square_size * 2, 3)

        return CharucoDetection(
            True, T, rv, tv, reproj, angle, dist, nm, nc,
            cc, cid, disp)