"""Intel RealSense wrapper."""
import numpy as np
import json
import os


class CameraInterface:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros(5)

    def connect(self, enable_depth=False) -> bool:
        try:
            import pyrealsense2 as rs
            self._rs = rs
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self.width, self.height,
                              rs.format.bgr8, self.fps)
            if enable_depth:
                cfg.enable_stream(rs.stream.depth, self.width, self.height,
                                  rs.format.z16, self.fps)
            self.profile = self.pipeline.start(cfg)
            self.align = rs.align(rs.stream.color) if enable_depth else None

            # Factory intrinsics as default
            stream = self.profile.get_stream(rs.stream.color)
            intr = stream.as_video_stream_profile().get_intrinsics()
            self.camera_matrix = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]], dtype=np.float64)
            self.dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

            # Warm up
            print(f"📷 Initializing camera ({self.width}x{self.height}"
                  f" @ {self.fps}fps)...")
            print("   Warming up camera...", end="", flush=True)
            for _ in range(30):
                self.pipeline.wait_for_frames()
            print(" done")
            print("✅ Camera initialized")
            return True
        except Exception as e:
            print(f"❌ Camera failed: {e}")
            return False

    def disconnect(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass

    def get_frame(self):
        """Return BGR colour frame or None."""
        frames = self.pipeline.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)
        cf = frames.get_color_frame()
        return np.asanyarray(cf.get_data()) if cf else None

    def get_frames(self):
        """Return (colour_frame, depth_frame) or (None, None)."""
        frames = self.pipeline.wait_for_frames()
        if self.align:
            frames = self.align.process(frames)
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        color = np.asanyarray(cf.get_data()) if cf else None
        return color, df

    def deproject_pixel(self, px, py, depth_m):
        """Pixel + depth → 3-D point in camera optical frame."""
        import pyrealsense2 as rs
        intr = rs.intrinsics()
        intr.width, intr.height = self.width, self.height
        intr.ppx = float(self.camera_matrix[0, 2])
        intr.ppy = float(self.camera_matrix[1, 2])
        intr.fx = float(self.camera_matrix[0, 0])
        intr.fy = float(self.camera_matrix[1, 1])
        intr.model = rs.distortion.none
        intr.coeffs = [0] * 5
        return rs.rs2_deproject_pixel_to_point(intr, [px, py], depth_m)

    # ── intrinsics I/O ────────────────────────────────────

    def load_intrinsics(self, filepath: str) -> bool:
        """Load intrinsics from JSON (supports multiple formats)."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Format A: {"intrinsics": {"fx":...}}
            if 'intrinsics' in data:
                intr = data['intrinsics']
                self.camera_matrix = np.array([
                    [intr['fx'], 0, intr['cx']],
                    [0, intr['fy'], intr['cy']],
                    [0, 0, 1]], dtype=np.float64)
                self.dist_coeffs = np.array(
                    intr.get('distortion', [0]*5), dtype=np.float64)

            # Format B: {"camera_matrix":[[...],...], "dist_coeffs":[...]}
            elif 'camera_matrix' in data:
                self.camera_matrix = np.array(
                    data['camera_matrix'], dtype=np.float64)
                self.dist_coeffs = np.array(
                    data['dist_coeffs'], dtype=np.float64)
            else:
                return False

            print(f"✅ Loaded calibrated intrinsics from {filepath}")
            print(f"   fx={self.camera_matrix[0,0]:.2f}, "
                  f"fy={self.camera_matrix[1,1]:.2f}")
            print(f"   cx={self.camera_matrix[0,2]:.2f}, "
                  f"cy={self.camera_matrix[1,2]:.2f}")
            print(f"   distortion={self.dist_coeffs.tolist()}")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load intrinsics: {e}")
            return False

    def save_intrinsics(self, filepath: str, source="factory"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'source': source,
            'resolution': [self.width, self.height],
            'intrinsics': {
                'fx': float(self.camera_matrix[0, 0]),
                'fy': float(self.camera_matrix[1, 1]),
                'cx': float(self.camera_matrix[0, 2]),
                'cy': float(self.camera_matrix[1, 2]),
                'distortion': self.dist_coeffs.tolist()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved intrinsics ({source}) to {filepath}")