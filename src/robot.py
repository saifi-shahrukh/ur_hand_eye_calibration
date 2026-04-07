"""RTDE wrapper for Universal Robots."""
import numpy as np
from .transforms import pose_to_matrix


class RobotInterface:
    def __init__(self, ip: str):
        self.ip = ip
        self.rtde_r = None
        self.rtde_c = None

    def connect(self) -> bool:
        try:
            import rtde_receive
            import rtde_control
            print(f"🔗 Connecting to robot at {self.ip}...")
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
            self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
            tcp = self.rtde_r.getActualTCPPose()
            print(f"✅ Connected. TCP position: "
                  f"[{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}] m")
            return True
        except Exception as e:
            print(f"❌ Robot connection failed: {e}")
            return False

    def disconnect(self):
        try:
            if self.rtde_c:
                self.rtde_c.stopScript()
        except Exception:
            pass

    def get_tcp_pose(self) -> np.ndarray:
        """Return [x,y,z,rx,ry,rz]."""
        return np.array(self.rtde_r.getActualTCPPose(), dtype=np.float64)

    def get_tcp_matrix(self) -> np.ndarray:
        """Return 4×4 T_base_tcp."""
        return pose_to_matrix(self.get_tcp_pose())