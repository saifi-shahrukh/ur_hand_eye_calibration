"""
Transformation math.

Convention:  T_dest_src  →  p_dest = T_dest_src @ p_src

Camera → Base:
    p_tcp  = T_tcp_cam  @ p_cam        (calibrateHandEye output)
    p_base = T_base_tcp @ p_tcp        (from RTDE)

Base → Camera:
    p_cam = inv(T_tcp_cam) @ inv(T_base_tcp) @ p_base
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_to_matrix(pose):
    """UR pose [x,y,z,rx,ry,rz] → 4×4 matrix."""
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(pose[3:6]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def matrix_to_pose(T):
    """4×4 matrix → UR pose [x,y,z,rx,ry,rz]."""
    pose = np.zeros(6)
    pose[:3] = T[:3, 3]
    pose[3:6] = R.from_matrix(T[:3, :3]).as_rotvec()
    return pose


def invert_transform(T):
    """Fast inversion of a 4×4 homogeneous matrix."""
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def transform_point(T, point):
    """p_dest = T_dest_src @ p_src  (single 3-vector)."""
    return (T @ np.append(point, 1.0))[:3]


def camera_to_base(p_cam, T_tcp_cam, T_base_tcp):
    """Camera → TCP → Base."""
    return transform_point(T_base_tcp, transform_point(T_tcp_cam, p_cam))


def base_to_camera(p_base, T_tcp_cam, T_base_tcp):
    """Base → TCP → Camera."""
    p_tcp = transform_point(invert_transform(T_base_tcp), p_base)
    return transform_point(invert_transform(T_tcp_cam), p_tcp)


def rotation_error(R1, R2):
    """Geodesic rotation error (radians)."""
    tr = np.clip(np.trace(R1 @ R2.T), -1.0, 3.0)
    return float(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0)))


def translation_error(t1, t2):
    """Euclidean distance."""
    return float(np.linalg.norm(t1 - t2))