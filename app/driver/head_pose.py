"""Head pose (yaw, pitch, roll) from 6 face landmarks with cv2.solvePnP.

Accuracy: this fits a generic average-face 3D model with an approximate camera
(focal length = image width, no lens calibration). Treat the angles as a
RELATIVE attention signal (is the head turning or dropping over time?), not as
precise measurements. Expect errors of several degrees.

Sign convention (camera facing the driver, image NOT mirrored):
    yaw   > 0 : face turned toward the image's right side (the driver's own left)
    pitch > 0 : head tilted down (chin toward chest, e.g. looking at a phone in the lap)
    roll  > 0 : head tilted clockwise as seen in the image
All angles are in degrees and are about 0 when the driver faces the camera.
If a webcam preview is mirrored, yaw and roll signs flip.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.driver.face import FaceLandmarks

# MediaPipe Face Mesh indices. "image-left" = left side of the (non-mirrored) image,
# which is the driver's right side.
NOSE_TIP = 1
CHIN = 152
EYE_OUTER_IMAGE_LEFT = 33
EYE_OUTER_IMAGE_RIGHT = 263
MOUTH_IMAGE_LEFT = 61
MOUTH_IMAGE_RIGHT = 291
POSE_LANDMARKS = [NOSE_TIP, CHIN, EYE_OUTER_IMAGE_LEFT, EYE_OUTER_IMAGE_RIGHT, MOUTH_IMAGE_LEFT, MOUTH_IMAGE_RIGHT]

# Generic face model in millimetres, in camera-aligned axes for a frontal face:
# x -> image right, y -> image down, z -> away from the camera (nose tip is the origin).
FACE_MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, 330.0, 65.0],  # chin
        [-225.0, -170.0, 135.0],  # outer eye corner, image-left
        [225.0, -170.0, 135.0],  # outer eye corner, image-right
        [-150.0, 150.0, 125.0],  # mouth corner, image-left
        [150.0, 150.0, 125.0],  # mouth corner, image-right
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Approximate pinhole camera: focal length = image width, principal point = centre."""
    f = float(width)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def rotation_to_angles(rotation: np.ndarray) -> HeadPose:
    """Convert an object->camera rotation matrix to (yaw, pitch, roll) with the convention above."""
    (ax, ay, az), *_ = cv2.RQDecomp3x3(rotation)
    return HeadPose(yaw=float(-ay), pitch=float(ax), roll=float(az))


def angles_to_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of rotation_to_angles (used by tests to build known poses)."""
    ax, ay, az = np.radians([pitch, -yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx  # the order RQDecomp3x3 decomposes


def estimate_head_pose(landmarks: FaceLandmarks) -> HeadPose | None:
    """Return the head pose, or None if the needed landmarks are missing or the fit fails."""
    if landmarks.points.shape[0] <= max(POSE_LANDMARKS) or not landmarks.in_frame(POSE_LANDMARKS):
        return None
    image_points = landmarks.pixel_xy(POSE_LANDMARKS).astype(np.float64)
    cam = camera_matrix(landmarks.image_width, landmarks.image_height)
    ok, rvec, _tvec = cv2.solvePnP(
        FACE_MODEL_3D, image_points, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    pose = rotation_to_angles(rotation)
    if not all(np.isfinite([pose.yaw, pose.pitch, pose.roll])):
        return None
    return HeadPose(round(pose.yaw, 2), round(pose.pitch, 2), round(pose.roll, 2))
