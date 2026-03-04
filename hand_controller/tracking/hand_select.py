from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Protocol
import numpy as np

# MediaPipe landmarks are normalized. We'll convert to pixels using frame shape.

def _lm_xy_pixels(hand_landmarks, frame_shape) -> np.ndarray:
    """Return (21,2) array of (x_px,y_px)."""
    h, w = frame_shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark], dtype=np.float64)
    return pts

def _palm_width_px(hand_landmarks, frame_shape) -> float:
    # Palm width approx: distance between index MCP (5) and pinky MCP (17)
    pts = _lm_xy_pixels(hand_landmarks, frame_shape)
    return float(np.linalg.norm(pts[5] - pts[17]))

def _bbox_area_px(hand_landmarks, frame_shape) -> float:
    pts = _lm_xy_pixels(hand_landmarks, frame_shape)
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    return float(max(0.0, maxx - minx) * max(0.0, maxy - miny))

def _mean_z(hand_landmarks) -> float:
    # z is normalized-ish; still useful as a tie-breaker
    zs = np.array([lm.z for lm in hand_landmarks.landmark], dtype=np.float64)
    return float(zs.mean())

class HandSelector(Protocol):
    def pick_index(self, hands: List[Dict[str, Any]], frame_shape) -> Optional[int]:
        ...

@dataclass(frozen=True)
class ClosestHandPolicy:
    # weight on apparent size
    w_palm: float = 0.7
    w_bbox: float = 0.3
    # small optional z tie-breaker (negative z = closer in many setups, but not guaranteed)
    w_z: float = 0.05

class ClosestHandSelector:
    """Public/Kiosk mode: choose the hand that looks closest (largest in image)."""

    def __init__(self, policy: ClosestHandPolicy = ClosestHandPolicy()):
        self.policy = policy

    def pick_index(self, hands: List[Dict[str, Any]], frame_shape) -> Optional[int]:
        if not hands:
            return None

        best_i = None
        best_score = -1e18

        for i, h in enumerate(hands):
            lm = h["landmarks"]
            palm = _palm_width_px(lm, frame_shape)
            area = _bbox_area_px(lm, frame_shape)
            zmean = _mean_z(lm)

            # Score: big palm + big bbox; z is weak tie-breaker
            score = self.policy.w_palm * palm + self.policy.w_bbox * np.sqrt(area) + self.policy.w_z * (-zmean)

            if score > best_score:
                best_score = score
                best_i = i

        return best_i