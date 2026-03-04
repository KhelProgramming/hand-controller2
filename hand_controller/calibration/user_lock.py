from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np

from hand_controller.ml.geo18 import extract_geo18  # your Geo18 extractor

@dataclass(frozen=True)
class UserProfile:
    mean: np.ndarray  # (18,)
    std: np.ndarray   # (18,)
    max_zdist: float = 3.0

def _zdist(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    std2 = np.maximum(std, 1e-6)
    z = (x - mean) / std2
    return float(np.linalg.norm(z))

class UserLockSelector:
    """User-lock mode: only accept the calibrated user's hand."""

    def __init__(self, profile: UserProfile):
        self.profile = profile

    def pick_index(self, hands: List[Dict[str, Any]], frame_shape=None) -> Optional[int]:
        if not hands:
            return None

        best_i = None
        best_d = 1e18

        for i, h in enumerate(hands):
            f18 = np.asarray(extract_geo18(h["landmarks"]), dtype=np.float64)
            d = _zdist(f18, self.profile.mean, self.profile.std)

            if d < best_d:
                best_d = d
                best_i = i

        if best_i is None:
            return None

        return best_i if best_d <= self.profile.max_zdist else None