# hand_controller/ml/stabilizer.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class StablePolicy:
    stable_frames: int = 8  # tune later

class LabelStabilizer:
    def __init__(self, policy: StablePolicy):
        self.policy = policy
        self._last = None
        self._count = 0

    def update(self, label: str) -> str | None:
        if label != self._last:
            self._last = label
            self._count = 1
        else:
            self._count += 1

        if self._count >= self.policy.stable_frames:
            return self._last
        return None