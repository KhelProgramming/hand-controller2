from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from hand_controller.ml.geo18 import extract_geo18
from hand_controller.ml.mlp_global import GlobalMLP
from hand_controller.ml.gate import GatePolicy, accept
from hand_controller.ml.stabilizer import LabelStabilizer, StablePolicy

@dataclass(frozen=True)
class GestureEngineConfig:
    gate: GatePolicy
    stable: StablePolicy

class MLPGestureEngine:
    def __init__(self, mlp: GlobalMLP, selector, cfg: GestureEngineConfig):
        self.mlp = mlp
        self.selector = selector
        self.cfg = cfg
        self.stab = LabelStabilizer(cfg.stable)

    def update(self, hands, frame_shape) -> Optional[str]:
        """
        Returns:
          - None if not yet stable
          - stable label string if stable enough
        """
        if not hands:
            label = "Idle"
            return self.stab.update(label)

        idx = self.selector.pick_index(hands, frame_shape)
        if idx is None:
            label = "Idle"
            return self.stab.update(label)

        f18 = extract_geo18(hands[idx]["landmarks"])
        res = self.mlp.predict(f18)

        label = res.label
        if not accept(label, None, res.p1, res.margin, self.cfg.gate):
            label = "Idle"

        return self.stab.update(label)