from __future__ import annotations
import queue, threading
from hand_controller.ml.gate import GatePolicy, accept
from hand_controller.ml.stabilizer import LabelStabilizer, StablePolicy
from hand_controller.threading.types import InferPacket, ActionPacket

def logic_loop(stop: threading.Event,
               in_q: queue.Queue,
               out_q: queue.Queue,
               gate: GatePolicy,
               stable: StablePolicy):
    stab = LabelStabilizer(stable)
    try:
        while not stop.is_set():
            try:
                pkt: InferPacket = in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # if hand isn't the calibrated user, treat as Idle
            label = pkt.label if pkt.hand_ok else "Idle"

            # strict gate (no expected_label in live mode)
            if not accept(label, None, pkt.p1, pkt.margin, gate):
                label = "Idle"

            stable_label = stab.update(label)  # only returns when stable
            out = ActionPacket(t_capture=pkt.t_capture, action_label=stable_label)

            while out_q.full():
                try: out_q.get_nowait()
                except queue.Empty: break
            out_q.put(out)
    finally:
        pass