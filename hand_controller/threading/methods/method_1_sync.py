from __future__ import annotations
import time
import cv2

from hand_controller.vision import Camera, HandTracker
from hand_controller.ml.geo18 import extract_geo18
from hand_controller.ml.mlp_global import GlobalMLP
from hand_controller.ml.gate import GatePolicy, accept
from hand_controller.ml.stabilizer import LabelStabilizer, StablePolicy

def run(duration_sec: float, metrics: dict,
        mlp: GlobalMLP,
        selector,
        gate: GatePolicy,
        stable: StablePolicy,
        action_fn):
    cam = Camera(index=0, width=640, height=480)
    tracker = HandTracker(max_num_hands=2)
    stab = LabelStabilizer(stable)

    t_end = time.time() + duration_sec

    try:
        while time.time() < t_end:
            tcap = time.time()
            ret, frame = cam.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.time()
            res = tracker.process(rgb)
            hands = tracker.extract_hands(res)
            metrics["vision_time_sec"] += (time.time() - t0)

            metrics["frames_processed"] += 1

            label, p1, margin, hand_ok = "Idle", 0.0, 0.0, False
            if hands:
                idx = selector.pick_index(hands, frame.shape)
                if idx is not None:
                    hand_ok = True
                    f18 = extract_geo18(hands[idx]["landmarks"])
                    out = mlp.predict(f18)
                    label, p1, margin = out.label, out.p1, out.margin

            label = label if hand_ok else "Idle"
            if not accept(label, None, p1, margin, gate):
                label = "Idle"

            stable_label = stab.update(label)
            if stable_label is not None:
                action_fn(stable_label)
                metrics["actions_completed"] += 1
                metrics["total_latency_ms"] += (time.time() - tcap) * 1000.0

    finally:
        cam.release()
        tracker.close()