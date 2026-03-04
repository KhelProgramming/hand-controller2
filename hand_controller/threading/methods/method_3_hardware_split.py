from __future__ import annotations
import threading, time, queue
from hand_controller.threading.threads.io_capture import io_capture_loop
from hand_controller.threading.threads.action_exec import exec_loop
from hand_controller.threading.types import FramePacket, ActionPacket

from hand_controller.vision import HandTracker
from hand_controller.ml.geo18 import extract_geo18
from hand_controller.ml.gate import GatePolicy, accept
from hand_controller.ml.stabilizer import LabelStabilizer, StablePolicy

import cv2

def _infer_logic_loop(stop: threading.Event,
                     q_frames: queue.Queue,
                     q_action: queue.Queue,
                     metrics: dict,
                     mlp,
                     selector,
                     gate: GatePolicy,
                     stable: StablePolicy):
    tracker = HandTracker(max_num_hands=2)
    stab = LabelStabilizer(stable)

    try:
        while not stop.is_set():
            try:
                pkt: FramePacket = q_frames.get(timeout=0.1)
            except queue.Empty:
                continue

            frame = cv2.flip(pkt.frame_bgr, 1)
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
                outp = ActionPacket(t_capture=pkt.t_capture, action_label=stable_label)
                while q_action.full():
                    try: q_action.get_nowait()
                    except queue.Empty: break
                q_action.put(outp)

    finally:
        tracker.close()

def run(duration_sec: float, metrics: dict,
        mlp,
        selector,
        gate,
        stable,
        action_fn):

    stop = threading.Event()
    q_frames = queue.Queue(maxsize=2)
    q_action = queue.Queue(maxsize=2)

    t_io = threading.Thread(target=io_capture_loop, args=(stop, q_frames), daemon=True)
    t_il = threading.Thread(target=_infer_logic_loop, args=(stop, q_frames, q_action, metrics, mlp, selector, gate, stable), daemon=True)
    t_ex = threading.Thread(target=exec_loop, args=(stop, q_action, metrics, action_fn), daemon=True)

    t_io.start(); t_il.start(); t_ex.start()

    t_end = time.time() + duration_sec
    try:
        while time.time() < t_end:
            time.sleep(0.02)
    finally:
        stop.set()
        for t in (t_io, t_il, t_ex):
            t.join(timeout=0.5)