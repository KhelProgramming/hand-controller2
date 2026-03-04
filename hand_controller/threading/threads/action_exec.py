from __future__ import annotations
import time, queue, threading
from hand_controller.threading.types import ActionPacket

# You will map labels -> your repo actions (mouse/keyboard controllers).
# For now, this is the clean seam where you plug it in.
def exec_loop(stop: threading.Event, in_q: queue.Queue, metrics: dict, action_fn):
    while not stop.is_set():
        try:
            pkt: ActionPacket = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if pkt.action_label is None:
            continue

        action_fn(pkt.action_label)
        metrics["actions_completed"] += 1
        metrics["total_latency_ms"] += (time.time() - pkt.t_capture) * 1000.0