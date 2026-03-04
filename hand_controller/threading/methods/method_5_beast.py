from __future__ import annotations
import threading, time, queue
from hand_controller.threading.threads.io_capture import io_capture_loop
from hand_controller.threading.threads.mp_infer import infer_loop
from hand_controller.threading.threads.logic import logic_loop
from hand_controller.threading.threads.action_exec import exec_loop

def run(duration_sec: float, metrics: dict,
        mlp,
        selector,
        gate,
        stable,
        action_fn):

    stop = threading.Event()

    # slightly bigger queues so stages can breathe
    q_frames = queue.Queue(maxsize=3)
    q_infer  = queue.Queue(maxsize=3)
    q_action = queue.Queue(maxsize=3)

    t_io = threading.Thread(target=io_capture_loop, args=(stop, q_frames), daemon=True)
    t_in = threading.Thread(target=infer_loop, args=(stop, q_frames, q_infer, 2, mlp, selector), daemon=True)
    t_lg = threading.Thread(target=logic_loop, args=(stop, q_infer, q_action, gate, stable), daemon=True)
    t_ex = threading.Thread(target=exec_loop, args=(stop, q_action, metrics, action_fn), daemon=True)

    t_io.start(); t_in.start(); t_lg.start(); t_ex.start()

    t_end = time.time() + duration_sec
    try:
        while time.time() < t_end:
            time.sleep(0.02)
    finally:
        stop.set()
        for t in (t_io, t_in, t_lg, t_ex):
            t.join(timeout=0.5)