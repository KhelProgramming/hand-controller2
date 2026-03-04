from __future__ import annotations
import threading, time

from hand_controller.threading.threads.io_capture import io_capture_loop
from hand_controller.threading.threads.mp_infer import infer_loop
from hand_controller.threading.threads.logic import logic_loop
from hand_controller.threading.threads.action_exec import exec_loop

import queue

def run(duration_sec: float, metrics: dict,
        mlp,
        selector,
        gate,
        stable,
        action_fn):

    stop = threading.Event()

    q_frames = queue.Queue(maxsize=2)
    q_infer = queue.Queue(maxsize=2)
    q_action = queue.Queue(maxsize=2)

    t_io = threading.Thread(target=io_capture_loop, args=(stop, q_frames), daemon=True)
    t_inf = threading.Thread(target=infer_loop, args=(stop, q_frames, q_infer, 2, mlp, selector), daemon=True)
    t_log = threading.Thread(target=logic_loop, args=(stop, q_infer, q_action, gate, stable), daemon=True)
    t_exe = threading.Thread(target=exec_loop, args=(stop, q_action, metrics, action_fn), daemon=True)

    t_io.start(); t_inf.start(); t_log.start(); t_exe.start()

    t_end = time.time() + duration_sec
    try:
        while time.time() < t_end:
            time.sleep(0.02)
    finally:
        stop.set()
        for t in (t_io, t_inf, t_log, t_exe):
            t.join(timeout=0.5)