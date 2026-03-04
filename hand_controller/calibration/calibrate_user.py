from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from typing import List
import cv2
import numpy as np

from hand_controller.vision import HandTracker
from hand_controller.ml.geo18 import extract_geo18

@dataclass(frozen=True)
class CalibConfig:
    seconds: float = 5.0
    out_dir: str = "hand_controller/artifacts/user_profiles"
    user_name: str = "default"
    max_num_hands: int = 2

def run_user_lock_calibration(cfg: CalibConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    tracker = HandTracker(max_num_hands=cfg.max_num_hands)

    feats: List[List[float]] = []
    started = False
    t0 = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = tracker.process(rgb)
            hands = tracker.extract_hands(res)  # {"label","landmarks"} :contentReference[oaicite:3]{index=3}

            if not started:
                cv2.putText(frame, "USER-LOCK CALIB: Press SPACE to start (show your neutral/idle hand)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
                cv2.putText(frame, "Press Q to quit",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            else:
                rem = max(0.0, cfg.seconds - (time.time() - t0))
                cv2.putText(frame, f"Calibrating... {rem:0.1f}s left. Keep steady.",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

                if hands:
                    # take the "closest-looking" hand during calibration by size (simple)
                    # (in calibration, usually only the user hand is present anyway)
                    f18 = extract_geo18(hands[0]["landmarks"])
                    feats.append(f18)

                if rem <= 0.001 and len(feats) >= 20:
                    X = np.asarray(feats, dtype=np.float64)
                    mean = X.mean(axis=0)
                    std = X.std(axis=0)

                    payload = {
                        "user_name": cfg.user_name,
                        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "n_frames": int(X.shape[0]),
                        "mean": mean.tolist(),
                        "std": std.tolist(),
                        "max_zdist": 3.0
                    }

                    out_path = os.path.join(cfg.out_dir, f"{cfg.user_name}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)

                    cv2.putText(frame, "ACCEPTED ✅", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.imshow("Calibration", frame)
                    cv2.waitKey(350)
                    return out_path

            cv2.imshow("Calibration", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")):
                raise SystemExit("Quit")
            if (not started) and k == 32:
                started = True
                t0 = time.time()
                feats = []

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()