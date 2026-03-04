# hand_controller/threading/benchmark_methods.py
from __future__ import annotations

import os
import json
from pathlib import Path

from hand_controller.threading.orchestrator import run_method
from hand_controller.ml.mlp_global import GlobalMLP
from hand_controller.ml.gate import GatePolicy
from hand_controller.ml.stabilizer import StablePolicy

# choose selector mode for benchmark
from hand_controller.tracking.hand_select import ClosestHandSelector
# OR user-lock:
# from hand_controller.calibration.user_lock import UserLockSelector, UserProfile

from hand_controller.threading.methods import (
    method_1_sync,
    method_2_thread_baseline,
    method_3_hardware_split,
    method_4_sweet_spot,
    method_5_beast,
)

HERE = Path(__file__).resolve().parents[2]  # repo root-ish if you run as module
ART = HERE / "hand_controller" / "artifacts"

SCALER = str(ART / "validator_scaler.joblib")
LE     = str(ART / "validator_label_encoder.joblib")
MLP    = str(ART / "validator_MLP.joblib")

def noop_action_fn(label: str) -> None:
    # IMPORTANT: no pyautogui during benchmark
    # (we only measure pipeline speed/latency)
    pass

def main():
    duration = 20.0  # seconds per method

    mlp_model = GlobalMLP(SCALER, LE, MLP)
    selector = ClosestHandSelector()  # kiosk/public mode benchmark

    gate = GatePolicy(min_p1=0.70, min_margin=0.20)
    stable = StablePolicy(stable_frames=8)

    methods = {
        "1_sync": method_1_sync.run,
        "2_thread_baseline": method_2_thread_baseline.run,
        "3_hardware_split": method_3_hardware_split.run,
        "4_sweet_spot": method_4_sweet_spot.run,
        "5_beast": method_5_beast.run,
    }

    results = []
    out_dir = HERE / "hand_controller" / "artifacts" / "thread_bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, runner in methods.items():
        def wrapped_runner(dur, metrics):
            return runner(
                duration_sec=dur,
                metrics=metrics,
                mlp=mlp_model,
                selector=selector,
                gate=gate,
                stable=stable,
                action_fn=noop_action_fn,
            )

        out_json = str(out_dir / f"{name}.json")
        res = run_method(name, wrapped_runner, duration, out_json=out_json)
        print(res.summary())
        results.append(res.summary())

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {out_dir}")

if __name__ == "__main__":
    main()