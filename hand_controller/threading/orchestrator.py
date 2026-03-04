from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class BenchmarkResult:
    method: str
    duration_sec: float
    frames_processed: int
    vision_time_sec: float
    actions_completed: int
    total_latency_ms: float

    def summary(self) -> dict:
        fps = self.frames_processed / self.duration_sec if self.duration_sec > 0 else 0.0
        vfps = self.frames_processed / self.vision_time_sec if self.vision_time_sec > 0 else 0.0
        alat = self.total_latency_ms / self.actions_completed if self.actions_completed > 0 else 0.0
        return {
            "method": self.method,
            "duration_sec": self.duration_sec,
            "pipeline_fps": round(fps, 2),
            "vision_fps": round(vfps, 2),
            "actions": self.actions_completed,
            "avg_latency_ms": round(alat, 2),
        }

def run_method(method_name: str,
               runner: Callable[[float, dict], None],
               duration_sec: float,
               out_json: Optional[str] = None) -> BenchmarkResult:
    metrics = {
        "frames_processed": 0,
        "vision_time_sec": 0.0,
        "actions_completed": 0,
        "total_latency_ms": 0.0,
    }

    t0 = time.time()
    runner(duration_sec, metrics)
    dt = time.time() - t0

    res = BenchmarkResult(
        method=method_name,
        duration_sec=dt,
        frames_processed=int(metrics["frames_processed"]),
        vision_time_sec=float(metrics["vision_time_sec"]),
        actions_completed=int(metrics["actions_completed"]),
        total_latency_ms=float(metrics["total_latency_ms"]),
    )

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res.summary(), f, indent=2)

    return res