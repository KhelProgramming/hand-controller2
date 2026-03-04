import time
import threading
import cv2
from hand_controller.vision import Camera, HandTracker

stop_event = threading.Event()

metrics = {
    "frames_processed": 0,
    "vision_time_sec": 0.0,
    "actions_completed": 0,
    "total_latency_ms": 0.0
}

def extract_geometric_features(landmarks):
    return [0.0] * 42

def predict_gesture(features):
    return "TEST_GESTURE"

def execute_action(gesture):
    pass

def worker_thread(duration_sec):
    try:
        cam = Camera(index=0, width=640, height=480)
        tracker = HandTracker(max_num_hands=2)
        for _ in range(20): cam.read()
    except Exception as e:
        print("Init failed:", e)
        return

    start_time = time.time()
    while time.time() - start_time < duration_sec and not stop_event.is_set():
        capture_time = time.time()
        ret, frame = cam.read()
        if not ret: continue
        
        vision_start = time.time()
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        result = tracker.process(rgb_frame)
        hands = tracker.extract_hands(result)
        
        gesture = "NONE"
        if hands:
            features = extract_geometric_features(hands[0]["landmarks"])
            gesture = predict_gesture(features)
            
        metrics["vision_time_sec"] += (time.time() - vision_start)
        metrics["frames_processed"] += 1
        
        execute_action(gesture)
        
        latency_ms = (time.time() - capture_time) * 1000
        metrics["total_latency_ms"] += latency_ms
        metrics["actions_completed"] += 1

    cam.release()
    tracker.close()
    stop_event.set()

def run_benchmark(duration_sec=10):
    print(f"Starting 2-Thread (Baseline) Benchmark for {duration_sec} seconds...")
    
    worker = threading.Thread(target=worker_thread, args=(duration_sec,), daemon=True)
    worker.start()
    
    start_time = time.time()
    while not stop_event.is_set():
        time.sleep(0.016)

    worker.join()
    
    total_runtime = time.time() - start_time
    frames = metrics["frames_processed"]
    actions = metrics["actions_completed"]
    avg_pipeline_fps = frames / total_runtime if total_runtime > 0 else 0
    avg_vision_fps = frames / metrics["vision_time_sec"] if metrics["vision_time_sec"] > 0 else 0
    avg_latency = metrics["total_latency_ms"] / actions if actions > 0 else 0
    
    print("\n=== RESULTS: 2-THREAD SETUP (Baseline) ===")
    print(f"Pipeline FPS:   {avg_pipeline_fps:.2f} FPS")
    print(f"Vision FPS:     {avg_vision_fps:.2f} FPS")
    print(f"Actions Fired:  {actions}")
    print(f"Action Latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    run_benchmark()
