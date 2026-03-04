import time
import threading
import queue
import cv2
from hand_controller.vision import Camera, HandTracker

stop_event = threading.Event()
execution_queue = queue.Queue(maxsize=5)

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

def vision_logic_thread():
    try:
        cam = Camera(index=0, width=640, height=480)
        tracker = HandTracker(max_num_hands=2)
        for _ in range(20): cam.read()

        while not stop_event.is_set():
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
            
            # Use greedy queue strategy to avoid stale commands
            while execution_queue.full():
                try: execution_queue.get_nowait()
                except queue.Empty: break
            execution_queue.put((capture_time, gesture))
                
        cam.release()
        tracker.close()
    except Exception as e:
        import traceback
        traceback.print_exc()

def execution_thread():
    while not stop_event.is_set():
        try:
            capture_time, gesture = execution_queue.get(timeout=0.1)
            execute_action(gesture)
            latency = (time.time() - capture_time) * 1000
            metrics["total_latency_ms"] += latency
            metrics["actions_completed"] += 1
        except queue.Empty:
            continue

def run_benchmark(duration_sec=10):
    print(f"Starting 3-Thread (Hardware Split) Benchmark for {duration_sec} seconds...")
    t_vision = threading.Thread(target=vision_logic_thread, daemon=True)
    t_exec = threading.Thread(target=execution_thread, daemon=True)
    
    t_vision.start()
    t_exec.start()
    
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        time.sleep(0.016)

    stop_event.set()
    t_vision.join()
    t_exec.join()

    total_runtime = time.time() - start_time
    frames = metrics["frames_processed"]
    actions = metrics["actions_completed"]
    avg_pipeline_fps = frames / total_runtime if total_runtime > 0 else 0
    avg_vision_fps = frames / metrics["vision_time_sec"] if metrics["vision_time_sec"] > 0 else 0
    avg_latency = metrics["total_latency_ms"] / actions if actions > 0 else 0
    
    print("\n=== RESULTS: 3-THREAD SETUP (Hardware Split) ===")
    print(f"Pipeline FPS:   {avg_pipeline_fps:.2f} FPS")
    print(f"Vision FPS:     {avg_vision_fps:.2f} FPS")
    print(f"Actions Fired:  {actions}")
    print(f"Action Latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    run_benchmark()
