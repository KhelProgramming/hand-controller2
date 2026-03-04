import time
import cv2
from hand_controller.vision import Camera, HandTracker

def extract_geometric_features(landmarks):
    # Dummy feature extraction
    return [0.0] * 42

def predict_gesture(features):
    # Dummy ML prediction
    return "TEST_GESTURE"

def execute_action(gesture):
    # Dummy OS interaction
    pass

def run_benchmark(duration_sec=10):
    """
    🧵 The 1-Thread Setup (Synchronous)
    This is the absolute baseline.
    """
    print(f"Starting 1-Thread (Synchronous) Benchmark for {duration_sec} seconds...")
    try:
        cam = Camera(index=0, width=640, height=480)
        tracker = HandTracker(max_num_hands=2)
    except Exception as e:
        print("Failed to initialize camera or tracker:", e)
        return
    
    frames_processed = 0
    vision_time_sec = 0.0
    actions_completed = 0
    total_latency_ms = 0.0
    
    for _ in range(20): cam.read()

    start_time = time.time()
    while time.time() - start_time < duration_sec:
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
        
        vision_time_sec += (time.time() - vision_start)
        frames_processed += 1
        
        execute_action(gesture)
        
        latency_ms = (time.time() - capture_time) * 1000
        total_latency_ms += latency_ms
        actions_completed += 1

    cam.release()
    tracker.close()
    
    total_runtime = time.time() - start_time
    avg_pipeline_fps = frames_processed / total_runtime if total_runtime > 0 else 0
    avg_vision_fps = frames_processed / vision_time_sec if vision_time_sec > 0 else 0
    avg_latency = total_latency_ms / actions_completed if actions_completed > 0 else 0
    
    print("\n=== RESULTS: 1-THREAD SETUP (Synchronous) ===")
    print(f"Pipeline FPS:   {avg_pipeline_fps:.2f} FPS")
    print(f"Vision FPS:     {avg_vision_fps:.2f} FPS")
    print(f"Actions Fired:  {actions_completed}")
    print(f"Action Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_benchmark()
