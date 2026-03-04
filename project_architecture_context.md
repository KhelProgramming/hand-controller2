# Project Architecture Context: Touch (Hand Controller)

This document serves as the permanent, comprehensive shared memory for the gesture recognition application ("Touch"). It details every architectural decision, internal data flow, concurrency model, and granular structural choice up to the current version.

---

## 1. System Overview

### Core Purpose
"Touch" is a computer vision-based application that translates real-time hand gestures tracked from a standard webcam into functional operating system actions (mouse movement, clicks, and keyboard typing). 

### Overarching Architecture
The application uses a modular, multi-threaded architecture with cleanly separated layers to ensure scalability and maintainability, especially for future integration of Machine Learning (ML) models.

**Pipeline Flow:**
1. **Vision:** Captures frames, applies preprocessing (mirroring, color formatting).
2. **Hand Tracking (MediaPipe):** Extracts hand landmarks and handedness (Left/Right).
3. **Gesture Recognition:** Processes landmarks through geometric rules (with ML support planned) to emit standardized `GestureResult` events (e.g., `PALM_FACING`, `PINCH_THUMB_INDEX_DOWN`).
4. **Mode Management:** Evaluates gestures to act as a state machine toggling between `mouse` and `keyboard` modes.
5. **Controllers (`mouse_controller`, `keyboard_controller`):** Interprets current gestures and modes to decide *what* the system should do, returning `Action` objects (e.g., `MoveTo`, `KeyPress`).
6. **Action Executor:** Consumes `Action` objects and interfaces with the OS via `pyautogui` to create actual system side-effects.
7. **UI / Overlay:** Renders a transparent, fullscreen feedback overlay and a settings manipulation window, strictly isolated in the main UI thread.

---

## 2. Setup & Environment Requirements

### Hardware Requirements
- A standard webcam (defaults to OpenCV device index `0`).
- Processing capability suitable for real-time 30FPS inference on MediaPipe (standard modern CPU).

### Software Dependencies
The required packages are defined in `requirements.txt`:
```text
opencv-python
mediapipe
pyautogui
PyQt5
pillow
```

### Python Environment Setup
A Python version between **3.8 to 3.11** is recommended, as MediaPipe historically supports these cleanly.

**Step-by-Step Virtual Environment Setup (Windows):**
```bash
# 1. Create the virtual environment in the project root
python -m venv venv

# 2. Activate the virtual environment
venv\Scripts\activate

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Run the application
python run.py
```

---

## 3. Machine Learning Pipeline

### MediaPipe Configuration (`vision/hand_tracker.py`)
MediaPipe's `Hands` model is configured with:
- `max_num_hands=2`
- `min_detection_confidence=0.7`
- `min_tracking_confidence=0.7`
The tracker expects `RGB` frames. Because OpenCV defaults to `BGR`, the `cv_loop` in `app.py` specifically applies `cv2.cvtColor(BGR2RGB)` before passing the frame. The results are parsed into a normalized list of dictionaries containing `label` ("Left" or "Right") and `landmarks`.

### Gesture Recognition (`gestures/rule_based.py` & `gestures/base.py`)
The system abstracts raw XYZ landmarks into symbolic logic via `GestureResult` objects. The `RuleBasedGestureRecognizer` relies heavily on pixel-space arithmetic:
- **Normalization:** MediaPipe landmarks `[0..1]` are explicitly multiplied by the camera frame dimensions (`640x480`) in `coords.py` so that gesture thresholds represent stable pixel distances.
- **Palm Direction Safety:** Due to the frame mirroring (`cv2.flip(frame, 1)`), the hand orientation math is inverted. For a right hand, it's considered `PALM_FACING` if `thumb.x < pinky.x`.
- **Hysteresis for Clicks:** To replicate a mechanical switch feel, pinches utilize a stateful hysteresis approach. A `press_th` (tighter threshold) triggers a one-shot `DOWN` event, and the state will not reset (re-arm) until the fingers separate past a looser `release_th`. These multipliers correlate to `KEY_PINCH_PRESS_MULTIPLIER` and `KEY_PINCH_RELEASE_MULTIPLIER` in `tuning.py`.
- **Open Hand:** To detect an open hand, the algorithm compares the y-coordinates of the finger tips vs the PIP (Proximal Interphalangeal) joints. If at least 3 fingers are extended (tip above PIP), the hand is "open".

### ML Integration Stub (`gestures/ml_stub.py`)
Currently, ` रूल-based` logic is the default. The `MLGestureRecognizerStub` is a designed placeholder intended for the team's planned KNN model or MediaPipe classifier. Once trained, the ML model will load inside the `__init__`, run inference in `recognize()`, and emit the identical string-based gesture names (e.g., `GESTURE_PALM_FACING`, `GESTURE_PINCH_INDEX`) so the application logic downstream requires exactly **zero** changes.

---

## 4. Concurrency & Threading

The application heavily depends on threading to prevent frame-processing from blocking the PyQt Gui, and vice versa.

### Threading Model
1. **Main UI Thread:** Owns the `QApplication`, the control panel (`MainWindow`), and the translucent `OverlayWindow`. It fundamentally blocks waiting for user events or Qt signals.
2. **Computer Vision Worker Thread (`cv_loop` in `app.py`):** Dedicated to reading camera frames, running MediaPipe tracking, calculating gestures, executing mouse/keyboard events (`pyautogui`), and packing data to render.

### Synchronization & Data Flow
- **Thread Termination:** The UI thread holds a `threading.Event()` (`stop_event`). When the application is stopped, `stop_event.set()` safely terminates the `while` loop in the worker thread.
- **Thread-Safe UI Injection (`ui/signals.py`):** **Worker threads must never directly update Qt widgets.** To safely render the skeleton/keyboard overlays, the `cv_loop` packages strictly primitive / generic Python data into a dictionary (`payload`). It then emits this using `overlay_bus.update_overlay.emit(payload)`. The `OverlaySignalBus` inherits from `QObject` and acts as an asynchronous bridge pushing the payload to the UI thread slot `OverlayWindow.apply_state()`. The UI thread safely unpacks and triggers a `paintEvent`.

---

## 5. Core Functionalities & Mapping

### Mode Management (`controllers/mode_manager.py`)
Driven by the `Thumb + Ring` pinch gesture. It enforces a strict "Hold and Consume" logic. The user must hold the pinch for `MODE_TOGGLE_HOLD_SECONDS` (`0.10s`), and it checks that `PALM_FACING` is true. Once triggered, the toggle is "consumed" to prevent repeated toggling in the same hold, and a strict cooldown (`0.80s`) ensures stability during transitions between Mouse and Keyboard mode.

### Mouse Controller (`controllers/mouse_controller.py`)
Maintains a stateful persistence (`MouseState`) per session to remember previous coordinates for smooth interpolation.
- **Movement (Clutching System):** Movement requires the hand to be both `PALM_FACING` AND `HAND_OPEN`. If the user closes their hand, movement tracking pauses (acts as a clutch to lift the hand/reset position) but clicks are still permitted.
- **Delta Vectoring:** Cursor isn't absolute; it uses relative deltas. Position is taken from the wrist (Landmark 0) in frame pixels, scaled by a tunable `SENSITIVITY`.
- **Smoothing:** A weighted lerp (`smoothing` parameter) against the previous frame alongside a `deadzone` threshold removes micro-jitter.
- **Clicks:**
    - `Thumb + Index Pinch`: Emits a `Click("left")` action. If triggered twice within `double_click_interval`, emits `DoubleClick()`.
    - `Thumb + Middle Pinch`: Emits `Click("right")`.

### Keyboard Controller (`controllers/keyboard_controller.py`)
Computes UI rectangles relative to fullscreen `screen_w/screen_h` on initialization mapping "QWERTY".
- **Tracking:** Unlike the wrist in Mouse Mode, here the "pointer" is specifically the *Index Fingertip* (`Landmark 8`). Supports concurrent dual-hand indexing.
- **Typing Action:** Employs the `GESTURE_PINCH_INDEX_DOWN` edge event. When the pinch fires, the key overlapping the index tip (`x, y`) is pressed.
- **Shortcuts (Speed Boosts):**
    - `Thumb + Middle Pinch (DOWN)`: Directly executes a `Backspace`.
    - `Thumb + Pinky Pinch (DOWN)`: Flags `shift_one_shot=True` in the state machine so the sequential keypress automatically sends `Shift+<key>`.

---

## 6. Component File Map

```text
hand_controller/
├── __init__.py
├── __main__.py               # Package entry execution
├── app.py                    # The "Glue". Initiates cv_loop worker thread and payload emit.
├── config/
│   └── tuning.py             # Central location for hyperparameters/timing modifiers.
├── controllers/
│   ├── action_executor.py    # ONLY location containing `pyautogui` side effect code.
│   ├── actions.py            # Dataclasses defining actions like Click, MoveTo, Hotkey.
│   ├── keyboard_controller.py# Layout bounds and typing logic math.
│   ├── mode_manager.py       # State tracking for Mouse vs Keyboard transitions.
│   └── mouse_controller.py   # Jitter-reduction, clutching, and mouse click logic.
├── core/
│   └── coords.py             # Pure geographic functions (Euclidean distance, normalizing).
├── gestures/
│   ├── base.py               # Interface abstracting the core logic + named String constants.
│   ├── ml_stub.py            # Stub for planned KNN model replacement.
│   └── rule_based.py         # Heavy math determining gesture heuristics (press vs release).
├── ui/
│   ├── main_window.py        # Settings layout; initializes worker threads and UI.
│   ├── overlay_window.py     # Translucent fullscreen renderer parsing cv_loop inputs.
│   └── signals.py            # Qt's QObject message bus handling cross-thread communication.
└── vision/
    ├── camera.py             # Minimal cv2.VideoCapture object-oriented wrapper.
    └── hand_tracker.py       # Encapsulates MediaPipe logic for cleaner separation.
```

---

## 7. Current State & Next Steps

### Architecture Grade
The architecture is well-developed into a robust `MVC/Event-Driven` pipeline. Codebase is modular—vision math doesn't affect GUI rendering, and gesture detection is correctly abstracted away from side-effect execution. Thread lifecycle is handled thoughtfully with signal buses.

### Known State & Defunct Code
- **Defunct Flex-Tap:** A legacy implementation within `keyboard_controller.py` referencing flex tapping (`key_tap_sensitivity`, `key_tap_cooldown`) has been disabled in favor of the newer pinch-to-type approach. Currently sitting as a legacy relic in settings.

### Next Steps & Action Items
1. **ML Model Integration:** Transition focus to injecting the trained classifier logic natively into `gestures/ml_stub.py`. Replace `RuleBasedGestureRecognizer` explicitly in `app.py`.
2. **Keyboard Settings Cleanup:** Refactor `KeyboardSettings` to deprecate flex-tap components and introduce robust modifier key tracking if necessary.
3. **Continuous Profiling:** While smoothing tracks well, frame-rate bottlenecks may occur depending heavily on the webcam processing power; optimizations in `cv_loop` (like skipping MP inference on alternate frames if struggling) could be explored.
