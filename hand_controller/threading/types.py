# hand_controller/threading/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class FramePacket:
    t_capture: float # time stamp of the frame capture
    frame_bgr: Any # this is any but it is actually a numpy array of shape (H,W,3) and dtype uint8
    
@dataclass
class InferPacket:
    t_capture: float # this is the timestamp of the original frame capture, to match with FramePacket
    label: str # this is the predicted label of the gesture
    p1: float # this is the probability of the predicted label
    margin: float # this is the margin between the predicted label and the second best label
    hand_ok: bool # this boolean indicates whether the hand 'passed the calibration check
    
@dataclass
class ActionPacket:
    t_capture: float # timestamp of the frame
    action_label: str | None # this is the label of the action to execute, None if no action should be executed.