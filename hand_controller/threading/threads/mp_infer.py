# hand_controller/threading/mp_infer.py
from __future__ import annotations
import queue, threading, cv2, numpy as np
from typing import Any
from hand_controller.vision import HandTracker
from hand_controller.ml.geo18 import extract_geo18
from hand_controller.ml.mlp_global import GlobalMLP
from hand_controller.calibration.user_lock import UserLockSelector
from hand_controller.threading.types import FramePacket, InferPacket

def infer_loop(stop: threading.Event, # the signal to stop the loop/thread
               in_q: queue.Queue, # the queue that receives the captured frames from the io_capture
               out_q: queue.Queue, # the queue to send the inference results to the next pipeline 
               tracker_hands=2, # max hands to track
               mlp: GlobalMLP | None = None, # the mlp model for inference
               locker: Any| None = None): # the user hand authentication
    
    tracker = HandTracker(max_num_hands=tracker_hands) # our mediapipe wrapper for hand tracking and feature extraction
    try: 
        while not stop.is_set(): # runs until the system is closed
            try: 
                pkt: FramePacket = in_q.get(timeout=0.1) # waits for a frame packet from the capture thread, with a timeout to allow checking the stop signal
            except queue.Empty:
                continue # if no frame is received within the timeout, check the stop signal again
            
            frame = cv2.flip(pkt.frame_bgr, 1) # flip the frame horizontally for a mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert the frame to RGB format since mediapipe expects RGB input
            
            result = tracker.process(rgb) # run the hand tracker on the frame to get the hand landmarks and labels
            hands = tracker.extract_hands(result) # transforms the mediapipe result into a more stable structure which are list of dicts with 'label' and 'landmarks'
            
            label, p1, margin, ok = "idle", 0.0, 0.0, False # default format for the inference result, if nothing is changed from here it means no hand was detected or the hand did not pass the authentication check
            
            if hands and mlp is not None: # if there are hands detected and we have a mlp model to run inference        
                pick = 0 # default index to pick the hand inside the feats_all
                if locker is not None: #
                    idx = locker.pick_index(hands, frame.shape) # if we have a locker for user authentication, we will use it to pick the index of the hand that is most likely to be the user's hand based on the extracted features.
                    if idx is None:
                        ok = False # if the locker cannot confidently pick a hand, we will set ok to false to indicate that the hand did not pass the authentication check
                        label, p1, margin = "idle", 0.0, 0.0 # we use the default inference result again
                    else:
                        pick = idx # if the locker can pick a hand, we will use that index to select the feature for inference
                        ok = True # we set ok to true to indicate that the hand passed the authentication check
                else: # if we do not have a locker, we will just pick the first hand (index 0) for inference and set ok to true since we are not doing authentication
                    ok = True
                
                if ok: # if the hand passed the authentication check (or we are not doing authentication), we will run inference using the mlp model
                    feats = extract_geo18(hands[pick]["landmarks"])
                    res = mlp.predict(feats) # we run the mlp model to get the predicted label
                    label, p1, margin = res.label, res.p1, res.margin # we exteract the predicted label, probability, and margin from the mlp result
                    
            out = InferPacket(t_capture=pkt.t_capture, label=label, p1=p1, margin=margin, hand_ok=ok) # we create an inference packet to send to the next pipeline, we also include the original timestamp from the frame capture to keep the timing consistent across the pipelines
            
            while out_q.full(): # we use the same greedy overwrite strategy for the inference results to always keep the latest inference result in the queue
                try: out_q.get_nowait() # remove the old inference result if the queue is full
                except queue.Empty: break # if the queue is empty, we can break the loop
            out_q.put(out) # put the new inference result into the queue for the next pipeline to consume
    finally:
        tracker.close() # release any resources used by the tracker when the loop is stopped