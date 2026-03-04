from __future__ import annotations
import time, queue, threading
from hand_controller.vision import Camera
from hand_controller.threading.types import FramePacket

# stop is a threading event to signal the loop to stop
# out_q is a queue to send the captured frames to the next pipeline (inference)

def io_capture_loop(stop: threading.Event, out_q: queue.Queue, cam_index=0, w=640, h=480):
    cam = Camera(index=cam_index, width=w, height=h) # this is just our wrapper class 
    for _ in range(20): # this is a camera warmup 
        cam.read()

    try:
        while not stop.is_set(): # captures frame until someone closes the system
            tcap = time.time() # we capture the timestamp as soon as possible to have an accurate timestamp
            ret, frame = cam.read() # ret is the sucess flag, frame is the captured frame in BGR format
            if not ret: 
                continue # skip failed frames
            
            pkt = FramePacket(t_capture=tcap, frame_bgr=frame) # create packet to send to the next stage/pipeline
            
            while out_q.full(): # greedy overwrite or the latest-frame strategy
                try: out_q.get_nowait() # removes one item from the queue. (get_nowait means DO NOT wait until the queue is empty)
                except queue.Empty: break # breaks if the queue is empty, which can happen if another thread consumes the frame
            out_q.put(pkt) # put the packet into the queue, this can block if the queue is full but we just made sure that there would be space so...
    finally:
        cam.release() # release the camera resource when the loop is stopped
        
        