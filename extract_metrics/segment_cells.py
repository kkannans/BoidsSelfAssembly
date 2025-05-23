import cv2
import numpy as np
from cellpose import models
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from collections import defaultdict
import time
from tqdm import tqdm

def getFramesFromVideo(video_path:str):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    if vidcap is not None:
        success,image = vidcap.read()
        count = 0
        while success:
            frames.append(image)      
            success,image = vidcap.read()
            count += 1
    return frames

def processImageForCellpose(frame):
    """
    Enhanced preprocessing for better cell detection
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame
    return gray_frame

def process_in_batches(frames, model, batch_size=32, diameter=None):
    """
    Process frames in batches for faster execution
    """
    all_masks = []
    
    # Process frames in batches
    for i in tqdm(range(0, len(frames), batch_size), desc="Processing batches"):
        batch_frames = frames[i:i + batch_size]
        
        # Process batch
        masks, flows, styles, diams = model.eval(batch_frames, 
                                               diameter=diameter,
                                               channels=[0,0],
                                               batch_size=batch_size,
                                               flow_threshold=1.0)
        all_masks.extend(masks)
    
    return all_masks

if __name__ == "__main__":
    # load video
    video_path = "./bw_longer.mp4"
    frames = getFramesFromVideo(video_path)
    
    # Process all frames
    processed_frames = [processImageForCellpose(frame) for frame in frames]
    
    # Model setup
    model_name = "cyto"
    diameter = None # auto
    model_cyto = models.Cellpose(gpu=True, model_type=model_name)
    
    # Process in batches
    batch_size = 32  # Adjust based on your GPU memory
    all_masks = process_in_batches(processed_frames, 
                                 model_cyto, 
                                 batch_size=batch_size,
                                 diameter=diameter)
    
    # Save masks
    np.save(f"masks_{model_name}_{diameter}_bw_longer.npy", all_masks)
