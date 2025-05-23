"""
Script to visualize binary masks overlaid on a video.

This script takes a list of binary masks and overlays them on top of a video,
creating a new video with the masks visualized.

Args:
    masks (list): List of numpy arrays containing binary masks
    video_path (str): Path to the input video file
    output_path (str): Path where the output video will be saved
    color (tuple): RGB color for the mask overlay (default: (0, 255, 0))
    alpha (float): Transparency of the overlay (default: 0.5)
"""

import cv2
import numpy as np
import os


def get_frames_from_video(video_path):
    """
    Load all frames from a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        list: List of frames as numpy arrays
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


def visualize_masks_over_video(masks, video_path, output_path, color=(0, 255, 0), alpha=0.5):
    """
    Overlay binary masks on a video and save the result.
    
    Args:
        masks (list): List of numpy arrays containing binary masks
        video_path (str): Path to the input video file
        output_path (str): Path where the output video will be saved
        color (tuple): RGB color for the mask overlay (default: (0, 255, 0))
        alpha (float): Transparency of the overlay (default: 0.5)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create a copy of the frame for overlay
        overlay = frame.copy()
        
        # If we have a mask for this frame, apply it
        if frame_idx < len(masks):
            mask = masks[frame_idx]
            # Ensure mask is binary
            if mask.dtype != bool:
                mask = mask > 0
            
            # Create colored overlay
            colored_mask = np.zeros_like(frame)
            # Properly handle color assignment for each channel
            for i, c in enumerate(color):
                colored_mask[..., i][mask] = c
            
            # Blend the colored mask with the frame
            cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
        
        # Write the frame to output video
        out.write(overlay)
        frame_idx += 1
    
    # Release everything
    cap.release()
    out.release()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize masks over a video")
    parser.add_argument("--color", nargs=3, type=int, default=[0, 255, 0], help="RGB color for overlay (default: 0 255 0)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency of overlay (default: 0.5)")
    
    args = parser.parse_args()
    
    try:
        # Load masks
        masks_path = "./masks_cyto_None_bw_longer.npy"
        masks = np.load(masks_path, allow_pickle=True)
        
        if not isinstance(masks, (list, np.ndarray)):
            raise ValueError("Masks must be a list or numpy array")
        
        video_path = "./bw_longer.mp4"
        output_path = "./masks_over_video_bw_longer.mp4"
            
        # Visualize masks over video
        visualize_masks_over_video(
            masks=masks,
            video_path=video_path,
            output_path=output_path,
            color=tuple(args.color),
            alpha=args.alpha
        )
        print(f"Successfully created visualization at: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

