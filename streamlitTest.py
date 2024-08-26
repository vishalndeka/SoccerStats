import streamlit as st
import torch
import os
import numpy as np
from typing import Generator, List
import cv2
import tempfile
from detectionUtilities import Detection

def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        yield frame
    video.release()

def detections2boxes(detections: List[Detection], with_confidence: bool = True) -> np.ndarray:
    return np.array([
        [
            detection.rect.top_left.x, 
            detection.rect.top_left.y,
            detection.rect.bottom_right.x,
            detection.rect.bottom_right.y,
            detection.confidence
        ] if with_confidence else [
            detection.rect.top_left.x, 
            detection.rect.top_left.y,
            detection.rect.bottom_right.x,
            detection.rect.bottom_right.y
        ]
        for detection
        in detections
    ], dtype=float)

if __name__=='__main__':
    uploaded_file = st.file_uploader("Choose a file")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', device=0)
    # cap = cv2.VideoCapture("F:\IIT G\Sem 2\EE-722 Video Analytics\Project\clips\\0a2d9b_1.mp4")
    # ret, frame = cap.read()
    # res = model(frame, size=1280)

    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(dir = 'temp/', delete = True)
        tfile.write(uploaded_file.read())
        # frame_iterator = iter(generate_frames(uploaded_file))
        # frame = next(frame_iterator)

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error('Failed to open a video file')
        
        frame_number = 0
        ret, frame = cap.read()
        results = model(frame, size=1280)
        st.write(results)
        st.image(frame, channels="BGR", caption=f"Frame {frame_number}")
        # st.video(uploaded_file, format="video/mp4", autoplay=True)
        # st.image(frame)

        cap.release()
        os.remove(tfile.name)

