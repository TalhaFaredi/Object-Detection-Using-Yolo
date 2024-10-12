import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
from typing import Dict, Tuple, Any
from numpy import ndarray
from io import BytesIO

# Define the Analyse class
class Analyse:
    def __init__(self) -> None:
        self.clswise_count = {}
        pass

    def Yolo_model(self, model_path: os.path = "yolov8s.pt") -> YOLO:
        """
        Load the YOLOv8 model.
        Args:
            model_path: Path to the YOLOv8 weights file.
        Returns:
            YOLO: Loaded YOLO model.
        """
        self.model = YOLO(model_path)
        return self.model

    def setup_frame_and_ultralytics(self, video_path: str) -> Tuple[cv2.VideoCapture, Any, Any, Any]:
        """
        Set up the video capture and get video properties.
        Args:
            video_path: Path to the video file.
        Returns:
            tuple: Video capture object, width, height, and FPS of the video.
        """
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        return cap, w, h, fps

    def process_frame(self, frame: ndarray, model: YOLO, w: int, h: int) -> ndarray:
        """
        Process a single video frame for object detection.
        Args:
            frame: Input video frame.
            model: YOLO model for detection.
            w: Width of the video frame.
            h: Height of the video frame.
        Returns:
            ndarray: Annotated video frame.
        """
        results = self.model.track(frame, persist=True, verbose=True)
        if results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            scores = results[0].boxes.conf.cpu().tolist()

            for box, cls, score in zip(boxes, clss, scores):
                x1, y1, x2, y2 = map(int, box)
                detected_object = model.names[int(cls)]
                if detected_object in ["car", "bus", "truck"]:
                    label = f"{detected_object} {score:.2f}"
                    self.clswise_count[model.names[int(cls)]] = self.clswise_count.get(model.names[int(cls)], 0) + 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1 + base_line - 10), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def run(self, video_path: str, output_path: str) -> None:
        """
        Run the YOLO-based analytics on the provided video.
        Args:
            video_path: Path to the video file to be processed.
            output_path: Path to save the output video file.
        """
        # Load YOLO model
        Yolo_model = self.Yolo_model()
        
        # Set up video capture
        cap, w, h, fps = self.setup_frame_and_ultralytics(video_path)

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Process the frame
                detection_frame = self.process_frame(frame, Yolo_model, w, h)

                if out is None:
                    height, width, _ = detection_frame.shape
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Write the processed frame to the output video
                out.write(detection_frame)
            else:
                break

        cap.release()
        if out is not None:
            out.release()

# Streamlit App Code
st.title("Vehicle Detection and Counting with YOLOv8")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Create a temporary file to save the output video
    output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')

    # Process the uploaded video and generate the output video
    model = Analyse()
    model.run(video_path=tfile.name, output_path=output_video.name)

    # Display the processed video
    st.success("Processing complete!")
    st.video(output_video.name)

    # Create a download button for the processed video
    with open(output_video.name, "rb") as video_file:
        video_bytes = video_file.read()
        st.download_button(label="Download Processed Video", data=video_bytes, file_name="processed_video.avi", mime="video/avi")
