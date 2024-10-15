
# Vehicle Detection and Counting with YOLOv8

This project implements a vehicle detection and counting system using the YOLOv8 model. The system processes video files to detect vehicles (such as cars, buses, and trucks), annotate the frames, and generate a processed video with detected objects. It also counts the number of vehicles of each type.

## Features

- **YOLOv8 Integration**: Utilizes the YOLOv8 object detection model to identify vehicles in video frames.
- **Real-time Video Processing**: Processes each frame of the video to detect vehicles and annotate them with bounding boxes and confidence scores.
- **Vehicle Counting**: Counts the detected vehicles (car, bus, truck) and displays them on the video frames.
- **Streamlit Interface**: A simple web-based interface for uploading videos, processing them, and downloading the output.
- **Download Processed Video**: Allows users to download the processed video with detected objects.

## How It Works

1. **Upload Video**: The user uploads a video file through the Streamlit interface.
2. **Object Detection**: The YOLOv8 model detects vehicles in each frame of the video.
3. **Annotation**: Each detected vehicle is annotated with a bounding box and the confidence score.
4. **Counting**: Vehicles are counted, and the total count is displayed on the video.
5. **Download**: The processed video is made available for download.

## Components

- **YOLOv8 Model**: Pre-trained YOLOv8 model used for detecting vehicles in video frames.
- **Video Processing**: OpenCV is used to read, process, and write video files.
- **Object Detection**: For each frame, the YOLOv8 model performs object detection and annotates vehicles.
- **Streamlit Interface**: Provides a user-friendly interface for uploading and processing video files.
  
## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/vehicle-detection-yolov8.git
   ```
2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have OpenCV, Streamlit, and the necessary YOLOv8 libraries installed.

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## How to Use

1. Open the Streamlit web interface.
2. Upload a video file (supported formats: `.mp4`, `.avi`, `.mov`).
3. The system processes the video using YOLOv8, detecting and counting vehicles.
4. Once processing is complete, the processed video is displayed.
5. You can download the processed video with annotations.

## Requirements

- Python 3.x
- Streamlit
- OpenCV
- YOLOv8 (ultralytics library)
  
## Example Use Case

This system can be used in traffic analysis to count vehicles in video footage, annotate vehicle detections, and generate data for further analysis, such as traffic density or vehicle flow.

## License

This project is licensed under the MIT License.

--- 

This `README.md` explains the project's key features, installation, and usage without including any code snippets.
