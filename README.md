

# Real-time Object Detection with YOLOv8 and MPS
## Overview
This code demonstrates real-time object detection using the YOLOv8 model from the Ultralytics library. The YOLO (You Only Look Once) algorithm is a popular real-time object detection algorithm. In this implementation, the script captures video frames from a webcam, applies object detection using YOLOv8, and overlays bounding boxes and class labels on detected objects.

## Dependencies
OpenCV (cv2)
Ultralytics (yolov8)
NumPy
PyTorch (torch)
### Install the required dependencies using the following:

bash
Copy code
**pip install opencv-python ultralytics numpy torch**
Model and Class Names
The YOLOv8 model is loaded using Ultralytics, and the COCO dataset class names are used for object detection. The class names include various common objects such as people, vehicles, animals, and household items.

### Usage
**Clone the repository:**

bash
Copy code
git clone https://github.com/your_username/real-time-object-detection.git
Navigate to the project directory:

bash
Copy code
cd real-time-object-detection
Run the script:

bash
Copy code
python object_detection.py
The script will open a window displaying real-time object detection from your webcam. Press the 'Esc' key to exit the application.

## Important Note
Ensure that your webcam is connected and accessible by the OpenCV library.
The script uses the YOLOv8 model with MPS (Mixed Precision Training). You may need a compatible GPU to run this efficiently.


link to access the video
https://drive.google.com/drive/folders/1QM6mvcELUSAtFJNPzDtKwYbSpfu-rGDh
