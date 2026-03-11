# Object Detection

Red object detection and tracking from video using OpenCV. Detects a single red object, computes its center and area via image moments, and visualizes the position relative to the frame center.

## Features

- **Red color detection** – HSV-based masking with two hue ranges (0–10 and 170–179) to cover red
- **Morphological processing** – Opening + closing to reduce noise and fill small gaps in the mask
- **Object tracking** – Center and area computed from mask moments (assumes a single circular object)
- **Visualization** – Original frame with circle around the object, center line, and offset bars
- **CLI** – Optional video path and minimum area via command-line arguments

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run with default video (F1.MOV)
python object_detection.py

# Specify video file
python object_detection.py --video path/to/video.mp4

# Set minimum object area (px²) to filter small detections
python object_detection.py --min-area 500

# Help
python object_detection.py --help
```

**Controls:** Press `q` or `ESC` to exit.

## How It Works

1. **HSV masking** – Converts each frame to HSV and creates a binary mask for red (two ranges, since red wraps around H=0).
2. **Morphology** – Applies opening (removes small noise) then closing (fills small holes).
3. **Moments** – Uses `cv2.moments()` to get object center (cx, cy), area, and radius (assuming circular shape).
4. **Visualization** – Draws a circle around the object, a vertical center line, and horizontal bars showing left/right offset from the frame center.

## Output

- **Window 1:** Original frame with tracking overlay (circle, center line, offset bars)
- **Window 2:** Processed binary mask (HSV + morphology)
