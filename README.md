# Object Detection

Red object detection and tracking from video using OpenCV. Detects a single red object, computes its center and area via image moments, and visualizes the position relative to the frame center.

## Features

- **Red color detection** – HSV-based masking with four hue ranges to cover various red tones (incl. 0–5, 170–180)
- **Morphological processing** – Opening + closing (5×5 kernel) to reduce noise and fill small gaps
- **Object tracking** – Center and area computed from mask moments (assumes single circular object)
- **Visualization** – Original frame with circle, center line, offset bars, and offset value in px
- **Video playback controls** – Pause, rewind, forward, jump to start/end
- **CLI** – Optional video path and minimum area via `--video` and `--min-area`
- **HSV picker tool** – Tune mask thresholds by hovering over video to see HSV values and IN MASK / OUT status

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Object detection (main script)

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

### HSV picker (calibration tool)

```bash
# Run HSV picker with default video
python hsv_picker.py

# Specify video
python hsv_picker.py --video path/to/video.mp4
```

Displays the red mask. Hover the cursor to see H, S, V values and whether the pixel is **IN MASK** (red) or **OUT**.

## Controls (both scripts)

| Key | Action |
|-----|--------|
| **Space** | Pause / resume |
| **A** | Rewind 2 seconds |
| **D** | Forward 2 seconds |
| **S** or **0** | Jump to start |
| **E** | Jump to end |
| **Q** or **ESC** | Quit |

## How It Works

1. **HSV masking** – Converts each frame to HSV and creates a binary mask for red (four ranges for better coverage).
2. **Morphology** – Opening (removes small noise) then closing (fills small holes).
3. **Moments** – Uses `cv2.moments()` to get object center (cx, cy), area, and radius (circular assumption).
4. **Visualization** – Circle around object, vertical center line, horizontal offset bars, numeric offset (px left/right from center).

## Output

### object_detection.py

- **Window 1:** Original frame with tracking overlay (circle, center line, offset bars, offset text)
- **Window 2:** Processed HSV preview
- **Window 3:** Binary mask (HSV + morphology)

### hsv_picker.py

- **Single window:** Red mask video with H/S/V and IN MASK / OUT overlay when hovering
