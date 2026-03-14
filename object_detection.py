import argparse
import sys

import cv2
import numpy as np


def detect_red_objects(frame: np.ndarray) -> np.ndarray:
    """Process single frame, return red color mask (same as object_detection.py)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 145, 100])
    upper_red1 = np.array([5, 255, 230])
    lower_red2 = np.array([170, 145, 100])
    upper_red2 = np.array([180, 255, 230])
    lower_red3 = np.array([0, 120, 175])
    upper_red3 = np.array([5, 150, 199])
    lower_red4 = np.array([0, 160, 230])
    upper_red4 = np.array([5, 194, 255])
    lower_red4 = np.array([0, 200, 80])
    upper_red4 = np.array([5, 241, 96])
    raw_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2) | cv2.inRange(hsv, lower_red3, upper_red3) | cv2.inRange(hsv, lower_red4, upper_red4)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return mask
    

def object_from_moments(mask_0_255: np.ndarray, min_area: float = 300.0):
    """
    Compute center, area, radius from mask moments (assumption: single red object).
    Returns: (cx, cy), area_px, radius_px or (None, None, None)
    """
    moments = cv2.moments(mask_0_255)
    if moments['m00'] == 0:
        return None, None, None

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    area = moments['m00']
    radius = np.sqrt(area / np.pi)

    if area < min_area:
        return None, None, None

    return (cx, cy), area, radius



def draw_bars(
    img: np.ndarray,
    cx: int,
    width: int,
    y: int = 90,
    bar_height: int = 16,
    margin: int = 10,
):
    """
    Draw bars showing left/right offset from frame center.
    - Reference bar: full width (within frame)
    - Center: vertical line
    - Fill: from center to object position (left/right)
    """
    x1 = margin
    x2 = width - margin
    center = width // 2

    cv2.rectangle(img, (x1, y), (x2, y + bar_height), (255, 255, 255), 2)
    cv2.line(img, (center, y), (center, y + bar_height), (0, 0, 255), 2)
    if cx < center:
        cv2.rectangle(img, (cx, y), (center, y + bar_height), (0, 255, 0), -1)
    elif cx > center:
        cv2.rectangle(img, (center, y), (cx, y + bar_height), (0, 255, 0), -1)

    return img

def main() -> int:
    parser = argparse.ArgumentParser(description="LAB1: Red object detection (moments + bars)")
    parser.add_argument("--video", default="F1.MOV", help="Path to video file (default: F1.MOV)")
    parser.add_argument("--min-area", type=float, default=300.0, dest="min_area", help="Minimum object area [px^2]")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay_ms = int(1000 / fps) if fps and fps > 1e-3 else 20
    jump_frames = int(fps * 2) if fps and fps > 0 else 60  # 2 seconds

    paused = False

    def process_frame(frame):
        h, w = frame.shape[:2]
        center_x = w // 2
        mask = detect_red_objects(frame)
        center, area, radius = object_from_moments(mask, min_area=args.min_area)

        cv2.imshow("Processed image (HSV)", cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        cv2.imshow("Processed image (HSV + morphology mask)", mask)

        display = frame.copy()
        cv2.line(display, (center_x, 0), (center_x, h), (255, 255, 255), 1)
        if center is not None:
            cx, cy = center
            offset = cx - center_x
            cv2.circle(display, (cx, cy), int(max(radius, 1)), (0, 0, 255), 2)
            cv2.circle(display, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(display, f"Offset: {offset} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            draw_bars(display, cx=cx, width=w, y=40, bar_height=16)
        status = "[PAUSED]" if paused else ""
        cv2.putText(display, f"{status} Space=Pause  A=Back  D=Fwd  S=Start  E=End  Q=Quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Original image (tracking + offset)", display)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)

        if paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord(" "):
            paused = not paused
        elif key == ord("q") or key == 27:
            break
        elif key == ord("a"):
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - jump_frames))
        elif key == ord("d"):
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, cur + jump_frames))
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur + jump_frames)
        elif key == ord("s") or key == ord("0"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == ord("e"):
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
