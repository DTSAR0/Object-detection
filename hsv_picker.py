import cv2
import argparse
import numpy as np

# Shared state for mouse callback
_current_hsv = [0, 0, 0]
_current_pos = [-1, -1]
_current_in_mask = False


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


def show_hsv(event, x, y, flags, param):
    global _current_hsv, _current_pos, _current_in_mask
    frame, hsv, mask = param
    if event == cv2.EVENT_MOUSEMOVE and 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
        _current_pos[0], _current_pos[1] = x, y
        h, s, v = hsv[y, x]
        _current_hsv[0], _current_hsv[1], _current_hsv[2] = int(h), int(s), int(v)
        _current_in_mask = mask[y, x] == 255


def main() -> int:
    parser = argparse.ArgumentParser(description="HSV color picker - hover over video to see pixel values")
    parser.add_argument("--video", default="F1.MOV", help="Path to video file")
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return 1

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    delay_ms = int(1000 / fps) if fps and fps > 1e-3 else 30
    jump_frames = int(fps * 2) if fps and fps > 0 else 60

    paused = False
    frame = None
    hsv = None

    while True:
        if not paused:
            ret, frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if frame is None:
            continue

        mask = detect_red_objects(frame)
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if _current_pos[0] >= 0 and _current_pos[1] >= 0:
            h, s, v = _current_hsv[0], _current_hsv[1], _current_hsv[2]
            cv2.putText(display, f"H={h} S={s} V={v}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.circle(display, (_current_pos[0], _current_pos[1]), 5, (0, 255, 0), 2)
            status = "IN MASK" if _current_in_mask else "OUT"
            color = (0, 255, 0) if _current_in_mask else (0, 0, 255)
            cv2.putText(display, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        h_display, w_display = display.shape[:2]
        cv2.putText(display, "[PAUSED] " if paused else "", (10, h_display - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(display, "Space=Pause  A=Back  D=Fwd  S=Start  E=End  Q=Quit",
                    (10, h_display - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", display)
        cv2.setMouseCallback("frame", show_hsv, (frame, hsv, mask))

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("a"):
            cur = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            video.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - jump_frames))
        elif key == ord("d"):
            cur = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            target = cur + jump_frames
            if total_frames > 0:
                target = min(total_frames - 1, target)
            video.set(cv2.CAP_PROP_POS_FRAMES, target)
        elif key == ord("s") or key == ord("0"):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == ord("e") and total_frames > 0:
            video.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
        # After seek: read new frame so display updates (especially when paused)
        if key in (ord("a"), ord("d"), ord("s"), ord("e")):
            ret, frame = video.read()
            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    video.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    exit(main())
