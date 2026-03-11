import cv2
import numpy as np

def detect_red_objects(frame: np.ndarray) -> np.ndarray:
    """Process single frame, return red color mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 140, 10])
    upper_red = np.array([10, 255, 255])
    raw_mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return mask
    

def object_from_moments(maska_0_255: np.ndarray, min_pole: float = 300.0):
    """
    Liczy momenty bez konturów (założenie: jest tylko jeden czerwony obiekt).
    Zwraca:
      (cx, cy), pole_px, promien_px  albo (None, None, None)
    """

    moments = cv2.moments(maska_0_255)
    if moments['m00'] == 0:
        return None, None, None

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    pole = moments['m00']
    promien = np.sqrt(pole / np.pi)

    if pole < min_pole:
        return None, None, None

    return (cx, cy), pole, promien


def main() -> int:    
    video = cv2.VideoCapture('F1.MOV')

    fps = video.get(cv2.CAP_PROP_FPS)    
    opoznienie_ms = int(1000 / fps) if fps and fps > 1e-3 else 20

    while True:
        ret, frame = video.read()
        if not ret:
            break
        mask = detect_red_objects(frame)
        cv2.imshow('Result', mask)

        button = cv2.waitKey(opoznienie_ms) & 0xFF
        if button in (ord('q'), 27):
            break

    video.release()
    cv2.destroyAllWindows()
    return 0    

if __name__ == "__main__":
    raise SystemExit(main())
