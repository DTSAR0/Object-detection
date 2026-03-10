import cv2
import numpy as np

video = cv2.VideoCapture('F1.MOV')
while True:
    ret, frame = video.read()
    if not ret:
        break  # відео завершилось або не вдалося зчитати кадр

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red) 

    result = cv2.bitwise_and(frame, frame, mask=mask) 
    cv2.imshow('Mask', result)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
