# Hand-Detector-
import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Find hands
    hands, img = detector.findHands(frame)

    # Process each hand
    if hands:
        for hand in hands:
            # Get hand type (Left or Right)
            handType = hand["type"]

            # Get hand landmarks
            lmList = hand["lmList"]  # List of 21 landmarks
            bbox = hand["bbox"]  # Bounding box info x,y,w,h
            centerPoint = hand["center"]  # Center of the hand cx,cy
            handLabel = handType

            # Display hand type and center
            cv2.putText(img, handLabel, (centerPoint[0] - 50, centerPoint[1] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display gesture information
            fingers = detector.fingersUp(hand)
            if fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumb Up"
            elif fingers == [0, 1, 0, 0, 0]:
                gesture = "Index Finger Up"
            elif fingers == [0, 0, 1, 0, 0]:
                gesture = "Middle Finger Up"
            elif fingers == [0, 0, 0, 1, 0]:
                gesture = "Ring Finger Up"
            elif fingers == [0, 0, 0, 0, 1]:
                gesture = "Little Finger Up"
            elif fingers == [1, 1, 1, 1, 1]:
                gesture = "All Fingers Up"
            else:
                gesture = "Unknown Gesture"

            # Display the gesture
            cv2.putText(img, gesture, (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Control', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
