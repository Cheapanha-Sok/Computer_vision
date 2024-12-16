import cv2
import HandTrackingModule as htm

# Camera setup for width and height
wCam, hCam = 500, 500
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detectionCon=0.75, maxHands=1)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img , draw=True)
    
    # Get positions for all detected hands
    hands = detector.findPosition(img)
    totalFingers = 0  # Initialize the total count
    
    if hands:
        for lmList in hands:
            fingers = []
            
            # Check the tump is open
            if lmList[tipIds[1]][1] < lmList[0][1]:
                # Left hand
                if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                    fingers.append(0)
            else:
                # Right hand
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(0)
            
            
            # 4 Fingers (index to pinky)
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]: # Finger open
                    fingers.append(0)
            
            # Sum up the fingers for this hand
            totalFingers += len(fingers)
    
    # Display the total finger count
    cv2.putText(img, str(totalFingers), (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    
    cv2.imshow('Finger Counter', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
