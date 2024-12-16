import cv2
import HandTrackingModule as htm

# Image path (provide the path to your image)
image_path = "./depositphotos_25651999-stock-photo-woman-hand-showing-the-five.jpg"

# Read the image
img = cv2.imread(image_path)

# Initialize the hand detector
detector = htm.HandDetector(detectionCon=0.75, maxHands=1)
tipIds = [4, 8, 12, 16, 20]

# Process the image
img = detector.findHands(img)

# Create a list of the landmarks detected
lmList = detector.findPosition(img)

if len(lmList) != 0:
    fingers = []

    # Determine if the hand is left or right

    for i, lm in enumerate(lmList):
                x, y = lm[1], lm[2]
                cv2.putText(img, f'{i}', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
    if lmList[tipIds[1]][1] < lmList[0][1]:
        # Left hand
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        # Right hand
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    

    # 4 Fingers (index to pinky)
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    

    totalFingers = fingers.count(1)
    print("Finger States:", lmList)
            # Draw landmarks and values
    for lm in lmList:
        cv2.putText(img, str(lm[0]), (lm[1], lm[2]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.circle(img, (lm[1], lm[2]), 5, (255, 0, 0), cv2.FILLED)

    # Draw the rectangle and display the result
    cv2.rectangle(img, (10, 10), (100, 70), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (34, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

# Display the image
cv2.imshow('Finger Counter', img)

# Wait until a key is pressed, then close
cv2.waitKey(0)
cv2.destroyAllWindows()
