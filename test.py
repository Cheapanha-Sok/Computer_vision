from mediapipe import solutions
import cv2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# Define indices for fingertips and key landmarks
FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky





def count_fingers(hand_landmarks, handedness_label):
    """
    Count the number of extended fingers for a detected hand.
    """
    extended_fingers = 0

    # Check non-thumb fingers
    for tip_idx in FINGER_TIPS[1:]:  # Skip the thumb
        lower_joint_idx = tip_idx - 2
        if is_finger_extended(hand_landmarks, tip_idx, lower_joint_idx):
            extended_fingers += 1

    # Thumb detection
    if handedness_label == "Right":
        thumb_extended = hand_landmarks[FINGER_TIPS[0]].x < hand_landmarks[FINGER_TIPS[0] - 1].x
    else:  # Left hand
        thumb_extended = hand_landmarks[FINGER_TIPS[0]].x > hand_landmarks[FINGER_TIPS[0] - 1].x

    if thumb_extended:
        extended_fingers += 1

    return extended_fingers


def draw_landmarks_and_count_fingers(rgb_image, results):
    """
    Draw hand landmarks and count the number of extended fingers.
    """
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness_label = results.multi_handedness[idx].classification[0].label

        # Draw hand landmarks
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Count extended fingers
        finger_count = count_fingers(hand_landmarks.landmark, handedness_label)

        # Get bounding box for text placement
        x_coordinates = [lm.x for lm in hand_landmarks.landmark]
        y_coordinates = [lm.y for lm in hand_landmarks.landmark]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Display handedness and finger count
        cv2.putText(annotated_image,
                    f"{handedness_label}: {finger_count} fingers",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


# Initialize Mediapipe Hands
mp_hands = solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Start Video Capture
cap = cv2.VideoCapture(0)

print("Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    # Annotate frame if hands are detected
    if results.multi_hand_landmarks:
        frame = draw_landmarks_and_count_fingers(frame, results)

    # Display annotated frame
    cv2.imshow('Finger Count', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
