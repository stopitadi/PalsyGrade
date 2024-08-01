import cv2
import dlib

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Calculate the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for blink detection
EAR_THRESHOLD = 0.2
# Counter to keep track of consecutive frames where eyes are closed
CONSEC_FRAMES = 3
frame_counter = 0

# Load video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = detector(gray)
    
    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)
        left_eye = []
        right_eye = []

        # Extract left and right eye landmarks
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Calculate eye aspect ratio for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if the eyes are closed
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            # If eyes were closed for a sufficient number of frames, count it as a blink
            if frame_counter >= CONSEC_FRAMES:
                print("Blink detected!")
            frame_counter = 0

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
