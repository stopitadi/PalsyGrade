import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

def facial_extraction(image_path):
    img = cv2.imread(image_path)

    # Create a FaceMesh instance
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    # Convert the image to RGB format (MediaPipe uses RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image
    results = face_mesh.process(img_rgb)

    # Check if any face landmarks are detected
    if results.multi_face_landmarks:
        # Access landmarks for the first detected face
        landmarks = results.multi_face_landmarks[0]
        print(landmarks)
    else:
        print("No face landmarks detected.")

    # Display the image
    cv2.imshow('output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img_path = r"C:\Users\hp\Desktop\detection of the bels palsy\mediapipe\outputDroppy\p1.jpg"
    facial_extraction(img_path)
