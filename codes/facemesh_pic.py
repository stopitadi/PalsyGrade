import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

def draw_face_landmarks(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Initialize mediapipe face mesh
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

    # Convert image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to detect face landmarks
    results = faceMesh.process(imgRGB)

    # Draw landmarks on the image
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACEMESH_CONTOURS,
                                  mpDraw.DrawingSpec(thickness=1, circle_radius=1),
                                  mpDraw.DrawingSpec(thickness=1, circle_radius=1))

    # Show the image with landmarks
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()

# Call the function with the provided image path
image_path = r"C:\Users\hp\Desktop\detection of the bels palsy\pic01.jpg"
draw_face_landmarks(image_path)
