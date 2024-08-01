import mediapipe as mp 
import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(static_image_mode= True , min_tracking_confidence= 0.6 , min_detection_confidence= 0.6)
draw = mp.solutions.drawing_utils
while True:
    _, frm = cap.read()
    cv2.imshow("Window" , frm)
    rgb = cv2.cvtColor(frm , cv2.COLOR_BGR2RGB)
    
    op = face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            draw.draw_landmarks(frm , i , mp.solutions.face_mesh.FACEMESH_CONTOURS,  connection_drawing_spec=draw.DrawingSpec(color=(225 , 225 , 0), circle_radius=1), landmark_drawing_spec=draw.DrawingSpec(color=(225 , 225 , 0 ), circle_radius=1 )  )
            
            
    cv2.imshow("Window", frm)
    
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break 