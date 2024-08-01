import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(r"C:\Users\hp\Desktop\detection of the bels palsy\mediapipe\CAAI_dataset\bels plasy video datadset\bels plasy videos dataset\Normal\6.mp4")
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 360, [0, 40], invert=True)

idList = [463, 398, 384, 385, 386, 387, 388, 466, 263 , 249, 390, 373 , 374, 380 , 381 , 362, 475 , 477 ]
ratioList = []
blinkCounter = -1
counter = 0
color = (255, 0, 255)
chk = True

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:  # Check if face is detected
        face = faces[0]  # Assuming only one face is detected
        for id in idList:
            if id < len(face):
                cv2.circle(img, face[id], 5, color, cv2.FILLED)

        leftUp = face[386]
        leftDown = face[374]
        leftLeft = face[263]
        leftRight = face[463]
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratioAvg = sum(ratioList) / len(ratioList)
        
        
        if ratioAvg < 19 and chk:
            blinkCounter += 1 
            color = (0, 200, 0)
            chk = False 

        if ratioAvg > 21 and not chk:
            color = (255, 0, 255)
            chk = True
            

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100),
                           colorR=color)

        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640, 360))
        print(ratioAvg)
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (640, 360))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Print total blink count after the video is processed
print("Total Blink Count:", blinkCounter)

cap.release()
cv2.destroyAllWindows()



# if ratioAvg < 25 and counter == 0:
#            blinkCounter += 1
#            color = (0, 200, 0)
#            counter = 1
            
#        if counter != 0:
#            counter += 1
#            if counter > 10:
#                counter = 0
#                color = (255, 0, 255)