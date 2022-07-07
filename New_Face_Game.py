import cv2
import mediapipe as mp
import time
import Face_Detection_Module as fdm


cap = cv2.VideoCapture(0) # 1 ?
detector = fdm.FaceDetector()
pTime = 0
cTime = 0
while 1:
	success, img = cap.read()
	img, bboxs = detector.find_faces(img)
	print(bboxs)
	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime
	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
	cv2.imshow("Image", img)
	cv2.waitKey(1)

