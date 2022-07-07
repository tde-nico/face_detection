import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # 1 ?

mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while 1:
	success, img = cap.read()
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = face.process(imgRGB)

	detections = results.detections
	if detections:
		for id, detection in enumerate(detections):
			bboxC = detection.location_data.relative_bounding_box
			h, w, c = img.shape
			bbox = (int(bboxC.xmin * w), int(bboxC.ymin * h),
				int(bboxC.width * w), int(bboxC.height * h))
			cv2.rectangle(img, bbox, (255,0,255), 2)
			cv2.putText(img, f'{int(detection.score[0]*100)}%',
				(bbox[0],bbox[1]-20),
				cv2.FONT_HERSHEY_PLAIN,
				2, (255, 0, 255), 2)

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

	cv2.imshow("Image", img)
	cv2.waitKey(1)
