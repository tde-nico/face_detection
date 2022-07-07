import cv2
import mediapipe as mp
import time

class FaceDetector:
	def __init__(self, min_detection_con=0.5):
		self.min_detection_con = min_detection_con
		self.mpFace = mp.solutions.face_detection
		self.face = self.mpFace.FaceDetection(min_detection_con)
		self.mpDraw = mp.solutions.drawing_utils


	def find_faces(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = self.face.process(imgRGB)
		bboxs = []
		self.detections = results.detections
		if self.detections:
			for id, detection in enumerate(self.detections):
				bboxC = detection.location_data.relative_bounding_box
				h, w, c = img.shape
				bbox = (int(bboxC.xmin * w), int(bboxC.ymin * h),
					int(bboxC.width * w), int(bboxC.height * h))
				bboxs.append([id, bbox, detection.score])
				if draw:
					img = self.fancy_draw(img, bbox)
					cv2.putText(img, f'{int(detection.score[0]*100)}%',
						(bbox[0],bbox[1]-20),
						cv2.FONT_HERSHEY_PLAIN,
						2, (255, 0, 255), 2)
		return img, bboxs

	
	def fancy_draw(self, img, bbox, length=30, thick=5, rect_thick=1):
		x, y, w, h = bbox
		x1, y1 = x + w, y + h
		# Top left
		cv2.line(img, (x,y), (x+length,y), (255,0,255), thick)
		cv2.line(img, (x,y), (x,y+length), (255,0,255), thick)
		# Top right
		cv2.line(img, (x1,y), (x1-length,y), (255,0,255), thick)
		cv2.line(img, (x1,y), (x1,y+length), (255,0,255), thick)
		# Bottom left
		cv2.line(img, (x,y1), (x+length,y1), (255,0,255), thick)
		cv2.line(img, (x,y1), (x,y1-length), (255,0,255), thick)
		# Bottom right
		cv2.line(img, (x1,y1), (x1-length,y1), (255,0,255), thick)
		cv2.line(img, (x1,y1), (x1,y1-length), (255,0,255), thick)
		# Rectangle
		cv2.rectangle(img, bbox, (255,0,255), rect_thick)
		return img


def main():
	cap = cv2.VideoCapture(0) # 1 ?
	detector = FaceDetector()
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


if __name__ == '__main__':
	main()
