import numpy as np
import cv2
import time

path = "haarcascade_frontalface_default.xml"
path2 = "haarcascade_eye.xml"



face_cascade = cv2.CascadeClassifier(path)
eye_cascade = cv2.CascadeClassifier(path2)
cap=cv2.VideoCapture(0)
while True:
	ret,img=cap.read()
	if not ret:
		break
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))
	eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.10,minNeighbors=10,minSize=(10,10))
	

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
	for (x, y, w, h) in eyes:
		xc = (x + x+w)/2
		yc = (y + y+h)/2
		radius = w/2
		cv2.circle(img, (int(xc),int(yc)), int(radius), (0,255,0), 2)
	cv2.imshow("Image",img)

	time.sleep(0.005)

	ch=cv2.waitKey(1)
	if ch & 0xFF==ord('q'):
		break
cap.release()
cv2.destroyAllWindows()