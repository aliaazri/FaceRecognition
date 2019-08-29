import cv2 as cv, time
import numpy as np
import argparse
import time
import imutils
from imutils.video import VideoStream

#the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#load model
net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])


#Create video capture object.
#Zero(0) is to trigger the camera/or can specify direct path to the video file/or use (1) if i want to use 1 external camera
video = cv.VideoCapture(0)
time.sleep(2.0)

a = 1 #as keypoint

#check: boolean return true if python able to read VideoCapture object
#frame: NumPy array, represent 1st image that video captures
while True:
	a = a + 1
	check, frame = video.read()
	print(frame)

	(h, w) = frame.shape[:2]
	blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (140.0, 177.0, 123.0))

	#pass the blob into network for detection and prediction
	net.setInput(blob)
	detections = net.forward()


		# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv.putText(frame, text, (startX, y),
			cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


	#convert into grayscale image
	#gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	cv.imshow('Capturing', frame)
	#waitKey(1) means generating new frame after 1 millisecond
	key = cv.waitKey(1)

	#press 'q' to exit
	if key == ord('q'):
		break

print(a) #print keypoint

#to add time delay for a seconds
#time.sleep(3)

#cv.imshow("Capture", frame)

#cv.waitKey(2000)

#release the camera for some milliseconds video.stop()
video.release()

cv.destroyAllWindows()
