import cv2 as cv, time, pandas
from datetime import datetime

first_frame = None
status_list = [None,None]
times = []
df = pandas.DataFrame(columns = ["Start", "End"])

video = cv.VideoCapture(0) #create videocapture object to record video

while True:

	check, frame = video.read()

	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #convert frame color to grayscale
	gray = cv.GaussianBlur(gray, (21, 21),0) #convert to gaussian blur

	if first_frame is None:		#storing first image/frame of the video
		first_frame = gray
		continue

delta_frame = cv.absdiff(first_frame, gray)	#calculate diff 1st frame and other frame
thresh_delta = cv.threshold(delta_frame, 30, 255, cv.THRESH.BINARY)[1]	#provide threshold value that will convert the diff value that less than 30 to black. if > than 30 convert to white
thresh_delta = cv.dilate(thresh_delta, None, iterations=0)
(_,cnts,_) = cv. find.Contours(thresh_delta.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)	#define contour area

for contour in cnts:
	if cv.contourArea(contour) < 1000:	#remove noise
		continue
	(x, y, w, h) = cv.boundingRect(contour)	#create rectangle box
	cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

cv.imshow('frame', frame)
cv.imshow('Capturing', gray)
cv.imshow('delta', delta_frame)
cv.imshow('thresh', thresh_delta)
