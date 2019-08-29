import cv2 as cv, time

#Create video capture object.
#Zero(0) is to trigger the camera/or can specify direct path to the video file/or use (1) if i want to use 1 external camera
video = cv.VideoCapture(0)

#check: boolean return true if python able to read VideoCapture object
#frame: NumPy array, represent 1st image that video captures
check, frame = video.read()

print(check)
print(frame)

#to add time delay for a seconds
time.sleep(3)

cv.imshow("Capture", frame)

cv.waitKey(2000)

#release the camera for some milliseconds
video.release()

cv.destroyAllWindows()
