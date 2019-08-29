import cv2

#Create a CascadeClassifier Object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Reading the image
img = cv2.imread("/home/alia/Downloads/download.jpg")

#Read the image as gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Search teh coordinates of image
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)

print(type(faces))
print(faces)

#to make the rectangle on the face detected on the image
for x, y, w, h in faces:
	#rectangle(yourimage, coordinate, coordinate, code color box's border, width of rectange)
	img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0),8)

resized = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
cv2.imshow("Gray", resized)

cv2.waitKey(2000)
cv2.destroyAllWindows()
