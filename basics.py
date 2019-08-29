import cv2
img = cv2.imread ("/home/alia/Downloads/download.jpg", 0)

resized_image = cv2.resize(img, (int(img.shape[1]/7),int(img.shape[0]/7)))

cv2.imshow("Me", resized_image)

cv2.waitKey(500)

#print(img)

cv2.destroyAllWindows()
