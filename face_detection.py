import cv2

img = cv2.imread('./data/txn.JPG')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier()

# 探测图片中的人脸

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.15,
    minNeighbors=5,
    minSize=(5, 5),
    flags=cv2.CASCADE_SCALE_IMAGE)


print ("发现{0}个人脸!".format(len(faces)))
for(x,y,w,h) in faces:
   cv2.rectangle(img,(x,y),(x+w,y+w),(0,255,0),2)

cv2.imwrite('rect_face.jpg',img)