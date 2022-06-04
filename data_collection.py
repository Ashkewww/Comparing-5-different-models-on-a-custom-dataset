from turtle import color

from matplotlib.pyplot import gray
import frame
import cv2

#list_of_images = frame.get_frame(1)

#for j in range(len(list_of_images)):
image = cv2.imread(f'data/test images/me/Picture 15.jpg')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face = face_cascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in face:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    face = image[y:y + h,x:x + w]
    cv2.imshow('face',face)
cv2.imshow('img',image)
cv2.waitKey()
    
    
    
    
    
    # cv2.imwrite("data\\train images\\Picture "+str(j)+".jpg", list_of_images[j])