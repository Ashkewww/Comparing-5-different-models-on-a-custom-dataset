import frame
import cv2

list_of_images = frame.get_frame(20)

for j in range(len(list_of_images)):
    cv2.imwrite("data\\Picture "+str(j)+".jpg", list_of_images[j])