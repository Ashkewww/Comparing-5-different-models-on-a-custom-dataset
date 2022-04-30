import cv2
import time 
def get_frame(r):
    temp = []
    feed = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    feed.set(1,500)
    feed.set(1,500)
    i = 0
    while i<r:
        success, frame = feed.read()
        temp.append(frame)
        cv2.imshow('',frame)
        i += 1
        time.sleep(3)
    
    return temp
    
    
    



