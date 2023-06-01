import cv2
import numpy as np
from PIL import Image,ImageEnhance

def nothing(x):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",600,100)
cv2.createTrackbar("p","TrackBars",0,255,nothing)

def MeiBai(img,d):
    height = img.shape[0]
    width = img.shape[1]
    img1 = np.zeros((height,width,3),np.uint8)
    for i in range(0,height):
        for j in range(0,width):
            (b,g,r) = img[i,j]
            b1 = int(b)+d
            g1 = int(g)+d
            r1 = int(r)+d
            if b1>255:
                b1 = 255
            if g1>255:
                g1 = 255
            if r1>255:
                r1 = 255
            img1[i,j] = (b1,g1,r1)
    cv2.imshow('2',cv2.resize(img1,(600,600)))


#读取图片
f = cv2.imread('picture6.jpg')
cv2.imshow('1',cv2.resize(f,(600,600)))
while True:
    p = cv2.getTrackbarPos("p","TrackBars")
    MeiBai(f,p)
    cv2.waitKey(1)
cv2.destroyAllWindows()
#摄像头
# f = cv2.VideoCapture(0)
# while True:
#     ret,frame = f.read()
#     if not ret:
#         break
#     p = cv2.getTrackbarPos("p","TrackBars")
#     MeiBai(frame,p)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# face_meibai(f)
# img3 = cv2.imread('59_2.jpg')
# cv2.imshow('3',cv2.resize(img3,(600,600)))
# cv2.destroyAllWindows()