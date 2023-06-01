import cv2
import numpy as np
from PIL import Image,ImageEnhance

def nothing(x):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",600,150)
cv2.createTrackbar("d","TrackBars",0,255,nothing)


def face_dermabrasion(img):
    while True:
        d = cv2.getTrackbarPos("d","TrackBars")
    #双边滤波
        bl_img = cv2.bilateralFilter(img,d,d*2.5,d*2.5)
    #图像融合
        rt_img = cv2.addWeighted(img,0.3,bl_img,0.7,0.45)
        image = Image.fromarray(cv2.cvtColor(rt_img, cv2.COLOR_BGR2RGB))
    #锐度调节
        eh_img = ImageEnhance.Sharpness(image)
        image_sharped = eh_img.enhance(1.5)
    #对比度调节
        c_img = ImageEnhance.Contrast(image_sharped)
        image_c = c_img.enhance(1.15)
        image_c.save('58_2.jpg')
        img2 = cv2.imread('58_2.jpg')
        cv2.imshow('2', cv2.resize(img2,(500,600)))
        cv2.waitKey(1)


#图片
img1 = cv2.imread('picture5.jpg')
cv2.imshow('1',cv2.resize(img1,(500,600)))
face_dermabrasion(img1)
cv2.destroyAllWindows()
#摄像头
# f = cv2.VideoCapture(0)
# while True:
#     ret,frame = f.read()
#     if not ret:
#         break
#     d = cv2.getTrackbarPos("d", "TrackBars")
#     # sC = cv2.getTrackbarPos("sigmaColor", "TrackBars")
#     # sS = cv2.getTrackbarPos("sigmaSpace", "TrackBars")
#     bl_img = cv2.bilateralFilter(frame, d, d*2.5, d*2.5)
#     cv2.imshow('result',bl_img)
#
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

# img2 = cv2.imread('58_2.jpg')
# img3 = filter_gaussian(img1, 3, 9)
# img3 = cv2.imread('58_3.jpg')
# cv2.imshow('2',img2)
# cv2.imshow('3',img3)
# cv2.waitKey()
# cv2.destroyAllWindows()