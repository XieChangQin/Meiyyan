import cv2
import numpy as np
from scipy.spatial import distance
import numpy as np
import dlib
import math
import sys


from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap,QImage
from 美颜 import Ui_MainWindow


class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        self.label.resize(800,800)
        self.label_2.resize(800,800)
        # 信号与槽
        # 打开文件
        self.pushButton.clicked.connect(self.open_picture_618)
        # 美白
        self.pushButton_2.clicked.connect(self.MB_618)
        self.horizontalSlider.valueChanged.connect(self.valueChanged_618)
        # 磨皮
        self.pushButton_3.clicked.connect(self.MP_618)
        self.horizontalSlider_2.valueChanged.connect(self.valueChanged1_618)
        #瘦脸
        self.pushButton_4.clicked.connect(self.SL_530)
        #左脸滑动条
        self.horizontalSlider_3.valueChanged.connect(self.valueChanged2_618)
        #右脸滑动条
        self.horizontalSlider_4.valueChanged.connect(self.valueChanged3_618)
        #红唇
        self.pushButton_5.clicked.connect(self.HC_530)
        #蓝色滑动条
        self.horizontalSlider_6.valueChanged.connect(self.valueChanged5_618)
        #绿色滑动条
        self.horizontalSlider_5.valueChanged.connect(self.valueChanged4_618)
        #红色滑动条
        self.horizontalSlider_7.valueChanged.connect(self.valueChanged6_618)
        #大眼
        self.pushButton_6.clicked.connect(self.DY_530)
        #半径滑动条
        self.horizontalSlider_8.valueChanged.connect(self.valueChanged7_618)
        #强度滑动条
        self.horizontalSlider_9.valueChanged.connect(self.valueChanged8_618)
        #打开摄像头
        self.pushButton_7.clicked.connect(self.open_video_618)

    # 将opencv的图片转换为QImage
    def cv2qimg_618(self, cvImg):
        # 获取图片的宽、高和通道数
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        return QImage(cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB).data, width, height, bytesPerLine, QImage.Format_RGB888)

    #打开文件
    def open_picture_618(self):
        global imagepath
        # 获取图片路径
        imagepath, imgtype = QFileDialog.getOpenFileName(self.centralwidget, '打开', 'D://pythonProject3','*.jpg;;*.png;;All Files(*)')
        img = QPixmap(imagepath)
        # 将图片显示到label框中
        self.label.setPixmap(img)
        self.label_2.setPixmap(img)
        # 自适应label框
        self.label.setScaledContents(True)
        self.label_2.setScaledContents(True)
    #美白滑动条值改变
    def valueChanged_618(self):
        size = self.horizontalSlider.sliderPosition()
        self.label_3.setText(str(size))
        return size
    #美白函数
    def meibai_618(self,img, d):
        height = img.shape[0]
        width = img.shape[1]
        img1 = np.zeros((height, width, 3), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                (b, g, r) = img[i, j]
                b1 = int(b) + d
                g1 = int(g) + d
                r1 = int(r) + d
                if b1 > 255:
                    b1 = 255
                if g1 > 255:
                    g1 = 255
                if r1 > 255:
                    r1 = 255
                img1[i, j] = (b1, g1, r1)
        self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img1)))
        self.label_2.setScaledContents(True)
    #美白
    def MB_618(self):
        # global imagepath
        # img = cv2.imdecode(np.fromfile(imagepath, dtype=np.uint8), -1)
        img = self.label_2.pixmap().toImage()
        img = self.qimagecv2_618(img)
        while True:
            d = self.valueChanged_618()
            self.meibai_618(img,d)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    #磨皮滑动条值改变
    def valueChanged1_618(self):
        size = self.horizontalSlider_2.sliderPosition()
        self.label_4.setText(str(size))
        return size
    #将QImage图片转换为RGB图片
    def qimagecv2_618(self, img):
        ptr = img.constBits()
        ptr.setsize(img.byteCount())
        # 注意这地方通道数一定要填4，否则出错
        mat = np.array(ptr).reshape(img.height(), img.width(), 4)
        img1 = cv2.cvtColor(mat,cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        return img2

    #磨皮
    def MP_618(self):
        img = self.label_2.pixmap().toImage()
        img = self.qimagecv2_618(img)
        while True:
            p = self.valueChanged1_618()
            img1 = cv2.bilateralFilter(img, p, p*2.5, p*2.5)
            self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img1)))
            self.label_2.setScaledContents(True)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    #左脸滑条值
    def valueChanged2_618(self):
        size = self.horizontalSlider_3.sliderPosition()
        self.label_5.setText(str(size))
        return size
    #右脸滑条值
    def valueChanged3_618(self):
        size = self.horizontalSlider_4.sliderPosition()
        self.label_6.setText(str(size))
        return size
    #双线性插值法
    def bilinear_insert_530(self,image, new_x, new_y):
        """
        双线性插值法
        """
        w, h, c = image.shape
        if c == 3:
            x1 = int(new_x)
            x2 = x1 + 1
            y1 = int(new_y)
            y2 = y1 + 1

            part1 = image[y1, x1].astype(float) * (float(x2) - new_x) * (float(y2) - new_y)
            part2 = image[y1, x2].astype(float) * (new_x - float(x1)) * (float(y2) - new_y)
            part3 = image[y2, x1].astype(float) * (float(x2) - new_x) * (new_y - float(y1))
            part4 = image[y2, x2].astype(float) * (new_x - float(x1)) * (new_y - float(y1))

            insertValue = part1 + part2 + part3 + part4

            return insertValue.astype(np.int8)
    #局部平移算法
    def local_traslation_warp_530(self,image, start_point, end_point, radius):
        """
        局部平移算法
        """
        radius_square = math.pow(radius, 2)
        image_cp = image.copy()

        dist_se = math.pow(np.linalg.norm(end_point - start_point), 2)
        height, width, channel = image.shape
        for i in range(width):
            for j in range(height):
                # 计算该点是否在形变圆的范围之内
                # 优化，第一步，直接判断是会在（start_point[0], start_point[1])的矩阵框中
                if math.fabs(i - start_point[0]) > radius and math.fabs(j - start_point[1]) > radius:
                    continue

                distance = (i - start_point[0]) * (i - start_point[0]) + (j - start_point[1]) * (j - start_point[1])

                if (distance < radius_square):
                    # 计算出（i,j）坐标的原坐标
                    # 计算公式中右边平方号里的部分
                    ratio = (radius_square - distance) / (radius_square - distance + dist_se)
                    ratio = ratio * ratio

                    # 映射原位置
                    new_x = i - ratio * (end_point[0] - start_point[0])
                    new_y = j - ratio * (end_point[1] - start_point[1])

                    new_x = new_x if new_x >= 0 else 0
                    new_x = new_x if new_x < height - 1 else height - 2
                    new_y = new_y if new_y >= 0 else 0
                    new_y = new_y if new_y < width - 1 else width - 2

                    # 根据双线性插值法得到new_x, new_y的值
                    image_cp[j, i] = self.bilinear_insert_530(image, new_x, new_y)
        return image_cp
    #瘦脸
    def SL_530(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('D:/pythonProject3/venv/Scripts/shape_predictor_68_face_landmarks.dat')
        img = self.label_2.pixmap().toImage()
        img = self.qimagecv2_618(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        while True:
            # 检测人脸
            rects = detector(gray, 0)
            # 遍历每一个检测到的人脸
            for rect in rects:
                # 获取坐标
                result = predictor(gray, rect)
                result=result.parts()
                points = [[p.x, p.y] for p in result]
                points = np.array(points)#这里需将上面的points转化为数组
                end_point = points[29]  # 30号点
                # 瘦左脸，计算3号点到5号点的距离作为瘦脸最大参考距离
                dist_left = distance.euclidean(points[3], points[5])
                dist_left = self.valueChanged2_618()
                out_put = self.local_traslation_warp_530(img, points[3], end_point, dist_left)
                # 瘦右脸，计算13号点到15号点的距离作为瘦脸的最大参考距离
                dist_right = distance.euclidean(points[13], points[15])
                dist_right = self.valueChanged3_618()
                out_put = self.local_traslation_warp_530(out_put, points[14], end_point, dist_right)
                self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(out_put)))
                self.label_2.setScaledContents(True)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    #蓝色滑条值
    def valueChanged5_618(self):
        size = self.horizontalSlider_6.sliderPosition()
        self.label_7.setText(str(size))
        return size

    # 绿色滑条值
    def valueChanged4_618(self):
        size = self.horizontalSlider_5.sliderPosition()
        self.label_14.setText(str(size))
        return size

    # 红色滑条值
    def valueChanged6_618(self):
        size = self.horizontalSlider_7.sliderPosition()
        self.label_16.setText(str(size))
        return size
    #红唇函数
    def red_lip_530(self,img, gray,points1, points2, color1, color2, color3):
        hull1 = cv2.convexHull(points1)  # 先计算嘴唇凸包（即获取嘴唇轮廓）
        hull2 = cv2.convexHull(points2)  # 先计算牙齿凸包（即获取牙齿轮廓）
        #  为了不影响图片效果，因为后面是将两张图片加权叠加，所以下面先生成和原图大小相同的图片，并在上面绘制嘴唇轮廓
        mask1 = np.zeros_like(gray)
        mask2 = np.zeros_like(gray)

        cv2.fillPoly(mask1, np.array([hull1]), 255)  # 填充多边形
        roi = cv2.bitwise_and(gray, gray, mask=mask1)  # 获取掩膜
        roi = np.stack((roi,) * 3, axis=-1)  # 在第二维操作
        # roi1 = cv2.drawContours(roi, [hull1], -1, (0, 0, color), -1)
        roi1 = cv2.drawContours(roi, [hull1], -1, (color1, color2, color3), -1)
        cv2.fillPoly(mask2, np.array([hull2]), 255)  # 填充多边形
        roi = cv2.bitwise_and(gray, gray, mask=mask2)  # 获取掩膜
        roi = np.stack((roi,) * 3, axis=-1)  # 在第二维操作
        # roi2 = cv2.drawContours(roi, [hull2], -1, (0, 0, color), -1)
        roi2 = cv2.drawContours(roi, [hull2], -1, (color1, color2, color3), -1)
        roi = roi1 - roi2
        image = cv2.addWeighted(roi, 0.5, img.copy(), 0.9, 0)
        return image

    #红唇
    def HC_530(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('D:/pythonProject3/venv/Scripts/shape_predictor_68_face_landmarks.dat')
        img = self.label_2.pixmap().toImage()
        img = self.qimagecv2_618(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        while True:
            # 检测人脸
            rects = detector(gray, 0)
            # 遍历每一个检测到的人脸
            for rect in rects:
                # 获取坐标
                result = predictor(gray, rect)  # 获取68个特征点坐标
                result = result.parts()  # 一个个分开
                points = [[p.x, p.y] for p in result]
                points = np.array(points)  # 这里需将上面的points转化为数组
                # 嘴唇变红
                color1 = self.valueChanged5_618()
                color2 = self.valueChanged4_618()
                color3 = self.valueChanged6_618()
                p1 = points[48:60]  # 嘴唇点集分布
                p2 = points[60:68]  # 牙齿点分布
                img2 = self.red_lip_530(img, gray, p1, p2, color1, color2, color3)
                self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img2)))
                self.label_2.setScaledContents(True)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # 半径滑条值
    def valueChanged7_618(self):
        size = self.horizontalSlider_8.sliderPosition()
        self.label_17.setText(str(size))
        return size
    #强度滑条值
    def valueChanged8_618(self):
        size = self.horizontalSlider_9.sliderPosition()
        self.label_18.setText(str(size))
        return size
    #图像局部缩放算法
    def local_zoom_warp_530(self,image, point, radius, strength):
        """
        图像局部缩放算法
        """
        height = image.shape[0]
        width = image.shape[1]
        left = int(point[0] - radius) if point[0] - radius >= 0 else 0
        top = int(point[1] - radius) if point[1] - radius >= 0 else 0
        right = int(point[0] + radius) if point[0] + radius < width else width - 1
        bottom = int(point[1] + radius) if point[1] + radius < height else height - 1

        radius_square = math.pow(radius, 2)
        for y in range(top, bottom):
            offset_y = y - point[1]
            for x in range(left, right):
                offset_x = x - point[0]
                dist_xy = offset_x * offset_x + offset_y * offset_y

                if dist_xy <= radius_square:
                    scale = 1 - dist_xy / radius_square
                    scale = 1 - strength / 100 * scale
                    new_x = offset_x * scale + point[0]
                    new_y = offset_y * scale + point[1]
                    new_x = new_x if new_x >= 0 else 0
                    new_x = new_x if new_x < height - 1 else height - 2
                    new_y = new_y if new_y >= 0 else 0
                    new_y = new_y if new_y < width - 1 else width - 2

                    image[y, x] = self.bilinear_insert_530(image, new_x, new_y)
    #大眼
    def DY_530(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('D:/pythonProject3/venv/Scripts/shape_predictor_68_face_landmarks.dat')
        img = self.label_2.pixmap().toImage()
        img = self.qimagecv2_618(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        while True:
            img1 = img.copy()
            img2 = img.copy()
            # 检测人脸
            rects = detector(gray, 0)
            # 遍历每一个检测到的人脸
            for rect in rects:
                # 获取坐标
                result = predictor(gray, rect)
                result = result.parts()
                points = [[p.x, p.y] for p in result]
                points = np.array(points)  # 这里需将上面的points转化为数组
                # for (x, y) in points[:]:
                #     cv2.circle(img1, (x, y), 3, (0, 255, 0), -1)  # 还是原谅色
                # 以左眼最左点和最右点之间的中点为圆心,43,46
                left_eye_top = points[43]
                left_eye_bottom = points[46]
                left_eye_center = (left_eye_top + left_eye_bottom) / 2
                # 以右眼最左点和最右点之间的中点为圆心,37,40
                right_eye_top = points[37]
                right_eye_bottom = points[40]
                right_eye_center = (right_eye_top + right_eye_bottom) / 2

                # 放大双眼
                radius = self.valueChanged7_618()
                strength = self.valueChanged8_618()
                self.local_zoom_warp_530(img2, left_eye_center, radius=radius, strength=strength)
                self.local_zoom_warp_530(img2, right_eye_center, radius=radius, strength=strength)
                # radius: 眼睛放大范围半径 strength：眼睛放大程度
                self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img2)))
                self.label_2.setScaledContents(True)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    #打开摄像头
    def open_video_618(self):
        f = cv2.VideoCapture(0)
        while True:
            ret,frame = f.read()
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('D:/pythonProject3/venv/Scripts/shape_predictor_68_face_landmarks.dat')
            if not ret:
                break
            self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(frame)))
            self.label_2.setScaledContents(True)
            #美白
            d = self.valueChanged_618()
            self.meibai_618(frame,d)
            #磨皮
            img = self.label_2.pixmap().toImage()
            img = self.qimagecv2_618(img)
            p = self.valueChanged1_618()
            img1 = cv2.bilateralFilter(img, p, p * 2.5, p * 2.5)
            self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img1)))
            self.label_2.setScaledContents(True)
            # #瘦脸
            # imag = self.label_2.pixmap().toImage()
            # imag = self.qimagecv2_618(imag)
            # img3 = imag.copy()
            # gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
            # # 检测人脸
            # rects = detector(gray, 0)
            # # 遍历每一个检测到的人脸
            # for rect in rects:
            #     # 获取坐标
            #     result = predictor(gray, rect)
            #     result = result.parts()
            #     points = [[p.x, p.y] for p in result]
            #     points = np.array(points)  # 这里需将上面的points转化为数组
            #     end_point = points[29]  # 30号点
            #     # 瘦左脸，计算3号点到5号点的距离作为瘦脸最大参考距离
            #     dist_left = distance.euclidean(points[3], points[5])
            #     dist_left = self.valueChanged2_618()
            #     out_put = self.local_traslation_warp_530(imag, points[3], end_point, dist_left)
            #     # 瘦右脸，计算13号点到15号点的距离作为瘦脸的最大参考距离
            #     dist_right = distance.euclidean(points[13], points[15])
            #     dist_right = self.valueChanged3_618()
            #     out_put = self.local_traslation_warp_530(out_put, points[14], end_point, dist_right)
            #     self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(out_put)))
            #     self.label_2.setScaledContents(True)
                # # 嘴唇变红
                # color1 = self.valueChanged5_618()
                # color2 = self.valueChanged4_618()
                # color3 = self.valueChanged6_618()
                # p1 = points[48:60]  # 嘴唇点集分布
                # p2 = points[60:68]  # 牙齿点分布
                # img2 = self.red_lip_530(imag, gray, p1, p2, color1, color2, color3)
                # self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img2)))
                # self.label_2.setScaledContents(True)
                #
                # # 以左眼最左点和最右点之间的中点为圆心,43,46
                # left_eye_top = points[43]
                # left_eye_bottom = points[46]
                # left_eye_center = (left_eye_top + left_eye_bottom) / 2
                # # 以右眼最左点和最右点之间的中点为圆心,37,40
                # right_eye_top = points[37]
                # right_eye_bottom = points[40]
                # right_eye_center = (right_eye_top + right_eye_bottom) / 2
                # # 放大双眼
                # radius = self.valueChanged7_618()
                # strength = self.valueChanged8_618()
                # self.local_zoom_warp_530(img3, left_eye_center, radius=radius, strength=strength)
                # self.local_zoom_warp_530(img3, right_eye_center, radius=radius, strength=strength)
                # # radius: 眼睛放大范围半径 strength：眼睛放大程度
                # self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img3)))
                # self.label_2.setScaledContents(True)

            # #红唇
            # imag = self.label_2.pixmap().toImage()
            # imag = self.qimagecv2_618(imag)
            # gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
            # # 检测人脸
            # rects = detector(gray, 0)
            # # 遍历每一个检测到的人脸
            # for rect in rects:
            #      # 获取坐标
            #     result = predictor(gray, rect)  # 获取68个特征点坐标
            #     result = result.parts()  # 一个个分开
            #     points = [[p.x, p.y] for p in result]
            #     points = np.array(points)  # 这里需将上面的points转化为数组
            #     # 嘴唇变红
            #     color1 = self.valueChanged5_618()
            #     color2 = self.valueChanged4_618()
            #     color3 = self.valueChanged6_618()
            #     p1 = points[48:60]  # 嘴唇点集分布
            #     p2 = points[60:68]  # 牙齿点分布
            #     img2 = self.red_lip_530(img, gray, p1, p2, color1, color2, color3)
            #     self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img2)))
            #     self.label_2.setScaledContents(True)
            # #大眼
            # img = self.label_2.pixmap().toImage()
            # img = self.qimagecv2_618(img)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img1 = img.copy()
            # img2 = img.copy()
            # # 检测人脸
            # rects = detector(gray, 0)
            # # 遍历每一个检测到的人脸
            # for rect in rects:
            #     # 获取坐标
            #     result = predictor(gray, rect)
            #     result = result.parts()
            #     points = [[p.x, p.y] for p in result]
            #     points = np.array(points)  # 这里需将上面的points转化为数组
            #     # 以左眼最左点和最右点之间的中点为圆心,43,46
            #     left_eye_top = points[43]
            #     left_eye_bottom = points[46]
            #     left_eye_center = (left_eye_top + left_eye_bottom) / 2
            #     # 以右眼最左点和最右点之间的中点为圆心,37,40
            #     right_eye_top = points[37]
            #     right_eye_bottom = points[40]
            #     right_eye_center = (right_eye_top + right_eye_bottom) / 2
            #     # 放大双眼
            #     radius = self.valueChanged7_618()
            #     strength = self.valueChanged8_618()
            #     self.local_zoom_warp_530(img2, left_eye_center, radius=radius, strength=strength)
            #     self.local_zoom_warp_530(img2, right_eye_center, radius=radius, strength=strength)
            #     # radius: 眼睛放大范围半径 strength：眼睛放大程度
            #     self.label_2.setPixmap(QPixmap.fromImage(self.cv2qimg_618(img2)))
            #     self.label_2.setScaledContents(True)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break






if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
