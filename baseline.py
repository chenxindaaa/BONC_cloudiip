import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
import math


class Apparatus:
    def __init__(self, name):
        self.name = name
        self.angle = []
        self.src = cv.imread(name)


    def line_detect_possible_demo(self, image, center, tg):
        '''
        :param image: 二值图
        :param center: 圆心
        :param tg: 直线检测maxLineGap
        '''
        res = {} # 存放线段的斜率和信息
        edges = cv.Canny(image, 50, 150, apertureSize=7)
        cv.imshow("abcdefg", edges)
        lines = cv.HoughLinesP(edges, 1, np.pi/360, 13, minLineLength=20, maxLineGap=tg)
        for line in lines:
            x_1, y_1, x_2, y_2 = line[0]
            # 将坐标原点移动到圆心
            x1 = x_1 - center[0]
            y1 = center[1] - y_1
            x2 = x_2 - center[0]
            y2 = center[1] - y_2

            # 计算斜率
            if x2 - x1 == 0:
                k = float('inf')
            else:
                k = (y2-y1)/(x2-x1)
            d1 = np.sqrt(max(abs(x2), abs(x1)) ** 2 + (max(abs(y2), abs(y1))) ** 2)  # 线段长度
            d2 = np.sqrt(min(abs(x2), abs(x1)) ** 2 + (min(abs(y2), abs(y1))) ** 2)
            # 将长指针与短指针做标记
            if d1 < 155 and d1 > 148 and d2 > 115:
                res[k] = [1]
            elif d1 < 110 and d1 > 100 and d2 > 75:
                res[k] = [2]
            else:
                continue
            res[k].append(1) if (x2 + x1) /2 > 0 else res[k].append(0)  # 将14象限与23象限分离
            cv.line(self.src, (x1 + center[0], center[1] - y1), (x2 + center[0],  center[1] - y2), (255, 0, 0), 1)
            cv.imshow("line_detect-posssible_demo", self.src)


            # 计算线段中点的梯度来判断是指针的左侧线段还是右侧线段
            middle_x = int((x_1 + x_2) / 2)
            middle_y = int((y_1 + y_2) / 2)
            grad_mat = image[middle_y-5:middle_y+6, middle_x-5:middle_x+6]
            cv.imshow("grad_mat", grad_mat)
            grad_x = cv.Sobel(grad_mat, cv.CV_32F, 1, 0)
            grad_y = cv.Sobel(grad_mat, cv.CV_32F, 0, 1)
            gradx = np.max(grad_x) if np.max(grad_x) != 0 else np.min(grad_x)
            grady = np.max(grad_y) if np.max(grad_y) != 0 else np.min(grad_y)
            if ((gradx >=0 and grady >= 0) or (gradx <= 0 and grady >= 0)) and res[k][1] == 1:
                res[k].append(1)  # 右测
            elif ((gradx <= 0 and grady <= 0) or (gradx >= 0 and grady <= 0)) and res[k][1] == 0:
                res[k].append(1)
            else:
                res[k].append(0) # 左侧
        # 计算角度
        angle1 = [i for i in res if res[i][0] == 1]
        angle2 = [i for i in res if res[i][0] == 2]
       # 长指针
        a = np.arctan(angle1[0])
        b = np.arctan(angle1[1])
        if a * b < 0 and max(abs(a), abs(b)) > np.pi / 4:
           if a + b < 0:
               self.angle.append(math.degrees(-(a + b) / 2)) if res[angle1[1]][1] == 1 else self.angle.append(
                   math.degrees(-(a + b) / 2) + 180)
           else:
               self.angle.append(math.degrees(np.pi - (a + b) / 2)) if res[angle1[1]][1] == 1 else self.angle.append(
                   math.degrees(np.pi - (a + b) / 2) + 180)
        else:
            self.angle.append(math.degrees(np.pi / 2 - (a + b) / 2)) if res[angle1[1]][1] == 1 else self.angle.append(math.degrees(np.pi / 2 - (a + b) / 2) + 180)
        print('长指针读数：%f' % self.angle[0])
        print(math.degrees(a - b))


        # 短指针
        a = np.arctan(angle2[0])
        b = np.arctan(angle2[1])
        if a * b < 0 and max(abs(a), abs(b)) > np.pi / 4:
           if a + b < 0:
               self.angle.append(math.degrees(-(a + b) / 2)) if res[angle2[1]][1] == 1 else self.angle.append(
                   math.degrees(-(a + b) / 2) + 180)
           else:
               self.angle.append(math.degrees(np.pi - (a + b) / 2)) if res[angle2[1]][1] == 1 else self.angle.append(
                   math.degrees(np.pi - (a + b) / 2) + 180)
        else:
            self.angle.append(math.degrees(np.pi / 2 - (a + b) / 2)) if res[angle2[1]][1] == 1 else self.angle.append(math.degrees(np.pi / 2 - (a + b) / 2) + 180)
        print('短指针读数：%f' % self.angle[1])
        print(math.degrees(a - b))



    def get_center(self, mask):
        edg_output = cv.Canny(mask, 66, 150, 2)
        cv.imshow('edg', edg_output)
        # 外接矩形
        contours, hireachy = cv.findContours(edg_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        center = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)  # 外接矩形
            area = w * h  # 面积
            if area > 1000 or area < 40:
                continue
            #print(area)
            # cv.circle(src, (np.int(cx), np.int(cy)), 3, (255), -1)
            cv.rectangle(self.src, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cx = w / 2
            cy = h / 2
            cv.circle(self.src, (np.int(x + cx), np.int(y + cy)), 1, (255, 0, 0))
            center.extend([np.int(x + cx), np.int(y + cy)])
            break

        # cv.imshow('final', src)
        return center


    def corner_detect(self, src, gray):
        dst = cv.cornerHarris(gray, 3, 27, 0.04)
        # print(dst.shape)  # (400, 600)
        src[dst > 0.01 * dst.max()] = [0, 255, 255]
        # cv.imshow('corner', src)

    def extract(self, image):
        red_lower1 = np.array([0, 43, 46])
        red_lower2 = np.array([156, 43, 46])
        red_upper1 = np.array([10, 255, 255])
        red_upper2 = np.array([180, 255, 255])
        frame = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask1 = cv.inRange(frame, lowerb=red_lower1, upperb=red_upper1)
        mask2 = cv.inRange(frame, lowerb=red_lower2, upperb=red_upper2)
        mask = cv.add(mask1, mask2)
        mask = cv.bitwise_not(mask)
        cv.imshow('mask', mask)
        return mask


    def test(self):
        # 缩小图片
        self.src = cv.resize(self.src, dsize=None, fx=0.5, fy=0.5)  # 此处可以修改插值方式interpolation
        cv.imshow('source', self.src)
        # 均值迁移
        blur = cv.pyrMeanShiftFiltering(self.src, 10, 17)
        # 提取红色
        mask = self.extract(blur)
        # 找圆心
        center = self.get_center(mask)
        try:
            self.line_detect_possible_demo(mask, center, 20)
        except IndexError:
            try:
                self.src = cv.imread(self.name)
                self.src = cv.resize(self.src, dsize=None, fx=0.5, fy=0.5)  # 此处可以修改插值方式interpolation
                self.src = cv.convertScaleAbs(self.src, alpha=1.4, beta=0)
                blur = cv.pyrMeanShiftFiltering(self.src, 10, 17)
                mask = self.extract(blur)
                self.line_detect_possible_demo(mask, center, 20)
            except IndexError:
                self.src = cv.imread(self.name)
                self.src = cv.resize(self.src, dsize=None, fx=0.5, fy=0.5)  # 此处可以修改插值方式interpolation
                self.src = cv.normalize(self.src, dst=None, alpha=200, beta=10, norm_type=cv.NORM_MINMAX)

                blur = cv.pyrMeanShiftFiltering(self.src, 10, 17)
                mask = self.extract(blur)
                self.line_detect_possible_demo(mask, center, 20)


if __name__ == '__main__':
    apparatus = Apparatus('./BONC cloudiip工业仪表表盘读数大赛/1_0340.jpg')
    # 读取图片
    apparatus.test()
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()
    elif k == ord('s'):
        cv.imwrite('new.jpg', apparatus.src)
        cv.destroyAllWindows()
