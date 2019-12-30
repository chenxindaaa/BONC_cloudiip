import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
import math

def line_detect_possible_demo(image):
    res = []
    edges = cv.Canny(image, 50, 150, apertureSize=3)
    cv.imshow("abcdefg", edges)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 35, minLineLength=30, maxLineGap=10)
    for line in lines:
        print(type(line))
        x1, y1, x2, y2 = line[0]
        k = (y2-y1)/(x1-x2)
        cv.line(src, (x1, y1), (x2, y2), (255, 0, 0), 1)
        d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        print(d)
        res.append(k)
    cv.imshow("line_detect-posssible_demo", src)
    res.sort()
    for i in range(len(res)-1):
        if abs(abs(res[i])-abs(res[i+1]))< 0.02:
            res[i+1] = (res[i] + res[i+1])/2
            res[i] = 0
    while 0 in res:res.remove(0)
    angle = []
    for j in range(0, len(res), 2):
        a =np.pi/2-(np.arctan(res[j])+np.arctan(res[j+1]))/2
        angle.append(math.degrees(a))
    print(angle)




def watershed(image):
    dist = cv.distanceTransform(image, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    a = dist.max()
    b = dist_output.max()
    cv.imshow('distance_transform', dist_output * 1000)
    cv.imshow('dist', dist)

    #ret, surface = cv.threshold(dist, 2, 255, cv.THRESH_BINARY)
    cv.imshow('surface', dist)
    surface_fg = np.uint8(dist)
    unknown = cv.subtract(image, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)
    # watershed transform
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(src, markers=markers)
    src[markers == -1] = [0, 255, 0]
    cv.imshow('result', src)

def median_blur_demo(image):
    dst = cv.medianBlur(image, 1.5)
    cv.imshow("median_blur_demo", dst)
    return dst


def contrast_brightness_demo(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow('con-bri-demo', dst)

def image_hist(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

def clahe_demp(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow('clahe_demp', dst)

def equalHist_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow('equalHist_demo', dst)


# 读取图片
src = cv.imread('1_0107.jpg')
# 缩小图片
src = cv.resize(src, dsize=None, fx=0.5, fy=0.5)  # 此处可以修改插值方式interpolation
cv.imshow('source', src)



# 二值化
blur = cv.pyrMeanShiftFiltering(src, 10, 15) # 均值滤波
cv.imshow('blur', blur)
gray = blur[:, :, 2]
# gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

# 提取红色
red_lower1 = np.array([0, 43, 46])
red_lower2 = np.array([156, 43, 46])
red_upper1 = np.array([10, 255, 255])
red_upper2 = np.array([180, 255, 255])
frame = cv.cvtColor(blur,cv.COLOR_BGR2HSV)
cv.imshow('hsv', frame)
mask1 = cv.inRange(frame, lowerb=red_lower1, upperb=red_upper1)
mask2 = cv.inRange(frame, lowerb=red_lower2, upperb=red_upper2)
mask = cv.add(mask1, mask2)
mask = cv.bitwise_not(mask)
cv.imshow('mask', mask)
line_detect_possible_demo(mask)





ret, binary = cv.threshold(gray, 226, 255, cv.THRESH_BINARY_INV)
print(ret)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
#binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
#binary = cv.dilate(binary, kernel)
cv.imshow('binary', binary)

#image_hist(src)

# 边缘检测
edg_output = cv.Canny(mask, 66, 150, 2)
cv.imshow('edg', edg_output)
# 外接矩形
contours, hireachy = cv.findContours(edg_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


watershed(binary)



for i, contour in enumerate(contours):
    x, y, w, h = cv.boundingRect(contour) # 外接矩形
    area = w*h # 面积
    #cv.circle(src, (np.int(cx), np.int(cy)), 3, (255), -1)
    cv.rectangle(src, (x, y), (x+w, y+h), (255, 0, 0), 1)
    mm = cv.moments(contour)
    type(mm)
    cx = mm['m10'] / mm['m00']  # x0
    cy = mm['m01'] / mm['m00']  # y0
    #cv.circle(src, (np.int(cx), np.int(cy)), 1, (255))

cv.imshow('final', src)





k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
elif k == ord('s'):
    cv.imwrite('new.jpg', src)
    cv.destroyAllWindows()