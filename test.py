import cv2  # 本来想用mpl的，但是cv2功能很强大，就用cv2了
import matplotlib.pyplot as plt
import numpy as np
import math

img = np.zeros((64, 64, 3), np.uint8)  # 生成一个空彩色图像


class Node():  # 构造结点类
    num_nodes = 0

    def __init__(self, name, postion1x, postion1y, width, height):
        self.name = name  # 名字
        self.position1x = postion1x  # 中心的水平方向坐标
        self.position1y = postion1y  # 垂直方向
        self.width = width  # 宽度
        self.height = height  # 高度

    def movement(obj):
        return np.sqrt(np.sum((obj.position1 - obj.position2) ** 2))  # 移动距离

    def displayNode(self):
        cv2.rectangle(img, (int(self.position1x - self.width / 2), int(self.position1y - self.height / 2)),
                      (int(self.position1x + self.width / 2), int(self.position1y + self.height / 2)), (55, 255, 155),
                      1)  # 给出左上角端点和右下角端点就可以画出矩形了
# 用cv2绘制出矩形

node1 = Node("A", 20, 45, 30, 15)
node1.displayNode()
node2 = Node("B", 35, 40, 20, 45)
node2.displayNode()
node3 = Node("C", 40, 25, 20, 10)
node3.displayNode()
node4 = Node("D", 55, 57, 20, 10)
node4.displayNode()
img = np.uint8(img)
img = cv2.resize(img, dsize=None, fx=5, fy=5)
cv2.imshow('brg', img)  # 把画布展示出来

k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('new.jpg', img)
    cv2.destroyAllWindows()


print(math.degrees(np.arctan(-1)))