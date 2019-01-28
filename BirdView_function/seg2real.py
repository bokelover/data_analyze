import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

## Debug
imagename = "./image/000562.png"
img_src = cv.imread(imagename, 1)
gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)

circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                          param1=1, param2=10, minRadius=0, maxRadius=100)
circles = np.uint16(np.around(circles))
print(circles.shape[1])
plotter = np.zeros((2, circles.shape[1]+1))
plotter[0, 0] = 0
plotter[1, 0] = 0
count = 1

## Circle_Detect
for i in circles[0, :]:

    cv.circle(img_src, (i[0], i[1]), i[2], (0, 255, 0), 1)
    cv.circle(img_src, (i[0], i[1]), 1,(0,255,255),2)
    y = 2.0 / (np.tan(np.radians((240 - i[1]) * -0.1 -1.0))) #[m]
    x = y * np.tan(np.radians((320 -i[0]) * 0.094)) #[m]
    plotter[0, count] = x
    plotter[1, count] = y
    print(x, y, np.sqrt(pow(x - plotter[0,count-1],2) + pow(y - plotter[1, count-1] , 2)))
    count += 1

#gray = cv.cvtColor(psp_img, cv.COLOR_BGR2GRAY)
cv.imshow("result", img_src)
cv.waitKey(0)
cv.destroyAllWindows()

plt.scatter(plotter[0, :],plotter[1,:])
x = [0, 5, 10, 15, 20, 30, 35, 40, 45, 50]

# 表示処理　
plt.plot(np.poly1d(np.polyfit(plotter[1,:], plotter[0,:], 2))(x), x)
plt.xlim(30, -30)
plt.ylim(-5, 50)
plt.gca().set_aspect('equal', adjustable='box')
plt.hlines([0], 30, -30, "black", linestyles='dashed')
plt.vlines([0], 50,  -5, "black", linestyles='dashed')
plt.show()