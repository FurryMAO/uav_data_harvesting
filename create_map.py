import cv2
import numpy as np

blue = (0, 0, 255)
red=(255,0,0)
green=(0,255,0)
yellow=(255, 255, 0)

width = 22
height = 22
img=np.zeros((width,height,3)).astype(np.uint8)
img[10:13,12:16,:]=yellow
img[7:10,2:4,:]=green
img[4:5,11:13,:]=red
img[18:22,14:17,:]=yellow
img[19:22,0:3,:]=blue
img[0:4,4:7,:]=yellow
img[16:18,19:22,:]=green
img[0:4,16:20,:]=green
img[14:16,2:10,:]=green
r_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.imwrite('temp.png',r_img)

img1=cv2.imread("temp.png")

#pg.draw.rect(win, red, pg.Rect(10, 7, 4,20 ))
#pg.draw.rect(win, blue, pg.Rect(2, 2, 4,4 ))
#pg.draw.rect(win, blue, pg.Rect(25, 25, 4,4 ))
#pg.draw.rect(win, green, pg.Rect(4, 4, 4,4 ))
#pg.draw.rect(win, yellow, pg.Rect(20, 20, 3,3 ))



