import cv2
import numpy as np

blue = (0, 0, 255)
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255, 255, 0)

width = 30
height = 30
img=np.zeros((width,height,3)).astype(np.uint8)
img[10:14,6:26,:]=red
img[14:18,24:26,:]=red
img[14:18,6:8,:]=red
img[2:6,22:24,:]=red
img[16:18,14:16,:]=blue
img[2:6,4:10,:]=green
img[24:26,24:28,:]=green
img[22:24,4:10,:]=yellow
r_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.imwrite('temp.png',r_img)

img1=cv2.imread("temp.png")

#pg.draw.rect(win, red, pg.Rect(10, 7, 4,20 ))
#pg.draw.rect(win, blue, pg.Rect(2, 2, 4,4 ))
#pg.draw.rect(win, blue, pg.Rect(25, 25, 4,4 ))
#pg.draw.rect(win, green, pg.Rect(4, 4, 4,4 ))
#pg.draw.rect(win, yellow, pg.Rect(20, 20, 3,3 ))



