import cv2
import numpy as np

blue = (0, 0, 255)
red=(255,0,0)
green=(0,255,0)
blue=(0,0,255)
yellow=(255, 255, 0)

width = 20
height = 20
img=np.zeros((width,height,3)).astype(np.uint8)
img[10:12,4:8,:]=red
img[10:12,12:16,:]=red
img[5:7,2:4,:]=red
img[5:7,8:10,:]=red
img[5:7,14:16,:]=red
img[18:20,0:2,:]=blue
img[2:4,4:6,:]=green
img[16:18,16:18,:]=green
img[14:16,2:10,:]=yellow
r_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.imwrite('temp.png',r_img)

img1=cv2.imread("temp.png")

#pg.draw.rect(win, red, pg.Rect(10, 7, 4,20 ))
#pg.draw.rect(win, blue, pg.Rect(2, 2, 4,4 ))
#pg.draw.rect(win, blue, pg.Rect(25, 25, 4,4 ))
#pg.draw.rect(win, green, pg.Rect(4, 4, 4,4 ))
#pg.draw.rect(win, yellow, pg.Rect(20, 20, 3,3 ))



