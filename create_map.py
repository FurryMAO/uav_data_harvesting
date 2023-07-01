import cv2
import numpy as np
# red #ff0000 no-fly zone (NFZ)
# green #00ff00 buildings blocking wireless links (UAVs can fly over)
# blue #0000ff start and landing zone
# yellow #ffff00 buildings blocking wireless links + NFZ (UAVs can not fly over)
if __name__ == '__main__':
    blue = (0, 0, 255)
    red=(255,0,0)
    green=(0,255,0)
    yellow=(255, 255, 0)

    width = 15
    height = 15
    img=np.zeros((width,height,3)).astype(np.uint8)
    img[14:15,6:8,:]=blue
    #draw red
    square_size = 2
    square_positions_red = [(4, 4), (8, 10), (8,2)]
    for position in square_positions_red:
        row, col = position
        img[row:row+square_size, col:col+square_size, :] = red


    #draw green
    square_positions_green = [(0, 0), (12, 2), (8, 6)]
    for position in square_positions_green:
        row, col = position
        img[row:row + square_size, col:col + square_size, :] = green

        # draw yellow
    square_positions_yellow = [(5, 0), (2, 10), (12, 10)]
    for position in square_positions_yellow:
        row, col = position
        img[row:row + square_size, col:col + square_size, :] = yellow



    # img[10:13,12:16,:]=yellow
    # img[7:10,2:4,:]=green
    # img[4:5,11:13,:]=red
    # img[18:22,14:17,:]=yellow
    # img[19:22,0:3,:]=blue
    # img[0:4,4:7,:]=yellow
    # img[16:18,19:22,:]=green
    # img[0:4,16:20,:]=green
    # img[14:16,2:10,:]=green
    r_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    cv2.imwrite('temp.png',r_img)

    img1=cv2.imread("temp.png")

#pg.draw.rect(win, red, pg.Rect(10, 7, 4,20 ))
#pg.draw.rect(win, blue, pg.Rect(2, 2, 4,4 ))
#pg.draw.rect(win, blue, pg.Rect(25, 25, 4,4 ))
#pg.draw.rect(win, green, pg.Rect(4, 4, 4,4 ))
#pg.draw.rect(win, yellow, pg.Rect(20, 20, 3,3 ))



