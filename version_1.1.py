import numpy as np
import cv2 
from imgTrat import *


def load_image():
    img = cv2.imread('imgs/imagem1.jpeg', cv2.IMREAD_COLOR)
    imTrat = imgTrat(img)
    img = imTrat.img_ProjTransformation(img)
    #cv2.imshow("test", img)
    #cv2.waitKey(0)
    return img

img = load_image()


elemento = elementos(img)

posicao_ball = elemento.get_Ball_pos()
player_red_pose = elemento.get_Player_pos(img,0)
print("Bola: ",posicao_ball)
print("Player: ",player_red_pose)

cv2.circle(img, posicao_ball, 7, (0, 0, 255), 2)
cv2.circle(img, player_red_pose, 7, (0, 0, 255), 2)
cv2.imshow("test", img)
cv2.waitKey(0)