import numpy as np
import cv2 
from imgTrat import *


def load_video():
    # Carrega a imagem do campo
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Não é possível abrir a câmera")
        exit()
    return cap 

def load_frame(cap):
    ok, frame = cap.read()
    assert frame is not None, "o arquivo não pôde ser lido, verifique com os.path.exists()"
    return frame 


def main():

    while True:
        cap = load_video()
        img = load_frame(cap)
        elemento = elementos(img)

        posicao_ball = elemento.get_Ball_pos()
        player_red_pose = elemento.get_Player_pos(img,0)
        print("Bola: ",posicao_ball)
        print("Player: ",player_red_pose)

        cv2.circle(img, posicao_ball, 7, (0, 0, 255), 2)
        cv2.circle(img, player_red_pose, 7, (0, 0, 255), 2)
        cv2.imshow("test", img)
        if cv.waitKey(1) == ord('q'):
            break


if __name__== "__main__":
    main()