import numpy as np
import cv2 
from imgTrat import *

img = cv2.imread('imagem4.jpeg', cv2.IMREAD_COLOR)

# corrigindo projeção, 
# ajuste de escala do campo
# corte de áreas externas 
imTrat = imgTrat(img)
img = imTrat.img_ProjTransformation(img)
cv2.imshow("test", img)
cv2.waitKey(0)

########################
## segementação de cores
########################
# conversão BGR em HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# aplicando media para uniformizar a cor da imagem
hsv_image_blur = cv2.medianBlur(hsv_image, 3)
# aplicando mascara de limiarização
HSV_Min = (19, 0, 0)
HSV_Max = (22, 256, 256)
mask = cv2.inRange(hsv_image_blur, HSV_Min, HSV_Max)
output = cv2.bitwise_and(img, img, mask = mask)
# aplicando media para eliminar pontos (ruido)
output_med = cv2.medianBlur(output, 15)
cv2.imshow("test", output)
cv2.waitKey(0)

########################
## segementação de cores
########################
# Convertendo em uma imagem tons de cinza
gray = cv2.cvtColor(output_med, cv2.COLOR_BGR2GRAY)

# liminarizando a imagem para converter em tons de branco e preto
bin_img = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("test", bin_img)
cv2.waitKey(0)
# find contours in the thresholded image
cnts = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# loop over the contornos para identificar objetos
coords = []
for c in cnts:
    # compute the center of the contour
    print(cv2.contourArea(c))
    if(cv2.contourArea(c)>15):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coords.append([cX,cY])
# Gerando um circulo sobre o objeto identificado
cv2.circle(img, coords[0], 7, (0, 0, 255), 2)
cv2.imshow("test", img)
cv2.waitKey(0)