import cv2
import imutils
import numpy as np
import json

class imgTrat:
    def __init__(self, img) -> None:
        self.bright = 255
        self.contrast = 127
        self.img = img
        #############################################
        # Opening JSON file
        f = open('config.json')
        # returns JSON object as a dictionary
        data = json.load(f)
        self.bright = int(data['imagem']['bright'])
        self.contrast = int(data['imagem']['contrast'])
        self.projCoords = np.array(data['imagem']['ProjTransfCoords'])
        # parametros da bola
        self.HSV_Min = np.array(data['ball']['HSV_Min'])
        self.HSV_Max = np.array(data['ball']['HSV_Max'])

        #############################################
    def img_ProjTransformation(self, img):
        #############################################
        # aplicando transposição da imagem
        # Coordenadas do Campo 
        fator = 4
        campo_dim=[[0,0],[fator*130,0],[fator*130,fator*150],[0, fator*150]]
        pts1 = np.float32(self.projCoords)
        pts2 = np.float32(campo_dim)
        M = cv2.getPerspectiveTransform(pts1,pts2)
        return cv2.warpPerspective(img,M,(fator*130, fator*150))
        ###cv2.imshow("Img_Transposta", img)
        # aplicando correção de brilho e contraste

    def apply_brightness_contrast(self, img): #, brightness = 255, contrast = 127):
        #brightness = self.map(brightness, 0, 510, -255, 255)
        #contrast = self.map(contrast, 0, 254, -127, 127)
        if self.bright != 0:
            if self.bright > 0:
                shadow = self.bright
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + self.bright
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
        else:
            buf = img.copy()
        if self.contrast != 0:
            f = float(131 * (self.contrast + 127)) / (127 * (131 - self.contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        #cv2.putText(buf,'B:{},C:{}'.format(self.bright,self.contrast),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return buf
    
class elementos:
    def __init__(self, imagem) -> None:
        self.imagemTrat = imagem
        #############################################
        # Opening JSON file
        f = open('config.json')
        # returns JSON object as a dictionary
        data = json.load(f)
        self.bright = int(data['imagem']['bright'])
        self.contrast = int(data['imagem']['contrast'])
        self.projCoords = np.array(data['imagem']['ProjTransfCoords'])
        self.Ball_HSV_Min = np.array(data['ball']['HSV_Min'])
        self.Ball_HSV_Max = np.array(data['ball']['HSV_Max'])
        self.T1Y_HSV_Min = np.array(data['time_amarelo']['HSV1_Min'])
        self.T1Y_HSV_Max = np.array(data['time_amarelo']['HSV1_Max'])
        self.T1G_HSV_Min = np.array(data['time_amarelo']['HSV2_Min'])
        self.T1G_HSV_Max = np.array(data['time_amarelo']['HSV2_Max'])
        self.T1Player_HSV_Max=np.zeros((3,3))
        self.T1Player_HSV_Min=np.zeros((3,3))
        self.T1Player_HSV_Min[0] = np.array(data['time_amarelo']['player1_HSV_Min'])
        self.T1Player_HSV_Max[0] = np.array(data['time_amarelo']['player1_HSV_Max'])
        self.T1Player_HSV_Min[1] = np.array(data['time_amarelo']['player2_HSV_Min'])
        self.T1Player_HSV_Max[1] = np.array(data['time_amarelo']['player2_HSV_Max'])
        self.T1Player_HSV_Min[2] = np.array(data['time_amarelo']['player3_HSV_Min'])
        self.T1Player_HSV_Max[2] = np.array(data['time_amarelo']['player3_HSV_Max'])

    def obj_position(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
        img = cv2.GaussianBlur(gray, (15, 15), 0)
        
        cv2.imshow("gaussi", img)
        cv2.waitKey(0)
      
      
        #img = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)[1]
        limiar, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imshow("limiariza", img)
        cv2.waitKey(0)

        # find contours in the thresholded image
        #cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        coords = []
        #coords = np.array(shape=(X,Y),dtype='u1')
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coords.append([cX,cY])
            # draw the contour and center of the shape on the image
        return coords
    
    def get_Ball_pos(self):
        hsv_image = cv2.cvtColor(self.imagemTrat, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.medianBlur(hsv_image, 3)
        mask = cv2.inRange(hsv_image, self.Ball_HSV_Min, self.Ball_HSV_Max)
        output = cv2.bitwise_and(self.imagemTrat, self.imagemTrat, mask = mask)
        
        cv2.imshow("get_ball: mask e medianblue1", output)
        cv2.waitKey(0)

        output_med = cv2.medianBlur(output, 15) # adcionoei 

        cv2.imshow("get_ball: medianblue2", output_med)
        cv2.waitKey(0)

        Ball_Pos = self.obj_position(output_med)
        return Ball_Pos[0]
    
    ######
    def detectat_posiion_tag(self, hsv_min, hsv_max,):
        hsv_image = cv2.cvtColor(self.imagemTrat, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.medianBlur(hsv_image, 5)
        cv2.imshow("hsv:", hsv_image)
        cv2.waitKey(0) 

        
        mask = cv2.inRange(hsv_image, hsv_min, hsv_max)
        
        cv2.imshow("mask_tag:", mask)
        cv2.waitKey(0)  

        output = cv2.bitwise_and(self.imagemTrat, self.imagemTrat, mask = mask)

        cv2.imshow("output:", output)
        cv2.waitKey(0) 

        output_med = cv2.medianBlur(output, 9)

        cv2.imshow("output_med:", output_med)
        cv2.waitKey(0) 

        posicao = self.obj_position(output_med)

        return posicao
    #########################
    def localizar_player(self, T1Y_Pos, T1G_Pos, Player_Pos):
        pPoints = np.zeros((3,2))
        pPoints[0] = Player_Pos[0]
        #encontrando feixa amarela mais proxima
        dist_min = 1000
        for k in T1Y_Pos:
            if(np.linalg.norm(np.array(pPoints[0]) - k)<dist_min):
                dist_min=np.linalg.norm(np.array(pPoints[0]) - k)
                pPoints[1] = k
        #encontrando feixa verde mais proxima
        dist_min = 1000
        for k in T1G_Pos:
            if(np.linalg.norm(np.array(pPoints[0]) - k)<dist_min):
                dist_min=np.linalg.norm(np.array(pPoints[0]) - k)
                pPoints[2] = k
        p1Pose = np.zeros(3)
        p1Pose[0] = (pPoints[0,0]+pPoints[1,0]+pPoints[2,0])/3 #coordenada X
        p1Pose[1] = (pPoints[0,1]+pPoints[1,1]+pPoints[2,1])/3 #coordenada Y
        #print([T1Y_Pos[0][0],T1Y_Pos[0][1]])
        p1Pose[2] = np.arctan2((pPoints[1,0]-p1Pose[0]),(pPoints[1,1]-p1Pose[1]))
        print(np.cos(p1Pose[2]))
        return p1Pose


    def get_Player_pos2(self, imagem, player):

        T1Y_Pos = self.detectat_posiion_tag(self.T1Y_HSV_Min, self.T1Y_HSV_Max)
        T1G_Pos = self.detectat_posiion_tag(self.T1G_HSV_Min, self.T1G_HSV_Max)
        Player_Pos = self.detectat_posiion_tag(self.T1Player_HSV_Min[player], self.T1Player_HSV_Max[player])
        
        var = (T1Y_Pos, T1G_Pos,Player_Pos)
        #return T1Y_Pos
        print(var)

        pos = self.localizar_player(T1Y_Pos, T1G_Pos, Player_Pos)

        return np.int0([pos[0], pos[1]])


    def get_Player_pos(self, imagem, player):
        hsv_image = cv2.cvtColor(self.imagemTrat, cv2.COLOR_BGR2HSV)
        # mascara de limiarização do time
        mask_T1Y = cv2.inRange(hsv_image, self.T1Y_HSV_Min, self.T1Y_HSV_Max)
        mask_T1G = cv2.inRange(hsv_image, self.T1G_HSV_Min, self.T1G_HSV_Max)
        # mascara de limiarização do jogador
        mask_Player = cv2.inRange(hsv_image, self.T1Player_HSV_Min[player], self.T1Player_HSV_Max[player])
        
        cv2.imshow("mask_player:", mask_Player)
        cv2.waitKey(0)
        
        
        # Aplicando mascara ao time
        output_T1Y = cv2.bitwise_and(self.imagemTrat, self.imagemTrat, mask = mask_T1Y)
       
        cv2.imshow("output_t1y:", output_T1Y)
        cv2.waitKey(0)
       
        output_T1G = cv2.bitwise_and(self.imagemTrat, self.imagemTrat, mask = mask_T1G)
        
        cv2.imshow("output_t1g:", output_T1G)
        cv2.waitKey(0)
        
        # Aplicando mascara ao jogador
        output_Player = cv2.bitwise_and(self.imagemTrat, self.imagemTrat, mask = mask_Player)
        # obtendo a posição dos objetos detectados
        T1Y_Pos = self.obj_position(output_T1Y)
        T1G_Pos = self.obj_position(output_T1G)
        Player_Pos = self.obj_position(output_Player)
        pPoints = np.zeros((3,2))
        pPoints[0] = Player_Pos[0]
        #encontrando feixa amarela mais proxima
        dist_min = 1000
        for k in T1Y_Pos:
            if(np.linalg.norm(np.array(pPoints[0]) - k)<dist_min):
                dist_min=np.linalg.norm(np.array(pPoints[0]) - k)
                pPoints[1] = k
        #encontrando feixa verde mais proxima
        dist_min = 1000
        for k in T1G_Pos:
            if(np.linalg.norm(np.array(pPoints[0]) - k)<dist_min):
                dist_min=np.linalg.norm(np.array(pPoints[0]) - k)
                pPoints[2] = k
        p1Pose = np.zeros(3)
        p1Pose[0] = (pPoints[0,0]+pPoints[1,0]+pPoints[2,0])/3 #coordenada X
        p1Pose[1] = (pPoints[0,1]+pPoints[1,1]+pPoints[2,1])/3 #coordenada Y
        #print([T1Y_Pos[0][0],T1Y_Pos[0][1]])
        p1Pose[2] = np.arctan2((pPoints[1,0]-p1Pose[0]),(pPoints[1,1]-p1Pose[1]))
        print(np.cos(p1Pose[2]))
        return p1Pose