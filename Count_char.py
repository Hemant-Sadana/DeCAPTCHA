import glob
import cv2 
import numpy as np 
height = []
width = []
# Reading the input image 
acc = 0
count = 0
path = "train/*.*"

for file in glob.glob(path):
    if(file[-8] == '\\'):
        numchar = 3
    else:
        numchar = 4
    img = cv2.imread(file, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    x = np.array(hsv[0][0],np.uint8)
    
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            if (hsv[i][j] == x)[0]:
                hsv[i][j]=0
    
    kernel = np.ones((8,8), np.uint8) 
    img_erosion = cv2.erode(hsv, kernel, iterations=1) 
    
    _, _, grayscale = cv2.split(img_erosion)
    
    cv2.threshold(grayscale,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    image, contours, hier = cv2.findContours(grayscale, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    detchar = 0
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for ctr in contours:

        x, y, w, h = cv2.boundingRect(ctr)
        width+=[w]
        height+=[h]
        roi = grayscale[y:y+h, x:x+w]
        cv2.imshow('abc',roi)
        cv2.waitKey(0)
        print('Width',w)
        print('Height',h)
        detchar = detchar + 1
    if detchar == numchar:
        acc = acc+1
    
