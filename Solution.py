import glob
import cv2 
import numpy as np 
  
# Reading the input image 
count = 1
numTest = 5
name = 0
path = "train/*.*"
test = 0
for file in glob.glob(path):
    if(test<2):
        test = test+1
        continue
    test = test + 1
    if(test == 4):
        break
    img = cv2.imread(file, 1)
    cv2.imwrite("Step_image/"+ str(name) +".png", img)
    name = name + 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite("C:/Users/hemant sadana/Desktop/Mtech first sem/Machine Learning/Assignments/Assignment 3/assn3/assn3/Step_image/"+ str(name) +".png", hsv)
    name = name + 1
    x = np.array(hsv[0][0],np.uint8)
    
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            if (hsv[i][j] == x)[0]:
                hsv[i][j]=0
#    cv2.imshow('Background Removed',hsv)
#    cv2.waitKey(0)
    cv2.imwrite("C:/Users/hemant sadana/Desktop/Mtech first sem/Machine Learning/Assignments/Assignment 3/assn3/assn3/Step_image/"+ str(name) +".png", hsv)
    name = name + 1
    kernel = np.ones((8,8), np.uint8) 
    img_erosion = cv2.erode(hsv, kernel, iterations=1) 
#    cv2.imshow('Eroded Image',img_erosion)
#    cv2.waitKey(0)
    cv2.imwrite("C:/Users/hemant sadana/Desktop/Mtech first sem/Machine Learning/Assignments/Assignment 3/assn3/assn3/Step_image/"+ str(name) +".png", img_erosion)
    name = name + 1
    _, _, grayscale = cv2.split(img_erosion)
#    cv2.imshow('Grayscale Image',grayscale)
#    cv2.waitKey(0)
    cv2.imwrite("C:/Users/hemant sadana/Desktop/Mtech first sem/Machine Learning/Assignments/Assignment 3/assn3/assn3/Step_image/"+ str(name) +".png", grayscale)
    name = name + 1
    
    fname = np.arange(26)
    alpha = 'A'
    allist = []
    for k in range(0, 26): 
        allist.append(alpha) 
        alpha = chr(ord(alpha) + 1) 
    cv2.threshold(grayscale,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    image, contours, hier = cv2.findContours(grayscale, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
#    cv2.imshow("contours", image)
#    cv2.waitKey(0)
    for ctr in contours:
        # Get bounding bo
        x, y, w, h = cv2.boundingRect(ctr)
        roi = image[y:y+h, x:x+w]
#        cv2.imshow('Segmented Image',roi)
#        cv2.waitKey(0)
        cv2.imwrite("C:/Users/hemant sadana/Desktop/Mtech first sem/Machine Learning/Assignments/Assignment 3/assn3/assn3/Step_image/"+ str(name) +".png", roi)
        name = name + 1
        filename = 'C:/Users/hemant sadana/Desktop/Mtech first sem/Machine Learning/Assignments/Assignment 3/assn3/assn3/Full_data/'+ str(count) +'.png'
#        cv2.imwrite(filename, roi)
        count = count + 1
