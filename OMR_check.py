# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 00:58:33 2021

@author: Sounak Majumder
"""
import cv2
import numpy as np
import utility

path = "img1.jpg"
widthImg = 700
heightImg = 700
questions = 5
choices = 4
ans = [0,2,1,3,0]
webcamFeed = False
cameraNo = 0


cap = cv2.VideoCapture(cameraNo)
cap.set(10,150)

while True:
    if webcamFeed:
        success,image = cap.read()
    else:
        image = cv2.imread(path)


    image=cv2.resize(image,(widthImg,heightImg))
    imgContours = image.copy()
    imgFinal = image.copy()
    imgBiggestContours = image.copy()
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,10,50)
    
    try:
        contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours, -1,(0,255,0),10)
        
        rect = utility.rectCountour(contours)
        biggestContour = utility.getCorners(rect[0])
        grades = utility.getCorners(rect[1])
        #print(biggestContour)
        if biggestContour.size != 0 and grades.size!=0:
            cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),10)
            cv2.drawContours(imgBiggestContours,grades,-1,(255,0,0),10)
        
            biggestContour = utility.reorder(biggestContour)
            grades = utility.reorder(grades)
            
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            imgwarpColored = cv2.warpPerspective(image,matrix,(widthImg,heightImg))
            
            pt3 = np.float32(grades)
            pt4 = np.float32([[0,0],[320,0],[0,150],[320,150]])
            matrixG = cv2.getPerspectiveTransform(pt3,pt4)
            imgGradeDisplay = cv2.warpPerspective(image,matrixG,(320,150))
            
            imgWarpGray = cv2.cvtColor(imgwarpColored,cv2.COLOR_BGR2GRAY)
            imgTh = cv2.threshold(imgWarpGray,120,255,cv2.THRESH_BINARY_INV)[1]
            
            boxes = utility.splitBoxes(imgTh)
            #cv2.imshow("Test",boxes[2])
            #print(cv2.countNonZero(boxes[2]))
            
            myPixelVal = np.zeros((questions,choices))
            countC = 0
            countR = 0
            
            for img in boxes:
                totalPixels = cv2.countNonZero(img)
                myPixelVal[countR][countC] = totalPixels
                countC+=1
                if countC==choices:
                    countC=0
                    countR+=1
            #print(myPixelVal)
            
            myIndex = []
            for x in range (questions):
                arr = myPixelVal[x]
                ind = np.where(arr==np.amax(arr))
                #print(ind[0])
                myIndex.append(ind[0][0])
            #print(myIndex)
            
            grading = []
            for x in range (questions):
                if ans[x]==myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            score = sum(grading) * 100 / questions
            #print(score)
            
            imgResult = imgwarpColored.copy()
            imgResult = utility.showAnswers(imgResult,myIndex,grading,ans,questions,choices)
            imgRaw = np.zeros_like(imgwarpColored)
            imgRaw = utility.showAnswers(imgRaw,myIndex,grading,ans,questions,choices)
            
            invMatrix = cv2.getPerspectiveTransform(pt2,pt1)
            imgInvWarp = cv2.warpPerspective(imgRaw,invMatrix,(widthImg,heightImg))
            
            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(int(score))+"%",(55,100),
                        cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
            #cv2.imshow("Grade",imgRawGrade)
            invMatrixG = cv2.getPerspectiveTransform(pt4,pt3)
            invGradeDisplay = cv2.warpPerspective(imgRawGrade,invMatrixG,(widthImg,heightImg))
            
            imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
            imgFinal = cv2.addWeighted(imgFinal,1,invGradeDisplay,1,0)
            
                
        
        cv2.imshow("Final Result",imgFinal)
    except:
        cv2.imshow("Final Result",image)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("FinalResult.jpg",imgFinal)
        cv2.waitKey(100)
     
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if webcamFeed==False:
        cv2.waitKey(0)
        break
    
cap.release()
cv2.destroyAllWindows()