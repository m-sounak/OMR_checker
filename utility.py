# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 01:19:20 2021

@author: Sounak Majumder
"""
import cv2
import numpy as np

def rectCountour(contours):
    
    rect=[]
    for i in contours:
        area=cv2.contourArea(i)
        #print("Area",area)
        if area>50:
            peri=cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            #print("Corner points",len(approx))
            if len(approx)==4:
                rect.append(i)
    #print(len(rect))
    rect = sorted(rect,key=cv2.contourArea,reverse=True)
    
    return rect

def getCorners(contour):
    peri=cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.02*peri,True)
    return approx

def reorder(points):
    points=points.reshape((4,2))
    pointsnew = np.zeros((4,1,2),np.int32)
    add = points.sum(1)
    #print(points,add)
    pointsnew[0]=points[np.argmin(add)] # [0,0]
    pointsnew[3]=points[np.argmax(add)] # [w,h]
    diff = np.diff(points,axis=1)
    pointsnew[1]=points[np.argmin(diff)] # [w,0]
    pointsnew[2]=points[np.argmax(diff)] # [0,h]
    
    return pointsnew

def splitBoxes(img):
    rows = np.vsplit(img,5)
    #cv2.imshow("Split",rows[0])
    boxes = []
    i=0
    for r in rows:
        cols = np.hsplit(r,4)
        for box in cols:
            boxes.append(box)
            #cv2.imshow("Split",box)
    return boxes

def showAnswers(img,myIndex,grading,ans,questions,choices):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)
    
    for x in range (questions):
        myAns = myIndex[x]
        cx = (myAns * secW) + int(secW*1.19)
        cy = int(x*secH*0.8) + int(secH//2.8)
        
        if grading[x]:
            optionColor = (0,255,0)
        else:
            optionColor = (0,0,255)
            correct = ans[x]
            dx = (correct * secW) + int(secW*1.19)
            
            cv2.circle(img,(dx,cy),25,(0,255,0),cv2.FILLED)
            
        cv2.circle(img,(cx,cy),40,optionColor,cv2.FILLED)
    return img
    