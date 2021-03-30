#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import dlib
import numpy as np
import socket
import numpy as np
from imutils import face_utils
import ast
import time
import pandas as pd
from numpy import loadtxt
import threading
import tensorflow
from keras.models import Sequential,model_from_yaml
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten,MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt


# In[3]:

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print('loaded dlib')

# In[4]:


yaml_file = open('ANNOpenCloseModel.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
ANN = model_from_yaml(loaded_model_yaml)
# load weights into new model
ANN.load_weights("ANNOpenCloseWeights.h5")
print("Loaded model from disk")
ANN.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
ANN

# In[1]:


def sleeping():

 
    while True:
        print('start')
        cap = cv2.VideoCapture(0)
        sleep=[]
        start_time = time.time()
        end =int(time.time() - start_time)
        while( end  < 15):
            # Capture frame-by-frame
            ret, img = cap.read()

            opCl=None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            mask = np.zeros_like(gray)
            
            fac=[]
            for face in faces:

                landmarks = predictor(gray, face)
                landmarks_points = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points.append((x, y))

                # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                x1=landmarks.part(36).x 
                x2=landmarks.part(39).x 
                y1=landmarks.part(37).y 
                y2=landmarks.part(40).y
                xb1=landmarks.part(18).x 
                xb2=landmarks.part(22).x 
                yb1=landmarks.part(18).y 
                yb2=landmarks.part(22).y
                righteye=gray[yb1:y2,x1:x2]
                x1=landmarks.part(42).x 
                x2=landmarks.part(45).x 
                y1=landmarks.part(43).y 
                y2=landmarks.part(46).y 
                xb1=landmarks.part(23).x 
                xb2=landmarks.part(27).x 
                yb1=landmarks.part(23).y 
                yb2=landmarks.part(27).y
                lefteye=gray[yb1:y2,x1:x2]
                points = np.array(landmarks_points, np.int32)
                convexhull = cv2.convexHull(points)
                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv2.imwrite('lefteye.jpg',lefteye)
                xlefteye=cv2.imread('lefteye.jpg')

                xlefteye=np.asarray(cv2.resize(xlefteye,(30,30)),dtype=np.float32)/255.0
                xlefteye=xlefteye.reshape(-1,30,30,3)
                cv2.imwrite('righteye.jpg',righteye)
                xrighteye=cv2.imread('righteye.jpg')
                xrighteye=np.asarray(cv2.resize(xrighteye,(30,30)),dtype=np.float32)/255.0
                xrighteye=xrighteye.reshape(-1,30,30,3)
                os.remove('lefteye.jpg')
                os.remove('righteye.jpg')

                left=None
                for i in ANN.predict_classes(xlefteye):
                    left=i
                print(left)
                right=None
                for i in ANN.predict_classes(xrighteye):
                    right=i
                print(right)
                if ((right == 0 )or(right == 1))or ((left == 0) or (left== 1)):
                    opCl=True
                else:
                    opCl=False
                print(opCl)
                sleep.append(opCl)
            end=int(time.time() - start_time)


        c=0

        for i in sleep:
            if not i :
                c+=1
        if len(sle)==0:
            continue
        sle=(c/len(sleep))*100
        print(sle)
        if sle>50.0:
            print('is sleeping')
        else :
            print('is not sleeping')
        print('wait 120 secound')
        cap.release()
        time.sleep(120)
        


# In[3]:
sle=threading.Thread(target=sleeping)
sle.start()


