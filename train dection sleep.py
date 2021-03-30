#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from numpy import loadtxt

from keras.models import Sequential,model_from_yaml
from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten,MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2


# In[2]:


IMG_SIZE=30
training_data=[]
Folder='dataset/'
def create_training_data():
    for img in os.listdir(Folder):
        s=''
        i=img.split('-')
        if 'O' in i:
            s='open'
        else:
            s='close'
        if 'L'in i:
            s+='-left'
        else:
            s+='-right'
        if 'open-left'==s:
            s=0.
        elif 'open-right'==s:
            s=1.
        elif 'close-left'==s:
            s=2.
        else :
            s=3.
        
        img_array=cv2.imread(Folder+img)
        new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
        training_data.append([new_array,s])
create_training_data()


# In[ ]:


import sys 
sys.getsizeof(training_data)


# In[ ]:


import random
random.shuffle(training_data)


# In[ ]:


x=[]
y=[]
for fea,label in training_data:
    x.append(fea)
    y.append(label)


# In[ ]:


k=int(len(x)*0.70)
X_train=x[:k]
Y_train=y[:k]
X_test=x[k:]
Y_test=y[k:]


# In[ ]:


X_train.shape[1:]


# In[ ]:


import pickle
pickle_out=open('X_train.pickle','wb')
pickle.dump(X_train,pickle_out)
pickle_out.close()
pickle_out=open('Y_train.pickle','wb')
pickle.dump(Y_train,pickle_out)
pickle_out.close()
pickle_out=open('X_test.pickle','wb')
pickle.dump(X_test,pickle_out)
pickle_out.close()
pickle_out=open('Y_test.pickle','wb')
pickle.dump(Y_test,pickle_out)
pickle_out.close()
print('Finish')


# In[4]:


import pickle
X_train=pickle.load(open('X_train.pickle','rb'))
Y_train=pickle.load(open('Y_train.pickle','rb'))


# In[6]:


X_train=np.asarray(X_train,dtype=np.float32)
X_train=X_train/255.0
X_train


# In[8]:


x=X_train.reshape(-1,30,30,3)
y=np.asarray(Y_train).reshape(-1,1)
print(x[:5])
y[:5]


# In[9]:


print(x.shape)
y.shape


# In[10]:


ANN=Sequential([
    Flatten(input_shape=(30,30,3)),
    Dense(3000,activation=('relu')),
    Dense(1000,activation=('relu')),
    Dense(4,activation=('sigmoid'))
    
    
])
ANN


# In[11]:


ANN.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
ANN


# In[12]:


ANN.fit(x,y , epochs=50)


# In[17]:


mode_yaml = ANN.to_yaml()
with open("ANNOpenCloseModel.yaml", "w") as yaml_file:
    yaml_file.write(mode_yaml)
# serialize weights to HDF5
ANN.save_weights("ANNOpenCloseWeights.h5")
print("Saved model to disk")


# In[22]:


cnn = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
cnn


# In[23]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
cnn


# In[27]:


cnn.fit(x, y, epochs=60)


# In[32]:


mode_yaml = cnn.to_yaml()
with open("CNNOpenCloseModel.yaml", "w") as yaml_file:
    yaml_file.write(mode_yaml)
# serialize weights to HDF5
cnn.save_weights("CNNOpenCloseWeights.h5")
print("Saved model to disk")


# In[9]:


import pickle
X_test=pickle.load(open('X_test.pickle','rb'))
Y_test=pickle.load(open('Y_test.pickle','rb'))
print(X_test[:5])
print(Y_test[:5])


# In[10]:


X_test=np.asarray(X_test,dtype=np.float32)
X_test=X_test/255.0
X_test[:5]


# In[12]:


x_test=X_test.reshape(-1,30,30,3)
y_test=np.asarray(Y_test).reshape(-1,1)
print(x_test[:5])
y_test[:5]


# In[14]:


ANN.evaluate(x_test,y_test)


# In[17]:


cnn.evaluate(x_test,y_test)


# In[29]:


x=np.asarray(cv2.resize(cv2.imread('dataset/C-R-15995.jpeg'),(30,30)),dtype=np.float32)/255.0
x


# In[30]:


x=x.reshape(-1,30,30,3)
x

