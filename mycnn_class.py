# %%

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, MaxPooling2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()    
    self.conv1=Conv2D(filters = 16,kernel_size = 5, padding='valid', strides = (1,1), activation = 'relu',input_shape=(32,32,3))
    self.dropout1 = Dropout(0.25)
    self.mp1=MaxPooling2D(pool_size=(2, 2), strides = (2,2))
    self.conv2=Conv2D(filters = 32,kernel_size = 5, padding='valid', strides = (1,1), activation = 'relu')
    self.mp2=MaxPooling2D(pool_size=(2, 2), strides = (2,2))
    self.conv3=Conv2D(filters = 64,kernel_size = 3, padding='valid', strides = (1,1), activation = 'relu')
    self.dropout2 = Dropout(0.25)
    self.flatten=Flatten()
    self.d1=Dense(units = 500, activation='relu')
    self.dropout3=Dropout(0.5)
    self.d2=Dense(units = 10, activation = 'softmax')
    #return model

  def call(self, x,training=False):
    x = self.conv1(x)
    if training:
      x = self.dropout1(x)
    x = self.mp1(x)
    x = self.conv2(x)
    x = self.mp2(x)
    x = self.conv3(x)
    if training:
        x = self.dropout2(x)
    x = self.flatten(x)
    x = self.d1(x)
    if training:
        x = self.dropout3(x)
    x = self.d2(x)
    return x


