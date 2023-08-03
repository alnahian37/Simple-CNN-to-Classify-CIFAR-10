# %%
import tensorflow as tf
#print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, MaxPooling2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from mycnn_class import MyModel



# %%
###### CHANGE THIS ACCORDING TO YOUR TESTING DATA ######
test_data = np.load('testing_data.npy')
te_labels = np.load('testing_label.npy')
test_data = np.float32(test_data/255)


# %%
#function for 1 hot label encoding
def label_encoding(label):
    
    y = np.zeros([int(max(label))+1,len(label)])
    
    for i in range(len(label)):
        y[int(label[i]),i] = 1

    y=y.T    
    return y

#function for computing classification performance
def performance_metrics(y_true,y_pred):
    matrix=confusion_matrix(y_true, np.argmax(y_pred,axis=1))
    accuracies=matrix.diagonal()/matrix.sum(axis=1)
    errors=1-accuracies
    return accuracies,errors

# %%
test_label = label_encoding(te_labels)
test_label = np.float32(test_label)

print(test_label.shape)

# %%
##### LOAD THE CNN CLASSIFIER #####
model = MyModel()

#### LOAD THE WEIGHTS #####
model.load_weights("model_weights")

y=model(test_data)
acc,err=performance_metrics(te_labels,y)

print("Test Accuracy:",acc)
print("Test Error:",err)
print("\nAverage Test Accuracy:",np.mean(acc))
print("Average Test Error:",np.mean(err))

# %%
weights_conv =  np.array(model.conv1.get_weights()[0])
print(weights_conv.shape)
plt.figure(figsize=(8, 6), dpi=100)
for i in range(16):
    image = weights_conv[:,:,:,i]
    plt.subplot(4,4,i+1)
    plt.imshow(image)
    plt.colorbar()
    plt.axis('off')
plt.show()

# %%



