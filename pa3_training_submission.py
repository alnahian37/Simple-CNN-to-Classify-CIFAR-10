# %%
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, MaxPooling2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# %%
train_data = np.load('training_data.npy')
tr_labels = np.load('training_label.npy')
test_data = np.load('testing_data.npy')
te_labels = np.load('testing_label.npy')

train_data = np.float32(train_data/255)

test_data = np.float32(test_data/255)

# %%
print(train_data.shape, tr_labels.shape, test_data.shape, te_labels.shape)
print(int(max(tr_labels)), int(max(te_labels)))

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
train_label = label_encoding(tr_labels)
train_label = np.float32(train_label)
test_label = label_encoding(te_labels)
test_label = np.float32(test_label)

print(train_label.shape, test_label.shape)

# %%
train_batches = tf.data.Dataset.from_tensor_slices((train_data, tr_labels)).shuffle(10000).batch(100)

test_batches = tf.data.Dataset.from_tensor_slices((test_data, te_labels)).batch(100)

# %%


# %%
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

# Create an instance of the model
model = MyModel()

# %%
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) 

# %%
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# %%
@tf.function
def training_function(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_acc(labels, predictions)


@tf.function
def testing_function(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_acc(labels, predictions)

# %%
EPOCHS = 100


train_errors=[]
train_accuracy=[]
test_errors=[]
test_accuracy=[]
train_losses=[]
test_losses=[]

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_acc.reset_states()
  test_loss.reset_states()
  test_acc.reset_states()

  for images, labels in train_batches:
    training_function(images, labels)

  for test_images, test_labels in test_batches:
    testing_function(test_images, test_labels)

  a=train_acc.result()
  b=test_acc.result()
  c=train_loss.result()
  d=test_loss.result()
  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {c}, '
    f'Accuracy: {a * 100}, '
    f'Test Loss: {d}, '
    f'Test Accuracy: {b * 100}'
  )
  train_losses.append(c.numpy())
  test_losses.append(d.numpy())
  train_accuracy.append(a.numpy())  
  test_accuracy.append(b.numpy())
  

  


 



# %%
print(train_data.shape,test_data.shape)

y_train_pred = []
#Performance calculation
for i in range(50):
    x=train_data[i*1000:(i+1)*1000]
    y=model(x)
    y_train_pred.append(y)
y_train_pred=np.concatenate(y_train_pred,axis=0)

print(y_train_pred.shape)


y_test_pred = []
for i in range(5):
    x=test_data[i*1000:(i+1)*1000]
    y=model(x)
    y_test_pred.append(y)
y_test_pred=np.concatenate(y_test_pred,axis=0)

print(y_test_pred.shape)


# %%
acc,err=performance_metrics(tr_labels,y_train_pred)
print("Train Accuracy:",acc)
print("Train Error:",err)

acctest,errtest=performance_metrics(te_labels,y_test_pred)
print("Test Accuracy:",acctest)
print("Test Error:",errtest)



# %%
print("Average train accuracy:",np.mean(acc))
print("Average test accuracy:",np.mean(acctest))
print("Average train error:",np.mean(err))
print("Average test error:",np.mean(errtest))

# %%
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(train_accuracy, label = "overall train accuracy")
plt.plot(test_accuracy, label ="overall test accuracy")
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel('training and test accuracy')
plt.show()

# %%
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(train_losses, label = "train loss")
plt.plot(test_losses, label = "test loss")
plt.xlabel('number of epochs')
plt.ylabel('training and test loss')
plt.legend()
plt.show()

# %%
weights_conv =  np.array(model.conv1.get_weights()[0])
print(weights_conv.shape)
plt.figure(figsize=(8, 6), dpi=100)
for i in range(16):
    image = weights_conv[:,:,:,i]
    plt.subplot(4,4,i+1)
    plt.imshow(image,cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.title('Filter {}'.format(i+1))
plt.show()

# %%
model.save_weights("model_weights")

# %%



