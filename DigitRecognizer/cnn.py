from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import pandas as pd
import numpy as np

#Load data
(X_train,y_train),(X_test,y_test) = mnist.load_data(path='mnist.npz')
train_data = pd.read_csv('./datasets/train.csv')
X_train_added = train_data.values[:,1:]
y_train_added = train_data.values[:,0:1]

#Load kaggle's test data
test_data = pd.read_csv('./datasets/test.csv')
test_data = test_data.values
test_data = test_data.reshape(test_data.shape[0],28,28,1)

#Reshape data
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
X_train_added = X_train_added.reshape(X_train_added.shape[0],28,28,1).astype('float32')
y_train_added = y_train_added.reshape(y_train_added.shape[0]).astype('float32')

#Add data
X_train = np.concatenate((X_train,X_train_added))
X_train = np.concatenate((X_train,X_train))
y_train = np.concatenate((y_train,y_train_added))
y_train = np.concatenate((y_train,y_train))

#Normalize data
X_train = X_train/255
X_test = X_test/255

#Convert label to categorical classes
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_categories = y_test.shape[1]

#Model definition
def cnn_model():
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_categories, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#Create model
model = cnn_model()

# Fit model
model.fit(X_train,y_train, validation_data = (X_test,y_test), epochs = 10, batch_size = 200, verbose = 2)

#Generate output
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred,axis = 1)
output = { 'ImageId' : [i for i in range(1,len(y_pred)+1)], 'Label' : y_pred}
output = pd.DataFrame(output,columns = ['ImageId','Label'])

#Write to csv 
output.to_csv('./cnn.csv',index = None, header = True)
