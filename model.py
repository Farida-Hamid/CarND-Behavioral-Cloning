# coding: utf-8

# Self-Driving Car Engineer Nanodegree
# Deep Learning
# Project: Behavioral Cloning 


# Read the lines in the driving_log.csv file

import csv

lines = []

with open('C:/Users/Hamids/Desktop/driving data/driving_log.csv') as csvf:
    reader = csv.reader(csvf)
    for line in reader:
        lines.append(line)


# Read data and store the images in the "images" list and the steering angles in the "steer" list
# The lines 19-20 are to  further adjust the manupulated data. The *if* statement checks wheather 
# the data is manipulated or not, if so, it's multiplied by a factor. 

from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')

images = []
steer = []

for l in range(len(lines)):
    
    filename = lines[l][0]
    img = mpimg.imread(filename)
    images.append(img)
    images.append(np.fliplr(img))
    
    steernig_measurment = float(lines[l][3])
    
    if (0 < l < len(lines)-1) and (lines[l][3] == lines[l-1][3] == lines[l+1][3]):
        steernig_measurment *= 0.4
            
    steer.append(steernig_measurment)
    steer.append(-steernig_measurment)


# Seperate 20% of the data for validation

im_val = []
st_val = []

n = int(0.2 * len(images))

while n >= 0:
    
    randomN = randint(0, len(images) - 1)
    im_val.append(images[randomN])
    st_val.append(steer[randomN])
    
    del images[randomN]
    del steer[randomN]
    
    n -= 1


# Visualize data
# plot a histogram for steering angle measurments
plt.hist(steer, bins=50)
plt.show()

# show 10 random pictures
for i in range(5):
    index  = randint(0, len(images))
    print("image number", index, "steering measurement:", steer[index])
    plt.figure(figsize = (4, 4))
    plt.imshow(images[index])
    plt.show()


# Prepare validation and training data

import numpy as np

X_train = np.array(images)
y_train = np.array(steer)

X_val = np.array(im_val)
y_val = np.array(st_val)


# Pipeline 

from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D, Cropping2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Lambda, Dropout
import matplotlib.pyplot as plt

# preprocessing images
model = Sequential()
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))  # crpooing the top 45 and the 20 bottom raws
model.add(Lambda(lambda x: x / 250.0 - 0.5))                               # normalize the data

# NVidea archetecture:
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation = 'relu'))   # 4 convolution layers
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(43, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64, 5, 5, activation = 'relu'))

# I removed the line: model.add(Convolution2D(64, 5, 5, activation = 'relu')) to reduce the complexity of the model

model.add(Dropout(0.5))      # I added the dropout layer to the NVidea archetecture to avoid overfitting
model.add(Flatten())         # flatten layer

model.add(Dense(100, init='he_normal'))   # 3 fully connected layers
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, shuffle=True, validation_data=(X_val, y_val), nb_epoch=7, batch_size=750)

# save the model
model.save('model5.h5')


# Plot the training and validation loss for each epoch

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()