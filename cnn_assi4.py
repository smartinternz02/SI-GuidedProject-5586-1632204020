# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:25:27 2021

@author: jayas
"""

import pandas 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen= ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory('C:/KEC/AI & ML/Assignment/train', 
                                          target_size=(128,128), batch_size=32, class_mode="categorical")


x_test=train_datagen.flow_from_directory('C:/KEC/AI & ML/Assignment/test', 
                                          target_size=(128,128), batch_size=32, class_mode="categorical")

print(x_train.class_indices)



"""builing the model"""

model = Sequential()
#add cnn layer
model.add(Convolution2D(32,(5,5),input_shape=(128,128,3),activation="relu"))

#add maxpooling layer
model.add(MaxPooling2D(2,2))

#add flatten layer
model.add(Flatten())

#add hidden layer
model.add(Dense(units=128,activation="relu"))

#add output layer
model.add(Dense(units=70, activation="softmax"))

print(model.summary())

#configure the learning process
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

#fit the model
model.fit(x_train,steps_per_epoch= 47, epochs=100, validation_data=x_test,validation_steps=20 )



model.save("animal.h5")