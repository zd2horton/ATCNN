#-------------------------------------------------------------------------------
# Name:        NeuralNetwork
# Purpose:
#
# Author:      Zach
#
# Created:     15/11/2019
# Copyright:   (c) Zach 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#import libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
import numpy as np
from keras.preprocessing import image

#cnn initialised, conv, pool and flatten
classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3
input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())

#full connection, connect convolutional network to neural network
#(sigmoid as activation function for last layer to find prob.)
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compile 2 layer neural network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
metrics = ['accuracy'])


#data augmentation, reduces overfitting on models, where amount of
#training data using information only in training data is increased
#fitting CNN to images

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
zoom_range = 0.2, horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
target_size = (64, 64), batch_size = 32, class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
target_size = (64, 64), batch_size = 32, class_mode = 'categorical')

#train model
#classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 10, validation_data = test_set, validation_steps = 800)
classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 10, 
                         verbose = 1, callbacks = None, validation_data = test_set, validation_steps = 800,
                         validation_freq = 1, class_weight = "auto", max_queue_size = 10, workers = 1, 
                         use_multiprocessing = False, shuffle = False, initial_epoch = 0)

#testing
test_image = image.load_img('catnormal.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] >= 0.5:
        prediction = 'dog'
else:
        prediction = 'cat'
print(prediction)

#from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())