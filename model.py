import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

#Function to generate batches of labeled data from the directory
def data_generator(dir_name, nrow, ncol, batch_size):
	data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
	return data.flow_from_directory(dir_name, target_size=(nrow,ncol), batch_size=batch_size, class_mode='binary')


nrow = 64	#fix height of image
ncol = 64	#fix width of image
nchannel = 3	#fix number of channels in the image

train_set = 2000	#size of training set
test_set = 600		#size of test set
batch_size = 32		#mini batch size
steps_per_epoch = train_set/batch_size
validation_steps = test_set/batch_size
n_epochs = 5		#number of epochs to train for

#Import the pretrained VGG16 model, except the 3 fully connected layers at the end of the network
base_model = VGG16(weights='imagenet', include_top = False, input_shape=(nrow,ncol,nchannel))

#Create a new model
model = Sequential()

#Add layers from VGG16 to my model, layer by layer
for layer in base_model.layers:
	model.add(layer)


#Freeze layers from the pre-trained model so that I don't train them again
for layer in model.layers:
	layer.trainable = False


#Add a couple of fully connected layers of my own, and an output layer. Flatten output of last conv layer first.
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#Generating training and test data
train_data = data_generator('./train', nrow, ncol, batch_size)
test_data = data_generator('./test', nrow, ncol, batch_size)

#Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the model
#Test data is fed as validation data here, for convenience
model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=n_epochs,validation_data=test_data,validation_steps=validation_steps, verbose = 1)

