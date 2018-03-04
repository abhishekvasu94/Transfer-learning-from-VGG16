# Transfer-learning-from-VGG16

This is a basic transfer learning project. I used the pretrained VGG16 model minus the last 3 fully connected layers, and added a couple of hidden layers and an output layer to it. This model aims to predict whether a given image is a car or a bicycle.
The data was obtained from the Flickr API.
This exercise was done as part of the Intro to Machine learning course at NYU Tandon School of Engineering, in the fall of 2017.

## Organisation of the files

The "train" folder contains 2 sub folders - "car" and "bicycle". Each of these folders contains 1000 images of cars and bicycles respectively. The "test" folder is organised in a similar fashion. The "car" and "bicycle" folders in the test set contains 300 images of each.

## Dependencies

Python
Keras
Numpy
