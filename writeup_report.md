**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model architecture with which I settle was 8 convolutional Layers with 3 by 3 lkernal size and 8 in depth. I have also applied 4 pooling layers each after 2 convolutinal layers. The convolutional layers are continued with 2 feed forward network with 256 and 128 neurons.

#### 2. Attempts to reduce overfitting in the model

First attempt was to create a overfitted model. The overfitting was than overcome by applying in total 3 dropout layers. One after fourth convolutional layer, second after first feed fprward layer and third after second feed forward layer

#### 3. Model parameter tuning

The optimizer used is ADAM optimizer with all default values

#### 4. Appropriate training data

The data was collected on the track 1 for almost 2 laps resulting in 12000 images. The training data was then augumented with images from left camera and right camera with a steering angle correction of 0.2 . Data was further augumented by flipping the center image and negating the steering angle.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

First approach was to build a minimal viable project. Only on the sample data given and trying to overfit on the same. 
After the training pipeline was written with minimal layers, the data augumentation module was created. Overfitting on the sample data was the first step after augumenting the data. Avoiding overfitting through dropout was the next step. This helped a lot but still the car was not driving correctly. A readon for this was that I was training on BGR format, but driving the car in RGB format. 

Collectinf more data and solving the color codes did the magic the car started driving correctly after trainingn for 10 epochs.

#### 2. Final Model Architecture

The final model architecture  consisted of a convolution neural network with the 8 convolutional layers (3*3*8) and two fered forward wth 256 and 128 neurons.
