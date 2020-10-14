# Behavioral Cloning

## Writeup Template

---

### Behavioral Cloning Project

##### The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior.
- Build, a convolution neural network in Keras that predicts steering angles from images.
- Train and validate the model with a training and validation set.
- Test that the model successfully drives around track one without leaving the road.
- Summarize the results with a written report.

---
## Rubric Points


##### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

- model.py containing the script to create and train the model. I also included model.HTML because it's easier.
- drive.py for driving the car in autonomous mode.
- model.h5 containing a trained convolution neural network.
- writeup_report. summarizing the results.
- run1.mp4 showing a video of the car driving 1 fill lap autonomously in track 1.

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

 
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
##### A) Preprocessing images
- Images are cropped (i.e. 70 rows from the top and 20 from the bottom) using Cropping2D.
- Images are normalized using Lambda.

##### B) NVIDIA architecture:
I used the NVIDIA model with some adjustments. The layers I ended up using are:
- 4 convolutional layers.
- A dropout layer.
- A flatten layer.
- 3 fully connected layers.

*Further explanation of the architecture is provided in the 'Solution Design Approach'.*

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layers in order to reduce overfitting.

The model was trained and validated on different shuffled data sets to ensure that the model was not overfitting. The validation, set was manually separated from training data since splitting using 'model.fit' isn't done at random as explained in the [post](https://discussions.udacity.com/t/validation-loss-remains-same-and-car-drunken-behavior/230842/16) .The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

- The model used adam optimizer, so the learning rate was not tuned manually.
- The steering measurements I calculated myself needed some tuning so I added lines (19-20 in the second section) for further tuning. Please refere to model.html for more explanation

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 1 laps of center lane driving both clockwise and counter-clockwise, recovering from the left and right sides of the road, going through hard and soft turns, and many places where the car couldn't stay on track, such as the end of the bridge.

*For details about how I adjusted the training data, please check the next section.*

#### 1. Solution Design Approach

The overall strategy for deriving the model architecture was to manipulate the steering angles to make it more consistent.

Since it was designed for self-driving cars, I figured the NVIDIA model would be closer to solving this problem than LeNet for example. I only changed the units of the last fully connected layer to fit our number of outputs, and removed the last convolution layer to reduce the complexity of the model, after all this project is a lot simpler than actual self-driving cars.

In order to gauge how well the model was working, I manually split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting, so I added a dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases; I kept augmenting adjusted data until the car was stable.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Architecture and Training Documentation

#### 2. Final Model Architecture

The final model architecture consisted of the following layers:
- A convolution layer with relu activation, 24 filters, 5X5 2D convolution window (ie, kernel size) and (2, 2) subsamples (strides).
- A convolution layer with relu activation, 36 filters, 5X5 2D convolution window and stireds of (2, 2).
- A convolution layer with relu activation, 24 filters, 5X5 2D convolution window and stireds of (2, 2).
- A convolution layer with relu activation and 24 filters.
- I added a dropout layer by factor of 50% to the NVIDIAmodel to reduce overfitting.
- A flatten layer.
- A fully connected layer with a 100 units and 'he_normal' initializer.
- A fully connected layer with 50 units.
- A fully connected layer with 1 unit.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first used a combination of 1 laps of center lane driving both clockwise and counter clockwise, data recovering from the left and right sides of the road, going through hard and soft turns, and the places where the car couldn't stay on track, such as the sandboxes.

The biggest shift in result happed when I manually modified the steering angles, as I didn't have a joystick. First I followed the images in the IMG folder until I got to the turning points and changed the steering measurement of the correlated data. Since I was using a keyboard, there were many turning images associated with 0 steering angles. I calculated the average for the steering angles for each turn and copied and pasted that data for augmentation. In the code I included a section (while reading the data) to further adjust the manipulated data. The figure bellow shows a histogram of the steering angles.

<figure>
 <img src="histogram for steering angle measurments.png" width="361" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> A histogram of steering angles (above)</p> 
 </figcaption>
</figure>
 <p></p> 

I randomly shuffled the data set and put of the data into a validation set and flipped images and angles. For example, here is an image that has then been flipped:

<figure>
 <img src="flipping.png" width="361" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> A sample of the data, original and flipped over (above)</p> 
 </figcaption>
</figure>
 <p></p> 

After the collection process, I had 14,488 data points, 11590 for training and 2898 for validation. The graph bellow shows their behavior over 7 epochs.

<figure>
 <img src="epoch.png" width="361" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Training for 7 epoches (above)</p> 
 </figcaption>
</figure>
 <p></p>

I preprocessed this data by cropping and normalizing them, before feeding them to the tuned NVIDIA architecture.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5-7 as evidenced by the figure above. I used an adam optimizer so that manually training the learning rate wasn't necessary.
