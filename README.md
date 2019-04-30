# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals of this project were the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Output_Images/Nvidia_Behavioral_Cloning_Architecture.png "Architecture"


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

The Behavioral-Cloning.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture model that I was used was based of the Nvidia research paper called "End-To-End Learning For Self-Driving Cars". 
The paper can be found ([here](https://arxiv.org/pdf/1604.07316v1.pdf))

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Along with this, I also only considered a select amount of data that had a steering angle of 0. This also seemed to help prevent overfitting and improved the models preformance.  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving on both tracks to collect data. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My training strategy consisted of first training the model based off the sample data that was provided to us by Udacity. After I had run the simulation using this data, I identified areas the car seemed to struggle with, such as when the car was near the edge of the road. I then went back and collected data using the simulator for these struggle areas for the car. Another thing I found very helpful was using all three cameras. Finding the correct angle correction was a bit challenging, but the using all three cameras to train the model was extremely useful. 

Before running the simulation on the new model I developed using new data, I made srue that the test data and validation data that I was using obtained low loss values, inidcating the model was preforming well. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded laps on track one using center lane driving method. I also drove the revesre direction and collected data to when the car was near the edge of the road and needed to be corrected towards the center. 

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles when the steering was not zero. 


After the collection process, I had around 24,000 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

