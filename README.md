
[//]: # (Image References)

[image1]: ./Output_Images/Nvidia_Behavioral_Cloning_Architecture.png "Architecture"
[image2]: ./Output_Images/track1.png "Track1"
[image3]: ./Output_Images/track2.png "Track2"

### Behavioral Cloning Project

#### Here is a link to the video result. Note the view of the video is from the viewpoint of the hood, not top-down as it was in training: [Video_Output](./Output_Video/Autonomous_Mode_Result.mp4)

The goals of this project were the following:
* Use the simulator provided by Udacity to collect data of good driving behavior
* Build, a convolution neural network using Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road



My project includes the following files:
* Behavioral_Cloning.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* video.py was used for generating an ouptut video based on the "frames" captured during the car driving autonomously
* model_preprocessed.h5 contains a trained CNN


The Behavioral-Cloning.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

### Model Architecture and Training Strategy

The architecture model that I was used was based of the Nvidia research paper called "End-To-End Learning For Self-Driving Cars". 
The paper can be found ([here](https://arxiv.org/pdf/1604.07316v1.pdf))

The model contains dropout layers in order to reduce overfitting. Along with this, I also only considered a select amount of data that had a steering angle of 0. This also seemed to help prevent overfitting and improved the models preformance.  

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving on both tracks to collect data. 


### Model Architecture and Training Strategy

My training strategy consisted of first training the model based off the sample data that was provided to us by Udacity. After I had run the simulation using this data, I identified areas the car seemed to struggle with, such as when the car was near the edge of the road. I then went back and collected data using the simulator for these struggle areas for the car. Another thing I found very helpful was using all three cameras. Finding the correct angle correction was a bit challenging, but the using all three cameras to train the model was extremely useful. 

Before running the simulation on the new model I developed using new data, I made sure that the test data and validation data that I was using obtained low loss values, inidcating the model was preforming well. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

![alt text][image1]

To capture good driving behavior, I recorded laps on track one using center lane driving method. I also drove the revesre direction and collected data to when the car was near the edge of the road and needed to be corrected towards the center. 

![alt text][image2]


Then I repeated this process on track two in order to get more data points.

![alt text][image3]


To augment the data sat, I also flipped images and angles when the steering was not zero. 


After the collection process, I had around 24,000 data points.

Finally, I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. I found that using a batch size of 120, 7 epochs, and a dropout percentage of .3 produced sound results. I used an adam optimizer so that manually training the learning rate wasn't necessary. 

The final result was a car that could drive autonomously around the tracks very smoothly, and my CNN achieved a loss value of .0160 after training had been complete. 

