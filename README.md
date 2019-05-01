
[//]: # (Image References)

[image1]: ./Output_Images/Nvidia_Behavioral_Cloning_Architecture.png "Architecture"
[image2]: ./Output_Images/track1.png "Track1"
[image3]: ./Output_Images/track2.png "Track2"
[image4]: ./Output_Images/data_sample.png "Data"


### Behavioral Cloning Project

#### Here is a link to the video result. Note the viewpoint from the video is from the hood, not top-down as it was in training: [Video_Output](./Output_Video/Autonomous_Mode_Result.mp4)

The goals of this project were the following:
* Use the simulator provided by Udacity to collect data of good driving behavior by manually driving the car
* Build a convolution Neural Network using Keras to predict steering angles from data/images gathered during the step above 
* Train and validate the model with a training and validation set
* Test that the car in the simulator successfully drives smoothly on the road using the developed CNN model



My project includes the following files:
* Behavioral_Cloning.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode using the model developed by the CNN
* video.py was used for generating an ouptut video based on the "frames" captured during the car driving autonomously
* model_preprocessed.h5 contains a trained CNN 

### Model Architecture and Training Strategy

The architecture model that I was used was based of the Nvidia research paper called "End-To-End Learning For Self-Driving Cars". 
The paper can be found ([here](https://arxiv.org/pdf/1604.07316v1.pdf))

The model contains a dropout layer in order to reduce overfitting. Along with this, I also only considered a select amount of data that had a steering angle of 0. This also seemed to help prevent overfitting and improved the models preformance. The detailed layout of the model can be seen in the Behavioral-Cloning.py script, but the general architecture is seen in image below.

![alt text][image1]


My training strategy consisted of first training the CNN model based off the sample data that was provided to us by Udacity. To see how it preformed, I applied this model to the car and observed how it preformed in autonomous mode. I identified areas the car seemed to struggle with, such as when the car was near the edge of the road. It was not able to drive back towards the center, but rather continued off of the track. 

After identifying areas of concern, I went back and collected data using the simulator for the "struggle" areas for the car. Along with these areas of concern, to capture good driving behavior, I recorded laps on track one using center lane driving method. I also drove the revesre direction and collected data to when the car was near the edge of the road and needed to be corrected towards the center.  The images below shows examples of what operating the car in the simulator to collect data looked like on track one and two respectively. 

![alt text][image2]

![alt text][image3]

To get the best results, the model was trained and validated on seperate subsets of the data sets to help validate that the model was not overfitting. After the data collection process, I had around 24,000 data points, where the data was stored using a csv file and a folder containing the different images associated with the data points. The KEY was for the model to learn to associate an image, or driving instance, with the correct steering angle that should be applied. 

A very small sample of data stored in a csv file is shown below. Note that each image listed under the different cameras in the csv file is stored in the Image folder generated from collecting data.
![alt text][image4]

Finally, I randomly shuffled the data set and put 20% of the data into a validation set. Before running the simulation on the new model I developed using new data, I made sure that the test data and validation data that I was using obtained low loss values, inidcating the model was preforming well. 
At the end of the process, the vehicle was able to drive successfully drive autonomously around the track without leaving the road, and I had a training loss of .0133 and validation loss of .017.
