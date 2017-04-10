# Traffic Sign Recognition 

## German Traffic Sign Classification Project Report

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./report_images/original.png "Original"
[image1]: ./report_images/bar_plot.png "Visualization of the training set"
[image2]: ./report_images/histogram_equalization.png "Histogram Equalization"
[image3]: ./report_images/grayscale.png "Grayscaling"
[image4]: ./report_images/normalization.png "Normalization"
[image5]: ./new_images/new_images_5.png "Five new German traffic signs"
[image10]: ./report_images/new_image.png "New Image Prediction"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/RobotMa/CarND-Traffic-Sign-Classifier-New)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Python built-in functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the images in the training set with respect to the labels of classes ranging from 0 to 42. It can be seen that the type of images are not evenly distributed which might result in different level of accuracy when classifying different images using the trained neural network. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To begin with, the original plotted images in the training set are as below.

![alt_text][image0]

As a first step, the images are equalized on their color histograms to sharpen the contrast. 

Here is an example of traffic sign images after histogram equalization.

![alt_text][image2]

Then a convertion from RGB to grayscale is performed based on the finding in Yann LeCun's paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" that traffic sign detection accuracy was improved by ignoring the color information in the images. This effect is also observed when training the neural network in this project. 

Here is an example of traffic sign images after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because it is considered to be helpful for smoothing the noises in the images. However, the difference between the non-normalized and normalized images is trivial in this case, and this might be due to the histogram equalization process performed in advance. 

Here is an example of traffic signs images after normalization.
![alt_test][image4]

I didn't generate or augment the existing data set because the achieved validation accuracy is above 93%. But it is known that the neural network can be better trained with a larger training set, which can be obtained by generating more images using various affine transformations. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is pretty much the same as the original LeNet, with the addition of a dropout layer after the 3rd layer. It is consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation    		|   											|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6   	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Activation    		|         								    	|
| Max pooling			| 2x2 stride, valid padding, outputs 5x5x16		|
| Flatten				| output 400									|
| Fully connected       | output 120       				   	    		|
| Activation            |                                               |
| Dropout               | 0.5 keepprob                                  |
| Fully connected       | output 84                                     |
| Activation            |                                               |
| Fully connected       | output 43                                     |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I selected the batch_size = 128, epochs = 80, the AdamOptimizer with learning_rate = 0.001. The basic pipeline follows that of LeNet, while the epochs is increased significantly given the time for the validation and training accuracies to converge. However, epochs = 80 is not the best number of epochs given the current architecture and data set, a larger epochs is highly likely to further improve both the validation and training accuracies.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.0%. 
* validation set accuracy of 94.6%. 
* test set accuracy of 92.2%. 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

ANS: The LeNet architecture was chosen at first because it was famous for classifying grayscaled images.
* What were some problems with the initial architecture?

ANS: The original training accuracy and validation accuracy were not high enough. 
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

ANS: A dropout layer is added after the 1st fully connected layer which boosted both the training accuracy and validation accuracy. In addition, a LeNet based architecture which takes RGB images was tested but failed to give high training and validation accuracy. Feeding grayscaled images is adopted instead.
* Which parameters were tuned? How were they adjusted and why?

ANS: Paramters that were tested include the hyperparameters mu, sigma, keep_prob, EPOCHS and learning rate. At the end, keep_prob is tuned to be 0.5 and EPOCHS is set to be 80.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

ANS: A dropout layer is helpful in terms of preventing the neural network from getting overfitting. 

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)	| Speed limit (120km/h)  						| 
| Road work    			| General caution  								|
| Road work				| Road work										|
| No passing    		| Speed limit (60km/h)              			|
| Stop      			| Stop                							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This result is inferior to the test accuracy of 92.2%. However, several trials were performed on retraining the entire network and classifying these five images. The trained classifier can give a maximum of 80% prediction rate. This shows that when the test set is different from the training set (in terms of contrast, clarity, etc), the exisiting network can be unstable. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below is the illustration of the predictions of all five new German traffic images. The softmax probabilities for each prediction are plotted as a horizontal bar plot.

![alt_text][image10]

The first image of road image is mistakenly classified as general caution with a probability over 40%.

The second image of speed limit at 20 km/h is mistakenly classified as speed limit at 120 km/h with a probability of almost 100%. However, the image is also recognized as the corret traffic sign with a very low probability.

The third image of a different road work is correctly classified with almost 100% accuracy.

The forth image of no passing is mistakenly recognized as speed limit at 60 km/h with a confidence of 70%, and no correct prediction is found within the top 5 softmax probabilities. A clos examination reals that the tilt in the new image  might be one of factors causing the incorrect predictions. Another reason can be that most of the no passing images in the training set are very dark while the new image is very bright. That can give the classifier a hard time as well.

The fifth image of stop is correctly classified with a confidence of 100%.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


