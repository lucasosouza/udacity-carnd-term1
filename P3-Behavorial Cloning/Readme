## Behavorial Cloning

What I found most interesting about this project is gathering data. That was very challenging for me - I got bad results with keyboard, and once I changed to joystick, I then had the issue I can't handle a joystick (last videogame I had was a 16bits Mega Drive, no joystick back then). So my training data is not good, and following the saying "trash in trash out", my model would therefore follow it.

### Data Preprocessing

I've tried several approaches of preprocessing:

- First I thought about training a model for each camera, and use an ensemble like results (like an average of the 3 predictions). However, I found out later driver.py only gets access to the center camera.

- I've done the data analysis on some sets and found out there were angle peaks, which were caused by bad driving, like the joystick slipping. I then smoothed the data using numpy.convolve, with an average of 30 frames windows. To smooth the data I first had to separate each training group (to ensure I would not blend frames of different groups). I captured the datetime from the filename and considered larger than 5 seconds transitions as a transition to a different group. Although smoothing helped, it is not used in the final solution.

- I've tinkered with removing outliers, above or below a datapoint, which did not perform well since the curves requires a wide angle (specially on track 2). I've also tinkered with using moving averages on drive.py to smooth the driving, which also did not perform well

- For the final model, I've recorded 3 types of driving: regular, and left-to-right and right-to-left corrections. In the correction trainings I would only consider the rows in which the angle indicates it is a correction (in right-to-left, the negative angles for example). That helped in making the training process faster, so I could avoid hitting pause and record button a lot of times.

#### Neural network model

I've tried the model described in nvidia paper, with 200x66 images, and a lot of layers. In all tests I run I've got very similar results, though, with a much smaller network and 32x16 images and 3 channels. I was first using a single channel (Y), but results improved a lot when I switched back to RGB. 

My final model has the following layers:
Features Creation:

- A BatchNormalization layer
- Convolutional 5x5 (stride 1x1)
- Max Pooling 2x2 
- Convolutional 4x4 (stride 1x1)
- Max Pooling 2x2

Classifier:

- Fully connected layer, 300
- Dropout 50%
- Fully connected layer, 50
- Fully connected layer, 1

For all layers tanh was used as the non linear activation function. The dropout seems to be at the optimal point to avoid overfitting and not underfit the model - adding another dropout layer, or removing the single dropout added, have a negative impact on the results.

Performance was a must for this project. I don't have local GPU, my local machine is simple, and using AWS was not practical since the upload speed of my network is not good enough to upload new images to AWS (even after I resized them). I usually have the option of using the high performance lab at the university, but I'm travelling currently.
 
#### Results

I got stuck at a suboptimal results, which I will try to push later when I get access to more hardware. It completes the first track most of the times. The first model I've created performed very good on the second track, but I started from scratch to try to improve the results further and decided to focus on the first track only to save some time.