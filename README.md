
# Project - 03 Behavioral Cloning

Writing documents like these are always difficult because you try to explain what was an iterative process (getting training data, trying pre-processing techinque, trying model architecture, trying training method, trying model on simulator, and repeat) in a sequential manner. For this README I'm going to fully explain the method used for each pipeline step sequentially, but also talk about the iterative process within each step.

The pipeline of the entire process goes, Data -> Pre-Process -> Model -> Training -> Simulator.


## Training Data

Before I did any model work I recorded as suggested, 3 laps driving smoothly on the centerline of the track and then one lap of starting on the left edge of the track and driving to the center and then the same with the right edge. For these last two laps I would only record data as I drove from the edge to the center, not as I drove back to the edge of the track.

Using only this data and a model that was very close to my final architecture I was able to get as far as the hard right corner (2nd corner after the bridge). However at that corner, the car would always take the corner a bit wide and then since it was heading into the turn at a sharp angle (since it was going wide instead of being on the inside of the corner) it wouldn't turn sharp enough and would go over the edge into the lake. Even though I was achieving a good fit of my model to both the training, validation, and test data, it seemed to always go over the edge as it took the corner wide and wouldn't turn sharp enough no matter how many epochs I did nor how well the model fit the data.

At this point I realized I had a data problem and needed more data for this corner. What I attempted to do (although unsuccessfully) was to do more parking at an edge and then driving to the center, on this corner. This lead to some large steering angle inputs in the training data as I tried to be super aggressive in getting back to the center of the lane in the hopes that this would carry over into the model (which remember wasn't turning sharp enough at this corner).

You can visually see this addition in the training data in the image below in the area called "Right Corner" correction.

In this image everything above 0.5 means steering to the right, and below 0.5 is steering to the left. You can make out the invididual laps as there is only one right turn in the whole track. After the first few laps you can see the from the edge driving to teh middle (as it has aggresive steering inputs) followed by a lap and then the focus on the right turn. After that there are more laps (these are the ones where I took off center and wide driving lines) and then a few more corrections followed by a single centerline lap.

![alt text](trainingdata-edited.png "Title")



Looking back at the training data I already collected for the right corner, I noticed that for the data I had when doing a full lap I had always taken the corner on the inside. For the correction data, I was always starting with the car parallel to the lines of the road and driving to the center. However when my autonomous car was approaching this corner it was always coming in wide and then driving straight at (perpendicular) to the road lines before veering over the edge into the lake.

At this point I was also having troubles with my car sometimes taking the dirt path instead of the turning left at the corner just after the bridge. I realized that just like with the hard right corner that my original 3 training laps were almost too perfect as every time I took a nice inside line on the corner where my autonomous car would always take the corner wide and then sometimes just go straight down the dirt path.

I then recorded all of the data seen after "Right Corner" which includes multiple laps of driving off center (taking the corners wide and driving closer to the lane lines in some sections) and also some added corrections for dirt path. I also included corrections for the hard right that started with my car directly facing the outside lane on the hard right (start perpendicular to the line as opposed to parallel with it) as this better captured the situation the car was getting into when taking this corner wide.

Using all 11,224 images and steering angles of this final data set I was able to get the car to round the track with throttle values up to 0.3.

## Data Pre-Process

Initially I was normalizing the data to values between -0.5 and 0.5 for both the steering angle and images. However during the model architecture selection I eventually decided on wanting to use as many ReLu activations as possible, even for the very end of my dense layers. Since a ReLu activation layer can't output a negative number, I decided to move all of my data between 0.1 and 0.9 so that the final output wouldn't need any bias applied to it.

I also decided to cut out roughly the top half of the image as just from visual inspection it can be seen that all of the required info for driving between the lines is in the bottom half. This was also a measure against over fitting as the model would not be able to memorize the surrounding track as well (For example seeing the lake or a specific tree and knowing that sharp turn is needed). The original image and my modified pre-process image can be seen below.

![alt text](cut.png "Title")

It should be noted that at no point in my code do I use a python generator. My computer and video card had ample RAM for all the training data I generated. However if I needed to use up to 40,000 images or even more I understand why I would have needed to write a python generator.


## Model Architecture Selection

My initial thoughts going into this project was that I didn't need a convolutional network as I'm not searching for patterns in the image that are spatial invariant. I imagined a simplified version of the project where the entire drivable portion of the track is black and the part outside the lines is green. In that case a neural network with only dense layers should perform really well as when a specific pixel (on the left or right side of the image) starts getting color, it should be easy to train to turn away from that edge. However upon implementing a purely dense neural network I found the results very poor. It could be that I wasn't making the network dense enough but I didn't find the initial results promising enough to pursue further.

I then created a convolutional neural network with 3 convolutional layers and 3 dense layers. However my parameter size was massive (because I wasn't scaling the image down far enough) and it was taking a long time to train. At this point I viewed the nvidia paper supplied and saw that their CNN ended with a 1x18 image where as mine was ending with a much larger image and thus my parameter size was huge. I then added two more convolutional layers to bring my final image down to 1x15. 

I had to do a few things differently then them since they start with a different image size. The changes I made was instead of my 5 convolutional layers using kernel size being (5,5), (5,5), (5,5), (3,3), (3,3) with max pooling I used (6,6), (6,6), (3,3), (3,3), (3,3) with 2x2 max pooling after each layer except the last layer. This gave me a final image of 1x15.

After flattening the result of the final convolution layer I had 960 outputs. I then scaled my network down in a similar fashion to the nvidia one but using slightly different number of neurons. They used (1164, 100, 50, 10, 1) and I used (1024, 128, 64, 16, 1). I started with 1024 since it was slightly larger than the 960 outputs.


### Dropout and Activation Problem

Originally when using this model I was using linear activation for all layers and a dropout of 0.5. I did this as a start to ensure that my model could do something. I knew I wanted some non-linearity so I decided to switch some of the outputs to ReLu's. However I noticed that when switching a few activation layers to ReLu's I would get a model that produced a constant output regardless of the input (NOT GOOD). I knew this was a major issue but had no idea why it was happening.

By randomly commenting out sections of my model and then trying it, I was able to find that the mixture of dropout(0.5) and ReLu activation layers, specifically on the smaller of my dense layers, was causing the output to be constant. This does make sense as if there are only 16 neurons and you are randomly eliminating half of them every training, you aren't going to learn very much. What I still don't understand is how having a model with dropout(0.5) on every layer but using linear activations instead of ReLu's produces a non-constant output.

I ended up finding that if I made the second layer of my CNN linear but the rest ReLu, I could have dropouts on every layer except my last three and still generate a good output.

I tried to cram as many dropouts as possible since I found early on that not having dropouts the model would greatly over fit and make lots of sudden jerky turns when running on the simulator. This jerkiness was greatly reduced by using dropouts.


The full model code can be seen in either the model.py (just an export of notebook) or the notebook html (I'd prefer you look at this).




## Model Training

For all of the training I randomly selected 15% of my 11,224 image training data to be split off to be used for validation data at each epoch. I also had recorded in a different simulator session an entire lap (821 images) to be used for test data.


When training the model I supplied in this submission, I first used a Adam optimizer with a lower than default learning rate (lr=0.0001), as I found that with the default learning rate, the difference from one epoch to the next was still very large even when it seemed as though the solution had converged. Switching to this lower learning rate greatly reduced this effect. I ran this lower learning rate for 30 epochs. An edited version of the first 30 epochs can be seen below.

Epoch 1/30
9540/9540 [==============================] - 16s - loss: 0.0194 - mean_absolute_error: 0.0858 - val_loss: 0.0880 - val_mean_absolute_error: 0.2869

Epoch 2/30
9540/9540 [==============================] - 15s - loss: 0.0074 - mean_absolute_error: 0.0569 - val_loss: 0.0697 - val_mean_absolute_error: 0.2539

Epoch 3/30
9540/9540 [==============================] - 15s - loss: 0.0064 - mean_absolute_error: 0.0535 - val_loss: 0.0404 - val_mean_absolute_error: 0.1903

Epoch 4/30
9540/9540 [==============================] - 15s - loss: 0.0043 - mean_absolute_error: 0.0450 - val_loss: 0.0139 - val_mean_absolute_error: 0.1037

Epoch 5/30
9540/9540 [==============================] - 15s - loss: 0.0032 - mean_absolute_error: 0.0393 - val_loss: 0.0062 - val_mean_absolute_error: 0.0595

Epoch 6/30
9540/9540 [==============================] - 15s - loss: 0.0029 - mean_absolute_error: 0.0368 - val_loss: 0.0041 - val_mean_absolute_error: 0.0399

Epoch 7/30
9540/9540 [==============================] - 15s - loss: 0.0028 - mean_absolute_error: 0.0354 - val_loss: 0.0032 - val_mean_absolute_error: 0.0334

...


Epoch 26/30
9540/9540 [==============================] - 15s - loss: 0.0018 - mean_absolute_error: 0.0269 - val_loss: 0.0020 - val_mean_absolute_error: 0.0260

Epoch 27/30
9540/9540 [==============================] - 15s - loss: 0.0018 - mean_absolute_error: 0.0267 - val_loss: 0.0018 - val_mean_absolute_error: 0.0252

Epoch 28/30
9540/9540 [==============================] - 15s - loss: 0.0018 - mean_absolute_error: 0.0272 - val_loss: 0.0020 - val_mean_absolute_error: 0.0274

Epoch 29/30
9540/9540 [==============================] - 15s - loss: 0.0017 - mean_absolute_error: 0.0264 - val_loss: 0.0019 - val_mean_absolute_error: 0.0250

Epoch 30/30
9540/9540 [==============================] - 16s - loss: 0.0017 - mean_absolute_error: 0.0261 - val_loss: 0.0018 - val_mean_absolute_error: 0.0243

Done.

6.7 minutes.

### Fine Tuning

I still felt I could go further though so I decided to drop the learning rate (to 0.00005) and continue training. After another 15 epochs I got a result I was satisfied with. I decided this by not only looking at the loss (mean squared error) and mean absolute error but also looking at a graph of the model output over layed onto both the training and test data, all of which can be seen below.


![alt text](45.png "Title")



## Simulator

The model I trained can do infinitely number of laps at either 0.2, 0.25, or 0.3 throttle settings. However at 0.3 its execution on the first corner is not as smooth as I like (even though it doesn't go on to the concrete curve), so I decided to submit the drive.py with the 0.25 throttle setting.

When testing my model please use resolution of 800x600 (even though I don't think this matters) and graphic quality good (this does matter, please use this setting as I haven't tested it on anything else).


