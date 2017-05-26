#**Behavioral Cloning** 



https://youtu.be/6H3FGSvtQ8k
https://youtu.be/ZDEl4v90jpQ


[//]: # (Image References)

[image1]: ./examples/CNNView.jpg "What My Model Sees"
[image2]: ./examples/ToyCar.jpg "BehavioralCloning on WiFi RC Car"

---

###Model Architecture and Training Strategy

This is a writeup for the Udacity "Self-Driving" - Behvaioral Cloning project.
On top of that, i also applyied the same technique and trained a wi-fi remote toy-car, using a raspberryPi to get webcam image values and input joyStick controls.

The videos for both performances can be found here : 

Udacity Track1 (simulator view + CNN view)
https://youtu.be/6H3FGSvtQ8k
https://youtu.be/ZDEl4v90jpQ

ToyCar RaspberryPi BehavioralCloning Obstacle/Wall detection : 
https://youtu.be/nN1Qf_X0rr4

Visualisation of CNN input after pre-processing : 
![alt text][image1]

ToyCar Environment : 
![alt text][image2]

####1. NVidia Model

After building the basic running pipeline for this project, I deployed the NVidia model (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and started tweaking parameters for experimentation. The car was driving pretty well just after my first two recording runs.

Keras Lambda layer is used for normalisation of input images.
Dropout(0.5 learning_rate) after the first pyramidal convolutions is used to reduce overfitting.
Elu layer replaced Relu for activation functions (suggested better performance and faster gradient descent)

I do not keep training data in memory, instead use a custom batch_generator for both validation and training, with random,on the fly,augmentation. Testing was done directly on the simulator with the best val_loss model. 

Adam optimizer is used along with a starting lr or 1e-3, observed to reach the same performance as smaller values, but with faster gradient descent.


###Model Architecture and Training Strategy

####2. Data-gathering

For my first try i just drove the car around the first track, without keeping it perfectly alligned to the road-center. The basic model and this data performed pretty poorly (kept within a straight line but did not handle corners)

I then gathered more "sanitised" data (dead center on the road) and some recovery cases (sideway swirl within 1 lap of record). I also drove a couple of laps on the jungle track and reversed the track-direction for both tracks, 1 lap. With the NVidia Model attached this already performed well in autonomous mode (finishing 1st track but failing 2nd track)

####3. Model and variation
 
Final Keras model structure : 

model = Sequential([
    Lambda(lambda x: x/255.0-0.5, input_shape=SHAPE),
    Conv2D(32,5,5, border_mode='same',activation='elu', subsample=(2, 2)),
    Conv2D(48,5,5, border_mode='same',activation='elu', subsample=(2, 2)),
    Conv2D(64,5,5, border_mode='same',activation='elu', subsample=(2, 2)),
    Conv2D(64, 5, 5, activation='elu'),
    Dropout(0.5),
    Flatten(),
    Dense(256, activation='elu'),
    Dense(128, activation='elu'),
    Dense(64,activation='elu'),
    Dense(1, name='regressSteer'),
    ])

I experimented with different dimensions for the FC Dense layers and also with adding More convolution, but the pre-defined NVidia model seemed pretty optimal for this situation. 

I tryied regressing both steer and throttle values (with Dense(2) - single 'mse' loss') with interesting results if post-thresholding applyed (basically turning the regresion into a forward-backward classification). The regression throttle results were keeping the car into more of a standstill (probably because it registered 0 whenever i tried to drive slowly for proper steer examples), never reaching max value.

I did not keep this implementation because my data-gathering strategy did not take it into consideration and i consider i could have done better from this perspective. However, for the Toy-RC car the following observed behavior was very interesting : 
* running into perpendicular obstacles/walls when training data was flipped (-1 hard left 1 hard right values) resulted into a min loss value of around 0, making the car susceptiple to full perpendicular contact. (if it attacked a wall/obstacle at an angle, the prediction was spot on)
* i regressed throttle values with a strategy of (1 full throttle for general case and -1 backward throttle for corner cases like perpendicular wall in proximity)
* this resulted in a toy car that actually rectified hitting a perpendicular wall by breaking and even going backwards when in close proximity -VERY Exciting stuff


####4. Creation of the Training Set & Training Process

For the RC car i experimented further with the training set by different data-gathering approach : 
* I would let a "naive" model drive the to-car around and when i corrected (gave joyStick input of steer or throttle) i would gather those frame/label tuples into a new dataRetrigger collection db. This way i had finer control on the cases where it performed poorly

The data was randomly split and augmented (flipping with steering iversion | center,left,right camera usage with +-2 steering residuals | cropping, reshaping, brightness augmentation)

Some blog posts from Udacity presenting cool projects (https://medium.com/udacity/how-udacitys-self-driving-car-students-approach-behavioral-cloning-5ffbfd2979e5) inspired my to use shadow and jitter augmentation, as well as data-smoothing, but i was too late with the delivery to go on any further.
