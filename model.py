import tensorflow as tf
import numpy as np
import csv
import argparse
import os
import cv2
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D
from keras.layers import Dropout, Flatten, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# Desired shape after cropping
SHAPE=(66,200,3)
# Had multiple train-data sessions, each corresponding to its own data.csv and folder location
ImgDir = ["./trainData1","./trainData1-Curves","./trainData-Reverse","./trainData-bits","./TrainData","./TrainDataRev2","./TrainDataReverse"]
# Local Path
CurrentPath = "/home/ubuntu/BehavioralCloning"

# Crop dimensions for the sky and the bumper
cropTop=60
cropBottom=25

# angle correction for side cameras
angleDelta=0.2

np.random.seed(0)

# return cropped image
def crop(image):
	return image[cropTop:-cropBottom,:,:]

# return resized image
def resize(image):
	#print('Resize from: ',image.shape,' to : ',(SHAPE[1],SHAPE[0]))
	return cv2.resize(image,(SHAPE[1],SHAPE[0]))

# return yuv image as indicated by some NVidia model approaches
def rgb2yuv(image):
	return cv2.cvtColor(image,cv2.COLOR_RGB2YUV)

# return flipped image and opposing angle value
def flip(image, angle):
	return cv2.flip(image,1),angle*(-1)

# random brightness augmentation found online
def brightness(image):
	level=1.0 - (np.random.rand()-0.5)
	level=max(level,0.6)
	hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	hsv[:,:,2] = hsv[:,:,2]*level
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
	return rgb

# return random uniform sample from dataset.
# i think this can be done better. I-m not exactly sure how to randomly 
# sample the dataset and also make sure that by the end of the dataset size 
# all examples will be accounted for
def randomSample(x,y):
	
	idx = int(np.random.uniform(0,len(x)))
	#print('X : ',x)
	#print('Y : ',y)
	#print('selected : ',idx)
	return x[idx],y[idx]

# augment data randomly with brightness, flipping and camera selection (center/right/left)
def augment_data(image,angle):
	
	#print('Image:',image)
	#print('Angle:',angle)

	# random camera selection + angle augmentation
	coin=np.random.choice(3)
	if (coin==0):
		image = image[0]
	if (coin==1):
		image = image[1]
		angle += angleDelta
	if (coin==2):
		image = image[2]
		angle -= angleDelta
	
	# compute the path to image. Images were obtained locally,then organised in aws,
	# so i need both basename and 2 orders of parent directory in order to open the correct file
	# this is similar to what tutorials on Udacity provides
	tokens = image.split('/')
	TrainPath = './'+image.split('/')[-3]+'/'+image.split('/')[-2]+'/'+image.split('/')[-1]
	image = mpimg.imread(TrainPath)

	# Randomly flip and brightness augment the chosen camera angle with 0.5 probability
	coin=np.random.choice(1)
	if (coin): image,angle = flip(image,angle)
	coin=np.random.choice(1)
	if (coin): image = brightness(image)
	
	# return the newly augmented image for that frame (center/left or right)
	return image,angle


# batch generator for both training and validation
def batch_generator(train, x, y, batch_size):
	# allocate stubs
	images = np.empty([batch_size,66,200,3])
	angles = np.empty(batch_size)

	idx=0
	# Continuously generate batch_size data of random choices (for Train) and centered image (for Val)
	while True:
		idx=0
		while(idx<batch_size):
			image, angle = randomSample(x,y)
			if (train):
				# randmply augment this frame
				image,angle = augment_data(image,angle)
			else:
				# return only center frame for validation
				tokens = image[0].split('/')
				# create local path for image
				ValPath = tokens[-3]+'/'+tokens[-2]+'/'+tokens[-1]
				image = mpimg.imread(ValPath)
			# apply processing to the resulting single image structure
			image = crop(image)
			image = resize(image)
			image = rgb2yuv(image)
			#print(image.shape)
			# append image and label
			images[idx] = image
			angles[idx] = angle
			idx+=1
		#print('*'*40)
		#print(images)
		#print('YIELD SHAPE from batch generator:',images.shape,' ',angles.shape)
		yield images,angles


def read_dataset_and_split(testSize):
	# Read the dataset from all the different locations
	# Split according to testSize param
	imgs = []
	labels= []


	names = ['center', 'left', 'right','steering', 'throttle', 'brake', 'speed']
	for imgDir in ImgDir:
		dataPath = os.path.join(imgDir,'driving_log.csv')
		data = pd.read_csv(dataPath,names=names)
	
		imgs.extend(data[['center', 'left', 'right']].values)
		labels.extend(data['steering'].values)


	X_train, X_val, Y_train, Y_val = train_test_split(imgs,labels, test_size=testSize, random_state=0)
	return X_train, X_val, Y_train, Y_val

# started with NVIDIA model, changed some elements just for experimentation 
# elu was presented by mentor as a better replacement for relu
# Lambda layer for normalisation and single Dropout to prevent overfitting

def create_architecture(printArhitecture=True):
	# NVIDIA model

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

	if (printArhitecture):
		#print out the arhitecture
		model.summary()
	return model

# callback for model checkpoint. this will save to disk if validation loss improves
def checkpoint_callback():
	checkpoint = ModelCheckpoint ("model-{loss:.4f}.h5",
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='auto')
	return [checkpoint]

def main():
	
	# Parse Cmd Args
	parser = argparse.ArgumentParser()
	parser.add_argument('-epochs',    help='epoch number', 		dest='epochs', 		type=int, 	default=15)
	parser.add_argument('-batch_size',help='batch size',		dest='batch_size',	type=int, 	default=40)
	parser.add_argument('-test_split',help='x percent of data',	dest='test_split', 	type=float,	default=0.1)

	args=parser.parse_args()

    # read the data. Split the data. 
	X_train,X_val,Y_train,Y_val = read_dataset_and_split(args.test_split)
	
	# define the model arhitecture
	model = create_architecture()
	model.compile(loss='mse', optimizer=Adam(lr=0.0001))

	# train the model arhitecture with generated data
	model.fit_generator(batch_generator(True,X_train,Y_train,args.batch_size),
			len(X_train),
			args.epochs,
			validation_data=batch_generator(False,X_val,Y_val,args.batch_size),
			nb_val_samples=len(X_val),
			callbacks = checkpoint_callback(),
			verbose=1)

if __name__ == '__main__':
	main()
