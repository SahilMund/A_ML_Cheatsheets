#importing the librariees
from keras.models import Sequential     #to initiate our neural network
from keras.layers import Convolution2D  #Used to add the convolutional layers
from keras.layers import MaxPooling2D   #Used to apply the Pooling step
from keras.layers import Flatten        #onverts the pooled feature map into the feature vector
from keras.layers import Dense          #Used to add fully connected layers in a classic ANN
#importing the CNN
classifier=Sequential()

#Step1:- Convolution
classifier.add(Convolution2D(32,3,3 ,input_shape=(64,64,3), activation='relu'))

#step2:- Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a second convolutional layer
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step3:- Flattening
classifier.add(Flatten())

#Step 4:- Full Connection
classifier.add(Dense(output_dim=128, activation='relu')) #This is the hidden layer
classifier.add(Dense(output_dim=1, activation='sigmoid')) #This is the output layer

#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Performinf Image augmentation and fitting our CNN to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)