#Import tensorlflow and keras libraries that are needed
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
tf.__version__

#the size of the image is assigned to the varibale size
size = 150

#the modifications for the images in the training set
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

#assign the variable training_set with the data stored from this path
training_set = train_datagen.flow_from_directory('4Class/training_set', target_size = (size, size), batch_size = 2, class_mode = 'categorical',)

#No modifications are performed on the test_set
test_datagen = ImageDataGenerator(rescale = 1./255)

#assign the variable test_set with the data stored from this path
test_set = test_datagen.flow_from_directory('4Class/test_set', target_size = (size, size), batch_size = 2, class_mode = 'categorical')

#Initialize the model with the Sequential model
model = tf.keras.models.Sequential()

#Convolutional Layer #1
model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=4, activation='relu', input_shape=[size, size,1]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Convolutional Layer #2
model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=4, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Array is Flattened
model.add(tf.keras.layers.Flatten())

#The number of nodes in the ANN
model.add(tf.keras.layers.Dense(units=32, activation='relu'))

#Output Nodes
model.add(tf.keras.layers.Dense(units=4, activation='softmax'))

#Optimizer is determined
opt = tf.keras.optimizers.Adamax()

#Compiling the model
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy', 'AUC', 'Precision', 'Recall', 'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives',])

#Early Stopping is assigned to the variable "es"
es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

#Print the Paramaters and Summary of the Model
print(model.summary())

#The model is fitted to the data
model.fit(x = training_set, validation_data = test_set, epochs = 100, verbose=2, callbacks=[es])

#The model is saved for further use with prediction or statistics                                               
model.save('saved_model/my_model_4_l') 
