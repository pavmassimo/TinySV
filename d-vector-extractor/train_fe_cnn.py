import os
# Set warning suppressions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import numpy as np
import tensorflow as tf
#Matplotlib import
import matplotlib as mpl
import matplotlib.pyplot as plt
#Math import
import math

import subprocess
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

import shutil
tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

dataset_path = "librispeech-train-100-clean-mfcc.npz"
dataset = np.load(dataset_path)

# Convert the npz to a dictionary

dataset = dict(dataset)

# Access the data in the dictionary
samples = dataset['arr_0']
classes = dataset['arr_1']

# Convert class indexes to integers:
classes.astype(int)

# Print the shapes of the data arrays
print(samples.shape)
print(classes.shape)

# Find the number of classes
# Find the unique elements in the array
classes_list = np.unique(classes)
classes.astype(int)

# Print the number of unique elements
print("Number of classes: " + str(len(classes_list)))

# Count the number of samples for each class
counts = np.unique(classes, return_counts=True)[1]

# Filter out the classes with fewer samples
threshold = 1448  # Keep only classes with at least 1455 samples
keep_classes = np.where(counts >= threshold)[0]

print("There are %d classes with more than %d samples." % (len(keep_classes), threshold))
print("Discarding classes with fewer samples...")

filtered_samples = []
filtered_classes = []

# Filter the samples and labels to include only the samples that belong to the keep_classes
for i in range(0, len(classes)):
  if(classes[i] in keep_classes):
    filtered_samples.append(samples[i])
    filtered_classes.append(classes[i])

filtered_samples = np.array(filtered_samples)
filtered_classes = np.array(filtered_classes)

# Count the number of samples for each remaining class
counts = np.unique(filtered_classes, return_counts=True)

## Last: adapt the numbering of classes from 0 to (number of classes)-1
for i in range(0, len(filtered_classes)):
  filtered_classes[i] = np.where(counts[0] == filtered_classes[i])[0]

# Count the number of samples for each remaining class
counts = np.unique(filtered_classes, return_counts=True)

print("New number of samples in dataset: %d" % (len(filtered_classes)))
for i in range(0, len(keep_classes)):
  print("Samples for class %d: %d" %(counts[0][i], counts[1][i]))

# Free up some memory or Google Colab will crash
del samples
del classes
del dataset

samples = filtered_samples
classes = filtered_classes

import sklearn
from sklearn.model_selection import train_test_split

train_percentage = 0.7
test_percentage = 0.15
val_percentage = 0.15

# Compute the correct percentages to be used in Sklearn Splitting code:
test_percentage = test_percentage + val_percentage
val_percentage = val_percentage / test_percentage

# Split the dataset into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(samples, classes, test_size=test_percentage)

print("Training set:")
print(X_train.shape, y_train.shape)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_percentage)

print("Validation set:")
print(X_val.shape, y_val.shape)
print("Testing set:")  
print(X_test.shape, y_test.shape)

# Random seed for reproducibility

seed = 22 #Choose a fixed seed to have reproducible results (22=Gonzales o Chiesa)

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

class DataGenerator(tfk.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, n_samples, batch_size, dim, n_channels,
                 n_classes, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        samples_list_temp = indexes
        # Generate data
        X, y = self.__data_generation(samples_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, samples_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, sample in enumerate(samples_list_temp):
            # Store sample
            mfcc = self.data[sample].reshape(49,40,1)
            X[i,] = mfcc
            # Store class
            y[i] = self.labels[sample]

        return X, tfk.utils.to_categorical(y, num_classes=self.n_classes)

batch_size = 32
n_classes = len(keep_classes)
spectrogram_size = (49,40,)
spectrogram_channels = 1

# Parameters
params = {'dim': spectrogram_size,
          'batch_size': batch_size,
          'n_classes': n_classes,
          'n_channels': spectrogram_channels,
          'shuffle': True}


# Generators
training_generator = DataGenerator(X_train, y_train, n_samples=len(y_train), **params)
validation_generator = DataGenerator(X_val, y_val, n_samples=len(y_val), **params)
testing_generator = DataGenerator(X_test, y_test, n_samples=len(y_test), **params)

example_spectrogram = training_generator.__getitem__(0)[0]
print("Neural Network input shape: " + str(example_spectrogram.shape))

input_shape = (*spectrogram_size, spectrogram_channels) #do not modify

# Assign the name you want to your model
model_name = 'cnn-librispeech-classifier'

# Build your model here:
def build_model(input_shape):

  input_layer = tfkl.Input(shape=input_shape, 
                           name='input')
  
  batchnorm = tfkl.BatchNormalization()(input_layer)
  
  conv_layer = tfkl.Conv2D(
              filters=8,
              kernel_size=(3, 3),
              strides = (1, 1),
              padding = 'same',
              activation='relu',
              input_shape=input_shape,
              use_bias = True,
              data_format = 'channels_last',
              kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(batchnorm)

  maxpool_layer = tfkl.MaxPooling2D(pool_size=(3, 3), padding="valid")(conv_layer)

  conv_layer_2 = tfkl.Conv2D(
              filters=16,
              kernel_size=(3, 3),
              strides = (1, 1),
              padding = 'same',
              activation='relu',
              input_shape=input_shape,
              use_bias = True,
              data_format = 'channels_last',
              kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(maxpool_layer)

  maxpool_layer_2 = tfkl.MaxPooling2D(pool_size=(2, 2), padding="valid")(conv_layer_2)

  conv_layer_3 = tfkl.Conv2D(
              filters=32,
              kernel_size=(3, 3),
              strides = (2, 2),
              padding = 'same',
              activation='relu',
              input_shape=input_shape,
              use_bias = True,
              data_format = 'channels_last',
              kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(maxpool_layer_2)

  conv_layer_4 = tfkl.Conv2D(
              filters=64,
              kernel_size=(3, 3),
              strides = (2, 2),
              padding = 'same',
              activation='relu',
              input_shape=input_shape,
              use_bias = True,
              data_format = 'channels_last',
              kernel_initializer = tfk.initializers.GlorotUniform(seed)
    )(conv_layer_3)

  flattening_layer = tfkl.Flatten()(conv_layer_4)

  fully_connected = tfkl.Dense(
                units=128, 
                activation='relu', 
                kernel_initializer=tfk.initializers.GlorotUniform(seed),
                use_bias = True, 
                name='fully_connected')(flattening_layer)

  dropout = tfkl.Dropout(rate=0.3)(fully_connected)

  output = output_layer = tfkl.Dense(
                    units=n_classes, 
                    activation='softmax', 
                    kernel_initializer=tfk.initializers.GlorotUniform(seed),
                    use_bias = True, 
                    name='output')(dropout)

  # Connect input and output through the Model class
  model = tfk.Model(inputs=input_layer, outputs=output_layer, name=model_name)

  optimizer = tfk.optimizers.Adam(learning_rate=0.0001)

  # Compile the model
  model.compile(loss=tfk.losses.CategoricalCrossentropy(), 
                optimizer=optimizer, 
                metrics='accuracy')

  # Return the model
  return model

model = build_model(input_shape)
model.summary()

# How many epochs?
# Train for a total of 700 epochs without early stopping at least.
epochs = 300

# Train the model
history = model.fit(
    x = training_generator,
    epochs = epochs,
    validation_data = validation_generator,
).history

model_metrics = model.evaluate(testing_generator, return_dict=True)

## Confusion Matrix Print ##

#Predict
y_prediction = model.predict(X_test)
y_prediction = np.argmax(y_prediction, axis = 1)

#Create confusion matrix and normalizes it over predicted (columns)
result = confusion_matrix(y_test, y_prediction , normalize=None)

import seaborn as sns
import matplotlib.pyplot as plt     

fig, ax = plt.subplots(figsize=(24,24))

sns.heatmap(result, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix - Testing Data'); 
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

model.save(os.path.join('models', model_name))

h5_model_name = model_name + '.h5'
tfk.models.save_model(model, os.path.join('models', h5_model_name))

### Extraction of the embeddings generator model ##

## Feature extractor part for D-Vector classification
fe_name = "d-vector-extractor-256"
fe_model = tfk.Sequential(name=fe_name)

for layer in model.layers[:-3]: # go through until last layer
    fe_model.add(layer)

fe_model.add(
    tfkl.Dropout(rate=0.5)
)

fe_model.summary()
fe_model.compile(optimizer='adam', loss='categorical_crossentropy')

### Saving the embeddings extractor model:

fe_model.save(os.path.join('models', fe_name))

h5_model_name = fe_name + '.h5'

tfk.models.save_model(fe_model, os.path.join('models', h5_model_name))