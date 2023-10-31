import os
# Set warning suppressions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Tensorflow import
import tensorflow as tf
#Numpy import
import numpy as np
#Matplotlib import
import matplotlib as mpl
import matplotlib.pyplot as plt
#Math import
import math

import sys

import random
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

import shutil
tfk = tf.keras
tfkl = tf.keras.layers

print(tf.__version__)

# Random seed for reproducibility

seed = 22 #Choose a fixed seed to have reproducible results

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

shutil.unpack_archive("data.zip", "dataset")

# Loading .npz files

auth_class = int(sys.argv[1])
train_size = int(sys.argv[2])

print("Testing with speaker id: " + str(auth_class) + " and train size: " + str(train_size))

train_dir = "dataset/output/train_" + sys.argv[1] + "_" + sys.argv[2] + ".npz"
training_npz = np.load(train_dir)
x_train = training_npz['data']

val_dir = "dataset/output/validation.npz"
validation_npz = np.load(val_dir)
x_val, y_val = validation_npz['data'], validation_npz['labels']

testing_dir = "dataset/output/testing.npz"
testing_npz = np.load(testing_dir)
x_test, y_test = testing_npz['data'], testing_npz['labels']

d_vector_size = 256 ## This is the size of the output of the Feature Extractor.

def compute_mean(d_vector_matrix):
  mean_d_vector = np.zeros(d_vector_size)
  for i in range(d_vector_size):
    value = 0
    for j in range(len(d_vector_matrix)):
      value = value + d_vector_matrix[j][i]
    value = value / (len(d_vector_matrix))
    mean_d_vector[i] = value
  return mean_d_vector


d_vector_model_name = "models/d-vector-extractor-256.h5"

dv_model = tfk.models.load_model(d_vector_model_name)

optimizer = tfk.optimizers.Adam(learning_rate=0.0001) 

# Compile the model
dv_model.compile(loss=tfk.losses.CategoricalCrossentropy(), 
                optimizer=optimizer, 
                metrics='accuracy')

dv_model.summary()

train_data = x_train[0:].reshape((train_size,49,40,1))
d_vectors = dv_model.predict(train_data)

mean_d_vector = compute_mean(d_vectors)

print("Mean D-Vector computed using " + str(len(train_data)) + " sample.")

def cosine_similarity(vec1, vec2):
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  return dot_product / (norm_vec1 * norm_vec2)

def predictDVector(d_vector,authlabel,input_data, input_labels, threshold, verbose=True):
  input_vectors = dv_model.predict(input_data)
  total = len(input_vectors)
  total_auth = 0
  total_denied = 0

  for i in range(len(input_labels)):
    if(input_labels[i]!=auth_class):
      total_denied = total_denied+1
    else:
      total_auth = total_auth + 1

  correct_auth=0
  correct_denied=0

  for i in range(len(input_vectors)):
    similarity=cosine_similarity(input_vectors[i], d_vector)
    result = " -- ERROR!"
    if(similarity>threshold and input_labels[i] == authlabel):
      correct_auth = correct_auth + 1
      result = ""
    if(similarity<=threshold and input_labels[i] != authlabel):
      correct_denied = correct_denied + 1
      result = ""
    if(verbose):
      print("similarity: " + str(similarity) + " --- Class: " + str(input_labels[i]) + " " + result)
  correct = correct_auth + correct_denied

  print('-----------------------')
  print(" --- Testing Results ---")
  true_positive = correct_auth
  false_positive = total_denied - correct_denied
  false_negative = total_auth - correct_auth
  prec = true_positive / (true_positive + false_positive)
  recall = true_positive / (true_positive + false_negative)
  f1score =  2 * prec*recall/(prec + recall)

  print("True Positive Rate: " + str(correct_auth) + "/" + str(total_auth) + " (" + str(correct_auth*100/total_auth) + "%)")
  print("False Positive Rate: " + str(false_positive) + "/" + str(total_denied) + " (" + str((false_positive)*100/total_denied) + "%)")
  print('******************')
  print("Total correct " + str(correct) + "/" + str(total))
  print("Accuracy on this dataset: " + str(correct/total))
  print("F1-Score on this dataset: " + str(f1score))

  return correct/total, f1score #returns the toal accuracy on the dataset

###############################################
def provide_predictions(d_vector, input_data):
  y_predictions_prob = np.zeros((len(input_data), 1))
  input_vectors = dv_model.predict(input_data)
  for i in range(len(input_vectors)):
    similarity=cosine_similarity(input_vectors[i], d_vector)
    y_predictions_prob[i] = similarity
  return y_predictions_prob #returns an array with the similarity values of the input data with the given d_vector

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# binary classifier predictions
y_pred_prob = provide_predictions(mean_d_vector, x_val)

# classes for testing
y_val_bin = np.ones((len(y_val), 1))

for i,classvalue in enumerate(y_val):
  if(classvalue!=auth_class):
    y_val_bin[i] = 0

    
# ROC curve
fpr, tpr, thresholds = roc_curve(y_val_bin, y_pred_prob)

roc_auc = auc(fpr, tpr)

'''
print("-----")
print("Plotting the Receiving Operating Characteristic curve:")

# Plot ROC curve
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''

# Finding the EER threshold
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
print("-----")
print("EER Threshold: ", eer_threshold)
abs_diffs = np.abs(fpr - fnr)
min_index = np.argmin(abs_diffs)
EER = np.mean((fpr[min_index], fnr[min_index]))
              
print("EER = " + str(EER))
print("AUC = " + str(roc_auc))


acc, f1score = predictDVector(mean_d_vector, auth_class, x_test, y_test, threshold=eer_threshold, verbose=False)

with open("test-results-td-meancos.txt", "a") as f:
    f.write("Speaker " + str(auth_class) + " - Enrollment samples: " + str(train_size) + ":\n")
    f.write("Accuracy: " + str(acc) + " - F1Score: " + str(f1score) + " - EER: " + str(EER) + " - AUC: " + str(roc_auc) + "\n")
    f.close