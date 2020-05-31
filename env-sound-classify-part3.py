#------------------------------------------------------------------------------
# To run your web server, open up your terminal / command prompt
# and type:
#    cd <path to this file>
#    python env-sound-classify-part3.py
#
#------------------------------------------------------------------------------

from flask import Flask, flash, request, redirect, url_for, Response
import random
import numpy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm
import librosa
import os
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from numpy.random import rand
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

# Initialize our Flask app.
# NOTE: Flask is used to host our app on a web server, so that
# we can call its functions over HTTP/HTTPS.
#
app = Flask(__name__)


#------------------------------------------------------------------------------
# This section contains all our audio signal processing codes.
#------------------------------------------------------------------------------

# Each of our sample (22khz) lasts exactly 5 seconds with 22050 * 5 samples.
#
spec_hop_length = 512
mfcc_hop_length = 512
spec_max_frames = int(22050 * 5 / spec_hop_length) + 1 # this is actually about 22050 / 512.
mfcc_max_frames = int(22050 * 5 / mfcc_hop_length) + 1

print ("MFCC Frames (for 5 sec audio):     %d" % (mfcc_max_frames))
print ("Spectral Frames (for 5 sec audio): %d" % (spec_max_frames))


num_classes = 10
max_samples = 22050 * 5  # 5 seconds
max_mfcc_features = 40

# Scale the values to be between 
def scale(arr):
    #arr = arr - arr.mean()
    safe_max = np.abs(arr).max()
    if safe_max == 0:
        safe_max = 1
    arr = arr / safe_max
    return arr


# Load a file and convert its audio signal into a series of MFCC
# This will return a 2D numpy array.
#
def convert_mfcc(file_name):
    signal, sample_rate = librosa.load(file_name) 
    signal = librosa.util.normalize(signal)
    signal_trimmed, index = librosa.effects.trim(signal, top_db=60)
    signal_trimmed = librosa.util.fix_length(signal_trimmed, max_samples)
    
    feature = (librosa.feature.mfcc(y=signal_trimmed, sr=sample_rate, n_mfcc=max_mfcc_features).T)
    #print (feature.shape)
    if (feature.shape[0] > mfcc_max_frames):
        feature = feature[0:mfcc_max_frames, :]
    if (feature.shape[0] < mfcc_max_frames):
        feature = np.pad(feature, pad_width=((0, mfcc_max_frames - feature.shape[0]), (0,0)), mode='constant')
    
    # This removes the average component from the MFCC as it may not be meaningful.
    #
    feature[:,0] = 0
        
    feature = scale(feature)
    #print(feature)
    return feature


# Load a file and convert its audio signal into a spectrogram
# This will return a 2D numpy array.
#
def convert_spectral(file_name):
    signal, sample_rate = librosa.load(file_name) 
    signal = librosa.util.normalize(signal)
    signal_trimmed, index = librosa.effects.trim(signal, top_db=60)
    signal_trimmed = librosa.util.fix_length(signal_trimmed, max_samples)
    
    feature = np.abs(librosa.stft(y=signal_trimmed, hop_length=spec_hop_length, win_length=spec_hop_length*4, n_fft=spec_hop_length*4, center=False).T)

    if (feature.shape[0] > spec_max_frames):
        feature = feature[0:spec_max_frames, :]
    if (feature.shape[0] < spec_max_frames):
        feature = np.pad(feature, pad_width=((0, spec_max_frames - feature.shape[0]), (0,0)), mode='constant')
        
    feature = librosa.amplitude_to_db(feature)
    feature = cv2.resize(feature, (224, 224), interpolation = cv2.INTER_CUBIC)
    feature = scale(feature)
    #print(feature)

    return feature    


#------------------------------------------------------------------------------
# Now let's load up BOTH our Keras model here.
#
# You can change the names of the files if necessary
#------------------------------------------------------------------------------

# TODO:
# Load your 2 models here                                   

# ..................... CODES START HERE ..................... #
mfc_model = load_model(os.getcwd() + '/mfcc_model.h5')
spe_model = load_model(os.getcwd() + '/spec_model.h5')
# ..................... CODES START HERE ..................... #


#------------------------------------------------------------------------------
# This is our predict URL 
#------------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    
    # Extracts the file from the upload and save it into
    # a temporary file.
    #
    f = request.files['file']
    filename = "%s.wav" % ((random.randrange(0, 1000000000)))
    f.save(filename)

    # ..................... CODES START HERE ..................... #

    # Convert the audio features to MFCC and Spectrogram    
    #
    mfcc_feat = convert_mfcc(filename)
    mfcc_feat = mfcc_feat.reshape(1, mfcc_feat.shape[0], mfcc_feat.shape[1], 1)
    spec_feat = convert_spectral(filename)
    spec_feat = spec_feat.reshape(1, spec_feat.shape[0], spec_feat.shape[1], 1)

    # Predict outputs separate from each of the MFCC        
    # and Spectrogram Keras model. 
    #
    prediction_1 = mfc_model.predict(mfcc_feat)
    prediction_2 = spe_model.predict(spec_feat)

    # Combine the results from both predictions by
    # averaging them.                                       
    #
    final_prediction = (prediction_1+prediction_2)/2

    # Get the class index of the highest scoring class      
    # based on your final prediction
    #
    best_class = np.argmax(final_prediction)

    # ...................... CODES END HERE ...................... #

    result = json.dumps(
        { 
            # Place the class index with the highest probability into
            # the "best_class" attribute.
            #
            # The use of item() converts best_class (which is a numpy.int64 
            # data type) to a native Python int.
            #
            "best_class": best_class.item(), 

            # Return the full prediction from Keras.
            # Convert a Numpy array to a native Python list.
            #
            "full_prediction" : final_prediction.tolist()[0]
        })

    os.remove(filename)

    return Response(result, mimetype='application/json')                           


#------------------------------------------------------------------------------
# This starts our web server.
# You can test your app using Google Chrome's Postman (or using the 
# HTML app that we created in Practical 3)
#------------------------------------------------------------------------------
if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1',port=5005)


