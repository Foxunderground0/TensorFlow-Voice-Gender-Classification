import keyboard
import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf


import os
import re
import warnings
import random
import numpy as np
# import pandas as pd
# import seaborn as sns

from matplotlib import pyplot as plt
# from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras.layers as L
from keras import optimizers, losses, metrics, Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model

chunkSize = 8000
numberOfChunks = 0
maxNumberOfChunks = 1  # 2secconds is the time to record

IMG_SIZE = 224
model = load_model("Model5")


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1,
                rate=8000, input=True, frames_per_buffer=chunkSize)

print("Ready")
triggred = False

classes = ["Female", "Male", "Noise"]
while 1:
    numpydata = np.frombuffer(stream.read(chunkSize), dtype=np.single)

    waveform = numpydata
    waveform = tf.cast(waveform, tf.float64)
    waveform = tf.squeeze(waveform)
    spectrogram = tf.signal.stft(
        waveform, frame_length=512, frame_step=32,  fft_length=8000)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.expand_dims(spectrogram, axis=0)

    prediction = (model.predict(spectrogram, verbose=0,
                  use_multiprocessing=True)*100)
    print(classes[np.argmax(prediction)])
    print(prediction)


# numpydata = nr.reduce_noise(y=numpydata, sr=41100)

# close stream
stream.stop_stream()
stream.close()
p.terminate()


# print(model.summary())
