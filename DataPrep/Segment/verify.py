import winsound
import librosa
import soundfile as sf
import numpy as np
import os

inputPath = "DataPrep/Segment/SplitProcessed/males/"

dirList = os.listdir(inputPath)

for fileName in dirList:
    if fileName.endswith(".wav"):
        filePath = inputPath + fileName
        samples, sampleRate = librosa.load(filePath)
        samples = librosa.to_mono(samples)

        if (sampleRate != samples.size):
            print(filePath)
