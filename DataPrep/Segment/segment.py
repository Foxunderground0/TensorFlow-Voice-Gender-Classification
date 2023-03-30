import winsound
import librosa
import soundfile as sf
import numpy as np
import os

inputPath = "DataPrep/Segment/Processed/noise/"
outputPath = "DataPrep/Segment/Processed - Copy/noise/"

dirList = os.listdir(inputPath)

for fileName in dirList:
    if fileName.endswith(".wav"):
        filePath = inputPath + fileName
        samples, sampleRate = librosa.load(filePath)
        samples = librosa.to_mono(samples)

        print(sampleRate)
        print(np.shape(samples))
        print(samples.size)

        samples = librosa.resample(samples, orig_sr=sampleRate, target_sr=8000)

        sampleRate = 8000
        duration = 1

        print(sampleRate)
        print(np.shape(samples))
        print(samples.size)

        for i in range(int((samples.size/sampleRate)*(1/duration))):
            print(i)
            sf.write(outputPath + fileName[:-4] + "-" + str(i) + ".wav",
                     samples[int(i * sampleRate * duration): int((i+1) * sampleRate * duration)], sampleRate, subtype='PCM_16')
