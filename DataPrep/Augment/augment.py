import winsound
import librosa
import soundfile as sf
import numpy as np
from audiomentations import PitchShift, AddGaussianNoise, Normalize, RoomSimulator, Shift

path = "Lumos.wav"
probability = 1

transformRoom = RoomSimulator(
    min_absorption_value=0.08,
    max_absorption_value=1,
    p=probability
)

transformShift = Shift(
    rollover=False,
    fade=True,
    min_fraction=-0.15,
    max_fraction=0.15,
    p=probability
)

transformPitch = PitchShift(
    min_semitones=-2,
    max_semitones=4,
    p=probability
)

transformNoise = AddGaussianNoise(
    min_amplitude=0.005,
    max_amplitude=0.05,
    p=probability
)

transformNormalize = Normalize(
    p=1
)

samples, sampleRate = librosa.load(path)
samples = librosa.to_mono(samples)

print(samples)
print(np.shape(samples))

samples = transformNormalize(samples, sample_rate=sampleRate)


for i in range(0, 5):
    augmented_sound = transformPitch(samples, sample_rate=sampleRate)

    sf.write("audioTransformed" + str(i) + ".wav",
             augmented_sound, sampleRate, subtype='PCM_16')

for i in range(5, 10):
    augmented_sound = transformNoise(samples, sample_rate=sampleRate)

    sf.write("audioTransformed" + str(i) + ".wav",
             augmented_sound, sampleRate, subtype='PCM_16')

for i in range(10, 15):
    augmented_sound = transformRoom(samples, sample_rate=sampleRate)

    sf.write("audioTransformed" + str(i) + ".wav",
             augmented_sound, sampleRate, subtype='PCM_16')

for i in range(15, 20):
    augmented_sound = transformShift(samples, sample_rate=sampleRate)

    sf.write("audioTransformed" + str(i) + ".wav",
             augmented_sound, sampleRate, subtype='PCM_16')

probability = 0.30

for i in range(20, 35):
    augmented_sound = transformPitch(samples, sample_rate=sampleRate)
    augmented_sound = transformNoise(augmented_sound, sample_rate=sampleRate)
    augmented_sound = transformRoom(augmented_sound, sample_rate=sampleRate)
    augmented_sound = transformShift(augmented_sound, sample_rate=sampleRate)

    sf.write("audioTransformed" + str(i) + ".wav",
             augmented_sound, sampleRate, subtype='PCM_16')
