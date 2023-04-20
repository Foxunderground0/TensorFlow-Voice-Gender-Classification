import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras.layers as L
from keras import optimizers, losses, metrics, Model
from keras.callbacks import EarlyStopping
import efficientnet.tfkeras as efn
from keras.models import Sequential, load_model

inputPath = "Processed/"

commands = np.array(tf.io.gfile.listdir(str(inputPath)))
print(commands)
N_CLASSES = len(commands)
print(N_CLASSES)
IMG_SIZE = 224  # 600


def get_waveform_and_label(filename):
    label = tf.strings.split(filename, "/")[-2]
    label = tf.argmax(label == commands)
    label = tf.one_hot(label, N_CLASSES)
    audio_binary = tf.io.read_file(filename)
    audio, _ = tf.audio.decode_wav(audio_binary)
    waveform = audio
    return waveform, label


def prepare_sample(spectrogram, label):
    HEIGHT, WIDTH = 128, 128
    a = spectrogram
    tf.print(a)
    y = np.array([])
    b = False
    i = 0
    for z in a:
        tf.print(z)
        z = tf.image.resize(images=z, size=[224, 224])
        z = tf.image.grayscale_to_rgb(z)
        print(y.shape)
        if b:
            z = np.expand_dims(z, axis=0)
            y = np.append(y, z, axis=0)
        else:
            y = np.array(z)
            y = np.expand_dims(y, axis=0)
            b = True
    tf.print(y)
    return y, label


def load_dataset(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    return dataset


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.299, test_split=0.001, shuffle=False, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def get_spectrogram(waveform, label):
    waveform = tf.cast(waveform, tf.float64)
    waveform = tf.squeeze(waveform)
    spectrogram = tf.signal.stft(
        waveform, frame_length=512, frame_step=32,  fft_length=8000)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    print("cum")

    return spectrogram, label


def get_dataset_old(fileNames, batch_size=32):

    label = np.array([])
    start = False

    # Get Labels
    for fileName in fileNames:
        toAdd = tf.strings.split(fileName, "/")[-2]
        toAdd = tf.argmax(toAdd == commands)
        toAdd = tf.one_hot(toAdd, N_CLASSES)
        print(toAdd)
        if start:
            toAdd = np.expand_dims(toAdd, axis=0)
            label = np.append(label, toAdd, axis=0)
        else:
            label = np.array(toAdd)
            label = np.expand_dims(toAdd, axis=0)
            start = True

    spectrograms = np.array([])
    start = False

    for fileName in fileNames:

        audio_binary = tf.io.read_file(fileName)
        audio, _ = tf.audio.decode_wav(audio_binary)

        waveform = tf.cast(audio, tf.float64)
        waveform = tf.squeeze(waveform)
        spectrogram = tf.signal.stft(
            waveform, frame_length=512, frame_step=32,  fft_length=8000)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=-1)

        spectrogram = tf.image.resize(spectrogram, [IMG_SIZE, IMG_SIZE])
        spectrogram = tf.image.grayscale_to_rgb(spectrogram)

        toAdd = spectrogram

        print(len(spectrograms))
        if start:
            toAdd = np.expand_dims(toAdd, axis=0)
            spectrograms = np.append(spectrograms, toAdd, axis=0)
        else:
            spectrograms = np.array(spectrogram)
            spectrograms = np.expand_dims(spectrograms, axis=0)
            start = True

    return spectrograms, label


def get_dataset(fileNames, batch_size=100):  # 256

    AUTO = tf.data.AUTOTUNE
    label = np.array([])
    start = False

    dataset = load_dataset(fileNames)
    dataset = dataset.shuffle(150000)  # dataset_len
    dataset = dataset.map(get_waveform_and_label, num_parallel_calls=AUTO)
    dataset = dataset.map(get_spectrogram,  num_parallel_calls=AUTO)
    dataset_len = tf.data.experimental.cardinality(dataset).numpy()

    origional = dataset
    dataset = origional.take(int((0.95*dataset_len)))
    val_ds = origional.skip(int(0.95*dataset_len))

    dataset = dataset.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    dataset.repeat()
    dataset = dataset.prefetch(AUTO)

    return dataset, val_ds


filesArray = []
for path, subdirs, files in os.walk(inputPath):
    for name in files:
        filesArray.append(os.path.join(path + "/", name))


dataset, val_ds = get_dataset(filesArray)


effnet = efn.EfficientNetB0(weights="imagenet",
                            include_top=False,
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
effnet.trainable = False


model = Sequential()
model.add(L.Input((235, 4001, 1), name='input_audio'))
model.add(L.Lambda(lambda image: tf.image.grayscale_to_rgb(
    tf.image.resize(image, [IMG_SIZE, IMG_SIZE]))))
model.add(effnet)

# Rebuild Top
model.add(L.GlobalAveragePooling2D())
model.add(L.BatchNormalization())
model.add(L.Dropout(0.2))
model.add(L.Dense(N_CLASSES, activation="softmax"))

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.000005),
              loss=losses.CategoricalCrossentropy(),
              metrics=[metrics.CategoricalAccuracy()])

print(model.summary())

model.fit(dataset,
          epochs=10,
          validation_data=val_ds)

model.save("Model4")
