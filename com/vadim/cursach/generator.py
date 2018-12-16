import numpy as np
from scipy.signal import decimate
import soundfile as sf
import math

import com.vadim.cursach.constants as const


def steps_per_epoch(total, batch):
    return math.ceil(total / batch)


def predict_audio_generator(data, rate):
    signals = []
    data = proc_sound(data, rate)
    signals.append(data)
    yield np.array(signals)


def predict_generator(files, batch_size):
    while True:
        for i in range(0, len(files), batch_size):
            signals = []
            for j in range(i, min(len(files), i + batch_size)):
                file = files[j]
                data, rate = sf.read(file, dtype='float32')
                data = proc_sound(data, rate)
                signals.append(data)
            yield np.array(signals)


def proc_sound(data, rate):
    data = decimate(data, rate // const.RATE, axis=0)
    if 2 == len(data.shape):
        data = np.sum(data, axis=1)
    pad = const.SAMPLES - len(data)
    if pad > 0:
        data = np.pad(data, ((0, pad)), mode='wrap')
    else:
        data = data[:const.SAMPLES]
    return data.reshape((-1, 1))
