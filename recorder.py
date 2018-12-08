import pyaudio
import numpy as np
from matplotlib import pyplot as plt

CHUNKSIZE = 1024 # fixed chunk size

def record(seconds):

    # initialize portaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=CHUNKSIZE)

    numpydata = []

    print(" * Recording")

    for i in range(int(44100 / CHUNKSIZE * seconds)):
        data = stream.read(CHUNKSIZE)
        numpydata = np.append(numpydata, np.fromstring(data, dtype=np.int16))

    print(" * End of record")

    # plot data
    # plt.plot(numpydata)
    # plt.show()

    # close stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    return numpydata
