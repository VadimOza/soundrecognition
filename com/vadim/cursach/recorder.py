import sounddevice as sd

CHUNKSIZE = 1024  # fixed chunk size
RATE = 44100


def record(seconds):
    # initialize portaudio
    # p = pyaudio.PyAudio()
    # stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    # array = []

    # print(" * Recording")

    # for i in range(int(RATE / CHUNKSIZE * seconds)):
    #     data = stream.read(CHUNKSIZE)
    #     array = np.append(array, np.fromstring(data, dtype=np.int16))

    # print(" * End of record")

    # plot data
    # plt.plot(array)
    # plt.show()

    # close stream
    # stream.stop_stream()
    # stream.close()
    # p.terminate()

    # plot data
    # plt.plot(array)
    # plt.show()

    print("--------Recording-----------")
    records = sd.rec(frames=int(seconds * RATE), samplerate=RATE, channels=1)
    sd.wait()
    print("--------Recording-----------")

    return records, RATE

# record(5)
# import time
# import sounddevice as sd
#
# records = sd.rec(frames=int(2 * 44100), samplerate=44100, channels=1)
# sd.wait()
# sd.play(records, 44100, blocking=True)
#
# time.sleep(5)
