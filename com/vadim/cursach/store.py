import keras

saveLocation = "../../../save/model.hdf5"


def store(model):
    model.save(saveLocation)


N_CLASSES = 10
RATE = 8000
CHANNELS = 1
LENGTH = 4
SAMPLES = RATE * LENGTH


def load():
    # model = keras.models.Sequential()

    # model.add(keras.layers.InputLayer((SAMPLES, CHANNELS)))
    # for n, k, s in ((30, 25, 5), (50, 19, 5), (100, 19, 5), (100, 19, 4), (100, 19, 4), (100, 15, 4), (100, 7, 4)):
    #     model.add(keras.layers.Conv1D(n, kernel_size=k, strides=s, padding='same'))
    #     model.add(keras.layers.LeakyReLU())
    #     model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(N_CLASSES, activation='softmax'))
    # model.load_weights(keras.models.load_model(saveLocation))

    return keras.models.load_model(saveLocation)
