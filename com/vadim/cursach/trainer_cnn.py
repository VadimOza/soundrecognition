from sklearn.preprocessing import LabelEncoder
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
from audioread import NoBackendError
import librosa.display
import numpy as np
import pandas as pd
import random

import com.vadim.cursach.constants as const
from com.vadim.cursach.store import store

train_folder = "../../../input/train/Train"
train_df = pd.read_csv('../../../input/train/train.csv')
train_df['file'] = train_df['ID'].apply(lambda x: train_folder + '/' + str(x) + '.wav')
test_folder = "../../../input/test/Test"
test_df = pd.read_csv('../../../input/test/test.csv')
test_df['file'] = test_df['ID'].apply(lambda x: test_folder + '/' + str(x) + '.wav')

labelEncoder = LabelEncoder()
train_df['classID'] = labelEncoder.fit_transform(train_df['Class'])
train_df['Class'].describe()

train = []
test = []

for row in train_df.itertuples():
    try:
        y, sr = librosa.load(row.file, duration=2.97)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 128):
            continue
        train.append((ps, row.classID))
    except NoBackendError:
        pass

for row in train_df.itertuples():
    try:
        y, sr = librosa.load(row.file, duration=2.97)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 128):
            continue
        test.append((ps, row.classID))
    except NoBackendError:
        pass

librosa.display.specshow(train[0][0], y_axis='mel', x_axis='time')

random.shuffle(train)
random.shuffle(test)

X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

# Reshape for CNN input
X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])

# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, 10))
y_test = np.array(keras.utils.to_categorical(y_test, 10))

model = Sequential()
input_shape=(128, 128, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=['accuracy'])

model.fit(
    x=X_train,
    y=y_train,
    epochs=12,
    batch_size=128,
    validation_data=(X_test, y_test))

score = model.evaluate(
    x=X_test,
    y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

store(model)

# predict_probs = model.predict_generator(predict_generator(test_df['file'], batch_size),
#                                         steps=steps_per_epoch(len(test_df), batch_size))
# predicts = np.argmax(predict_probs, axis=1)
# out_df = test_df[['ID']]
# out_df['Class'] = labelEncoder.inverse_transform(predicts)
# out_df.to_csv('submission.csv')
#
# out_df.head(10)
#
# for data in next(predict_generator(test_df['file'], 10)):
#     ipd.display(ipd.Audio(data.flatten(), rate=RATE))
