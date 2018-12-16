import com.vadim.cursach.store as store
import pandas as pd
import com.vadim.cursach.generator as gen
import com.vadim.cursach.recorder as rec
import numpy as np
from sklearn.preprocessing import LabelEncoder

test_folder = "../../../input/test/Test"
test_df = pd.read_csv('../../../input/test/test.csv')
train_df = pd.read_csv('../../../input/train/train.csv')
test_df['file'] = test_df['ID'].apply(lambda x: test_folder + '/' + str(x) + '.wav')
labelEncoder = LabelEncoder()
train_df['Class_id'] = labelEncoder.fit_transform(train_df['Class'])
model = store.load()

# print(model)

batch_size = 100

data, rate = rec.record(5)

predicted_probs = model.predict_generator(gen.predict_audio_generator(data, rate),
                                          steps=gen.steps_per_epoch(1, batch_size))

import sounddevice as sd
sd.play(data, rate, blocking=True)

# predicted_probs = model.predict_generator(
#     gen.predict_generator(['../../../input/test/custom/gun_shoot.wav'], batch_size),
#     steps=gen.steps_per_epoch(1, batch_size))

predicts = np.argmax(predicted_probs, axis=1)
print(labelEncoder.inverse_transform(predicts))

# predict_probs = model.predict_generator(gen.predict_generator(test_df['file'], batch_size),
#                                         steps=gen.steps_per_epoch(len(test_df), batch_size))
# predicts = np.argmax(predict_probs, axis=1)
# out_df = test_df[['ID']]
# out_df['Class'] = labelEncoder.inverse_transform(predicts)
# out_df.to_csv('submission.csv')
#
# print(out_df.head(10))
