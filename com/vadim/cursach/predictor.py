from com.vadim.cursach.store import load
import keras

model = load()
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
#