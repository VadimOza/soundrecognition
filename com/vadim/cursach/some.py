import numpy as np
import pandas as pd
import IPython.display as ipd
import math
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import soundfile as sf
import keras
import glob
import scipy
from scipy.signal import decimate
from scipy.io import wavfile
import seaborn as sns
import matplotlib.pyplot as plt

train_folder = "../input/train/Train"
train_df = pd.read_csv('../input/train/train.csv')
train_df['file'] = train_df['ID'].apply(lambda x: train_folder+'/'+str(x)+'.wav')
test_folder = "../input/test/Test"
test_df = pd.read_csv('../input/test/test.csv')
test_df['file'] = test_df['ID'].apply(lambda x: test_folder+'/'+str(x)+'.wav')