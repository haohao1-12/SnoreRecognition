import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from sklearn import preprocessing
import python_speech_features as mfcc

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""
 
    rows,cols = array.shape
    deltas = np.zeros((rows,cols))
    N = 2
    for t in range(rows): ## First layer is for each frame
        index = []
        n = 1
        while n <= N: ## Second layer is for the summation of the equation
            if t-n < 0:
                first = 0
            else:
                first = t-n
            if t+n > rows -1:
                second = rows -1
            else:
                second = t+n
            index.append((second,first))
            n+=1
        deltas[t] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas
 
def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines
    delta to make it 40 dim feature vector"""   
 
    mfcc_feat = mfcc.mfcc(audio,rate, 0.025, 0.01,25,appendEnergy = True)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    delta = calculate_delta(mfcc_feat)
    combined = np.hstack((mfcc_feat,delta))
    return combined