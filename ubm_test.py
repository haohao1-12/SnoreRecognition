from copyreg import pickle
import os
import numpy as np
from scipy.io.wavfile import read
from feature import extract_features
import warnings
import librosa
warnings.filterwarnings('ignore')
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

counter = 0
all = 0
modelpath = 'speaker_models_ubm2\\'
test_file = 'testdata1.txt'
file_path = open(test_file,'r')

x_axis = [i for i in range(72)]
y_axis = [j for j in range(72)]
heat_data = np.zeros((72,72))

gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

## Load GMM model
models = [pickle.load(open(fname,'rb')) for fname in gmm_files]
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

## Read the test directory and get the list of test audio files
for path in file_path:

    path = path.strip()
    #print(path)
    audio1, sr1 = librosa.load(path,sr=None)
    audio2 = librosa.resample(audio1,orig_sr=sr1,target_sr=sr1/2)

    vector1 = extract_features(audio1, sr1)
    #vector2 = extract_features(audio2, sr1)
    vector = vector1

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i] #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    heat_data[all] = log_likelihood
    winner = np.argmax(log_likelihood)
   
    all+=1
    if int(path[10:13]) != int(speakers[winner]):
        counter+=1
        print(path + " detected as - ", speakers[winner])

#print(all)
accuracy = (all-counter)/all
print('Accuracy: ', accuracy)

sns.heatmap(heat_data, cmap='Blues')
plt.show()
