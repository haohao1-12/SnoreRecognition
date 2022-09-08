import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
from feature import extract_features
import warnings
import librosa
from ffmpeg import audio
warnings.filterwarnings('ignore')

#source = "./dataset"
dest = "ubm_model\\"
train_file = "UBMenrolldata.txt"
file_paths = open(train_file,'r')
#print(file_paths)

count = 1
# Extracting features for each speakers (4 files per speaker)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    # read the audio

    audio, sr = librosa.load(path,sr=None)

    #audio2 = librosa.resample(audio1,orig_sr=sr1,target_sr=sr1/2)
    #audio3 = librosa.resample(audio1,orig_sr=sr1,target_sr=sr1*2)

    # extract 50 dimensional MFCC & delta MFCC features
    vector = extract_features(audio, sr)
    #vector2 = extract_features(audio2, sr1)
    #vector3 = extract_features(audio3, sr1)

    #vector = np.vstack((vector1,vector2,vector3))


    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    # when features of 4 files of speaker are concatenated, then do model training
    if count == 738:
        gmm = mixture.GaussianMixture(n_components=17,n_init=3,covariance_type='diag',max_iter=200)
        gmm.fit(features)

        # dumping the trained ubm gaussian model
  
        picklefile = 'ubm.gmm'

        cPickle.dump(gmm, open(dest+picklefile,'wb'))
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        print(gmm.weights_)
        features = np.asarray(())
        # count = 0
    count = count + 1

