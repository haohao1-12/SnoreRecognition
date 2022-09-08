import pickle as cPickle
import numpy as np
from sklearn import mixture
from feature import extract_features
import warnings
import librosa
import copy
warnings.filterwarnings('ignore')

def map_adaptation(gmm, data, max_iterations = 300, likelihood_threshold = 1e-20, relevance_factor = 16):
    N = data.shape[0]
    D = data.shape[1]
    K = gmm.n_components
    
    mu_new = np.zeros((K,D))
    n_k = np.zeros((K,1))
    
    mu_k = gmm.means_
    cov_k = gmm.covariances_
    pi_k = gmm.weights_

    old_likelihood = gmm.score(data)
    new_likelihood = 0
    iterations = 0
    while(abs(old_likelihood - new_likelihood) > likelihood_threshold and iterations < max_iterations):
        iterations += 1
        old_likelihood = new_likelihood
        z_n_k = gmm.predict_proba(data)
        n_k = np.sum(z_n_k,axis = 0)

        for i in range(K):
            temp = np.zeros((1,D))
            for n in range(N):
                temp += z_n_k[n][i]*data[n,:]
            mu_new[i] = (1/n_k[i])*temp

        adaptation_coefficient = n_k/(n_k + relevance_factor)
        for k in range(K):
            mu_k[k] = (adaptation_coefficient[k] * mu_new[k]) + ((1 - adaptation_coefficient[k]) * mu_k[k])
        gmm.means_ = mu_k

        log_likelihood = gmm.score(data)
        new_likelihood = log_likelihood
        #print(log_likelihood)
    return gmm

## -------Load ubm model---------
path = './ubm_model/ubm.gmm'
ubm = cPickle.load((open(path,'rb')))
weights = ubm.weights_
means = ubm.means_
covariances = ubm.covariances_
## ------------------------------


dest = "speaker_models_ubm2\\"
train_file = "enrolldata1.txt"
file_paths = open(train_file,'r')
#print(file_paths)

count = 1
# Extracting features for each speakers (4 files per speaker)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    # read the audio

    audio1, sr1 = librosa.load(path,sr=None)

    #audio2 = librosa.resample(audio1,orig_sr=sr1,target_sr=sr1/2)
    #audio3 = librosa.resample(audio1,orig_sr=sr1,target_sr=sr1*2)

    # extract 40 dimensional MFCC & delta MFCC features
    vector1 = extract_features(audio1, sr1)
    #vector2 = extract_features(audio2, sr1)
    #vector3 = extract_features(audio3, sr1)

    #vector = np.vstack((vector1,vector2,vector3))
    vector = vector1


    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    # when features of 4 files of speaker are concatenated, then do model training
    if count == 4:
        gmm = map_adaptation(gmm=ubm,data=features)
        #gmm = mixture.GaussianMixture(n_components=16,n_init=3,covariance_type='diag',max_iter=200)
        #gmm.fit(features)
        

        # dumping the trained gaussian model
        '''print(type(gmm))'''
        picklefile = str(path)[10:13]+'.gmm'
        '''print(picklefile)'''

        cPickle.dump(gmm, open(dest+picklefile,'wb'))
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1