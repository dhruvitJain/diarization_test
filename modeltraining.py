import pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

#path to training data
source   = "trainingData/"   

#path where training speakers will be saved
dest = "Speakers_models/"
train_file = "trainingDataPath.txt"        
file_paths = open(train_file,'r')

count = 1
# Extracting features for each speaker (11 files per speakers)
features = np.asarray(())
for path in file_paths:    
    path = path.strip()   
    print (path)
    
    # read the audio
    sr,audio = read(source + path)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    if count == 11:    
        gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        bicc = gmm.bic (features)
        # dumping the trained gaussian model
        picklefile = path.split("-")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,mode='wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        print (f'BIC = {bicc}')    
        features = np.asarray(())
        count = 0
    count = count + 1
