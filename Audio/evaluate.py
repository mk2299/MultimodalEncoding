import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import imageio

import matplotlib.pyplot as plt
import nibabel as nib
import h5py
import numpy as np
import itertools
import matplotlib.image as mpimg
import argparse
from keras.models import load_model
from keras.models import Model
from scipy.stats import pearsonr, linregress
from keras import backend as K



def data_generation_stimulus_vggish():
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        
        
        ypred = []  
        ytrue = []
        
        movie_idx = '4'
        clips = np.load('../preprocess/clip_times_24.npy')
        movie_audio = np.load('../preprocess/audio' + movie_idx + '_audioset.npy')
        idxs = clips.item().get(movie_idx)
        
        frame_idx = []
        for c in range(len(idxs)-1): ## Get rid of the last segment (it is repeated in the first 3 movies)
                frame_idx.append(np.arange(idxs[c,0]/24, idxs[c,1]/24).astype('int'))
        frame_idx = np.array(list(itertools.chain(*frame_idx)))
              
        x = [movie_audio[int(frame)][:,:,np.newaxis] for frame in frame_idx]
                    
        return np.asarray(x)
    
    
def data_generation_stimulus_context_vggish(timesteps = 20):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
  
        
        ypred = []  
        ytrue = []
        
        movie_idx = '4'
        clips = np.load('../preprocess/clip_times_24.npy')
        idxs = clips.item().get(movie_idx)
        movie_audio = np.load('../preprocess/audio' + movie_idx + '_audioset.npy')
        
        frame_idx = []
        for c in range(len(idxs)-1): ## Get rid of the last segment (it is repeated in the first 3 movies)
                frame_idx.append(np.arange(idxs[c,0]/24, idxs[c,1]/24).astype('int'))
        frame_idx = np.array(list(itertools.chain(*frame_idx)))
        x = [] 
        for frame in frame_idx: 
            arr = [movie_audio[(int(frame)+ t-(timesteps-1))][:,:,np.newaxis] for t in range(timesteps)]
            x.append(arr)  
                    
        return np.asarray(x)    
    
    
    
def main():
    parser = argparse.ArgumentParser(description='Single frame model')
    parser.add_argument('--model_file', default=None, type = str, help='Model')
    parser.add_argument('--predictions_file', default=None, type = str, help='File for saving predictions')
    parser.add_argument('--context', default = 1, type =int, help = 'Is it a contextual model')
    args = parser.parse_args()
    
    def evaluate_corr(Ypred, Yt):
        _Ypred = Ypred[:,:,:,:,0]
        _Ytrue = np.moveaxis(Yt, 3, 0)

        pred_mean = np.mean(_Ypred,0, keepdims=True)
        true_mean = np.mean(_Ytrue,0, keepdims=True)

        pred_std = np.std(_Ypred,0, keepdims=False)
        true_std = np.std(_Ytrue,0, keepdims=False)

        num = np.mean((_Ypred-pred_mean)*(_Ytrue-true_mean),0)
        den = pred_std*true_std

        p = num/den
        return p

    model = load_model(args.model_file) 
    if args.context:
        X = data_generation_stimulus_context_vggish() 
    else:
        X = data_generation_stimulus_vggish() 
    print('Predicting..')    
    Y = model.predict(X, batch_size = 1)
    print('Saving..')
    np.save(args.predictions_file, Y)
    
if __name__ == '__main__':
    main()
    