import os
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
from models_audio_visual import pretrained_volumetric_audio_visual_context_FPN_version3_small

def data_generation_stimulus_vggish():
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
    
        ypred = []  
        ytrue = []
        
        movie_idx = '4'
        clips = np.load('../preprocess/clip_times_24.npy', allow_pickle = True)
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
        clips = np.load('../preprocess/clip_times_24.npy', allow_pickle = True)
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
    parser = argparse.ArgumentParser(description='Evaluation of single frame models')
    parser.add_argument('--model_file', default=None, type = str, help='Model')
    parser.add_argument('--predictions_file', default=None, type = str, help='File for saving predictions')
    parser.add_argument('--context', default = False, type = bool, help = 'Is it a contextual model')
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
    model = pretrained_volumetric_audio_visual_context_FPN_version3_small()
    model.load_weights(args.model_file)
    #model = load_model(args.model_file) 
    if args.context:
        X_audio =  data_generation_stimulus_context_vggish(timesteps = 20) 
        X_visual = np.load('../preprocess/stimulus_context.npy')[:699] #Removed last clip because it is repeated
    else:
        X_audio =  data_generation_stimulus_vggish() 
        X_visual = np.load('../preprocess/stimulus_singleframe.npy')
    print('Predicting..')    
    #idx = np.arange(699)
    #np.random.shuffle(idx) # Uncomment this for randomization experiment
    
    Y = model.predict([X_audio, X_visual], batch_size = 1) 
    print('Saving..')
    np.save(args.predictions_file, Y)
    
if __name__ == '__main__':
    main()