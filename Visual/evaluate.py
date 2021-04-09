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

def main():
    parser = argparse.ArgumentParser(description='Single frame model')
    parser.add_argument('--model_file', default=None, type = str, help='Model')
    parser.add_argument('--predictions_file', default=None, type = str, help='File for saving predictions')
    parser.add_argument('--context', default = 0, type = int, help = 'Is it a contextual model')
    parser.add_argument('--stimulus_dir', default = '/home/mk2299/HCP_Movies/Results', type = str, help = 'Path to saved images from test clip')
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
        X = np.load(os.path.join(args.stimulus_dir, 'stimulus_context.npy'), mmap_mode = 'r')[:699] ## Remove last clip from analyses as it is repeated
    else:
        X = np.load(os.path.join(args.stimulus_dir, 'stimulus_singleframe.npy'))
    print('Predicting..')    
    Y = model.predict(X, batch_size = 1)
    print('Saving..')
    np.save(args.predictions_file, Y)
    
if __name__ == '__main__':
    main()
    