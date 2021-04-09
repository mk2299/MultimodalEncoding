import numpy as np
import keras
import imageio
from skimage.transform import resize
import nibabel as nib
import os
import cv2

class DataGenerator_vol(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size=32,  dim=(96,64),  volume = (111, 127, 111), n_channels = 1, delay = None, shuffle=True,stimulus_dir = '/home/mk2299/HCP_Movies/preprocess/', response_dir = '/nfs03/data/HCP_7TMovies/preprocessed/MinMax/' ):
        'Initialization'
        
    
        self.root = stimulus_dir
        self.vol_root = response_dir
        self.delay = delay
        
        
        self.audio_files = ['audio1_audioset.npy', 'audio2_audioset.npy', 'audio3_audioset.npy']
        self.audios = []
        for audio in self.audio_files:
            path = os.path.join(self.root, audio)
            self.audios.append(np.load(path))
            
        self.dim = dim
        self.volume = volume 
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs

        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

   
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.volume, 1))

        # Generate data
        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            subject, movie, frame = self.list_IDs[idx]
           
            
            X[i,:,:,0] = self.audios[int(movie)-1][int(int(frame)/24)]  
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root, subject, 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay]
            

        return X, y 
    


    
############################################ Context #################################################


class DataGenerator_vol_context(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size=32,  dim=(96, 64), volume = (111, 127, 111),  n_channels = 1, delay = None, shuffle=True, time_steps = 20, stimulus_dir = '/home/mk2299/HCP_Movies/preprocess/', response_dir = '/nfs03/data/HCP_7TMovies/preprocessed/MinMax/' ):
        'Initialization'
        
         
        self.root = stimulus_dir
        self.vol_root = response_dir
        self.delay = delay
        
        self.audio_files = ['audio1_audioset.npy', 'audio2_audioset.npy', 'audio3_audioset.npy']
        self.audios = []
        for audio in self.audio_files:
            path = os.path.join(self.root, audio)
            self.audios.append(np.load(path))
        
        
        self.time_steps = time_steps   
        self.dim = dim
        self.volume = volume 
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs

        
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

       
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.time_steps, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.volume, 1))

        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            subject, movie, frame = self.list_IDs[idx]
            
            for t in range(self.time_steps):
                X[i, t] = self.audios[int(movie)-1][int(int(frame)/24)+ t - (self.time_steps-1)] [:,:,np.newaxis]
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root, subject, 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay] 
        return X, y 
    
