import numpy as np
import keras
import imageio
from skimage.transform import resize
import nibabel as nib
import os
import cv2

class DataGenerator_volumetric_audio_visual_context(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size = 32,  dim1=(96, 64), dim2=(720,1024), volume = (111, 127, 111), n_channels1 = 1, n_channels2 = 3, delay = None, shuffle=True, time_steps = 20):
        'Initialization'
        
        self.vol_root = '/share/sablab/nfs03/data/HCP_7TMovies/preprocessed/MinMax/'
        self.root_data='/share/sablab/nfs02/data/HCP_movie/Post_20140821_version/'    
        self.root = '/home/mk2299/HCP_Movies/preprocess/'
        self.delay = delay
       
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']
        
        
        
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path,  'ffmpeg'))
        
        self.audio_files = ['audio1_audioset.npy', 'audio2_audioset.npy', 'audio3_audioset.npy']
        self.audios = []
        for audio in self.audio_files:
            path = os.path.join(self.root, audio)
            self.audios.append(np.load(path))
        
        self.time_steps = time_steps 
        self.dim1 = dim1
        self.dim2 = dim2
        self.volume = volume
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs
#         train_idx = int(split[0]*self.total_size)
#         val_idx = int((split[0]+split[1])*self.total_size)
#         if train == True:
#             self.list_IDs = list_IDs[:train_idx]
#         else:
#             self.list_IDs = list_IDs[train_idx:val_idx]
         
        
        self.n_channels1 = n_channels1
        self.n_channels2 = n_channels2
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
        X1 = np.empty((self.batch_size, self.time_steps, *self.dim1, self.n_channels1))
        X2 = np.empty((self.batch_size, self.time_steps, *self.dim2, self.n_channels2))
        y = np.empty((self.batch_size, *self.volume, 1))

        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            subject, movie, frame = self.list_IDs[idx]
                
            for t in range(self.time_steps):
                X1[i, t] = self.audios[int(movie)-1][int(int(frame)/24)+ t - (self.time_steps-1)] [:,:,np.newaxis]
                X2[i, t] = np.array(self.videos[int(movie)-1].get_data(int(frame) + 24*(t - (self.time_steps-1))))
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root, subject, 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay]

        return [X1, X2], y  

class DataGenerator_volumetric_audio_visual_singleframe(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size = 32,  dim1=(96, 64), dim2=(720,1024), volume = (111, 127, 111), n_channels1 = 1, n_channels2 = 3, delay = None, shuffle=True):
        'Initialization'
        
        self.vol_root = '/share/sablab/nfs03/data/HCP_7TMovies/preprocessed/MinMax/'
        self.root_data='/share/sablab/nfs02/data/HCP_movie/Post_20140821_version/'    
        self.root = '/home/mk2299/HCP_Movies/preprocess/'
        self.delay = delay
       
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']
        
        
        
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path,  'ffmpeg'))
        
        self.audio_files = ['audio1_audioset.npy', 'audio2_audioset.npy', 'audio3_audioset.npy']
        self.audios = []
        for audio in self.audio_files:
            path = os.path.join(self.root, audio)
            self.audios.append(np.load(path))
        
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.volume = volume
        self.batch_size = batch_size
        self.total_size = len(list_IDs)
        self.list_IDs = list_IDs
        
#         train_idx = int(split[0]*self.total_size)
#         val_idx = int((split[0]+split[1])*self.total_size)
#         if train == True:
#             self.list_IDs = list_IDs[:train_idx]
#         else:
#             self.list_IDs = list_IDs[train_idx:val_idx]
         
        
        self.n_channels1 = n_channels1
        self.n_channels2 = n_channels2
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
        X1 = np.empty((self.batch_size,  *self.dim1, self.n_channels1))
        X2 = np.empty((self.batch_size,  *self.dim2, self.n_channels2))
        y = np.zeros((self.batch_size, *self.volume, 1))

        # Generate data
        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            subject, movie, frame = self.list_IDs[idx]
               
            
            X1[i] = self.audios[int(movie)-1][int(int(frame)/24)] [:,:,np.newaxis]
            X2[i] = self.videos[int(movie)-1].get_data(int(frame))
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root, subject, 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay] 


        return [X1, X2], y               