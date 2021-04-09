import numpy as np
import keras
import imageio
from skimage.transform import resize
import nibabel as nib
import os

class DataGenerator_vol(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size = 16, fraction = 0, dim=(720,1024),  volume = (111, 127, 111), n_channels = 3, delay = None, shuffle=True, stimulus_dir = '/share/sablab/nfs02/data/HCP_movie/Post_20140821_version/', response_dir = '/share/sablab/nfs03/data/HCP_7TMovies/preprocessed/MinMax/'):
        'Initialization'
        
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']

        self.fraction = fraction
        self.root_data= stimulus_dir
        self.vol_root = response_dir
        
        self.delay = delay
     
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path,  'ffmpeg'))
            
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
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.volume, 1))

        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            subject, movie, frame = self.list_IDs[idx]
            
            X[i] = np.array(self.videos[int(movie)-1].get_data(int(frame)-12*self.fraction))
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root, subject, 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay] 

        return X, y    
    
    
class DataGenerator_vol_context(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,  train= True, split = [0.9,0.1], batch_size=16, fraction = 0, dim=(720,1024),  volume = (111, 127, 111), n_channels = 3, time_steps = 20, delay = None, shuffle=True, stimulus_dir = '/share/sablab/nfs02/data/HCP_movie/Post_20140821_version/', response_dir = '/share/sablab/nfs03/data/HCP_7TMovies/preprocessed/MinMax/'):
        'Initialization'
        
        self.movie_files = ['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']

        self.fraction = fraction
        self.root_data= stimulus_dir
        self.vol_root = response_dir 
        
        self.delay = delay
        self.time_steps = time_steps
        self.videos = []
        for movie in self.movie_files:
            path = os.path.join(self.root_data, movie)
            self.videos.append(imageio.get_reader(path,  'ffmpeg'))
            
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
        X = np.zeros((self.batch_size, self.time_steps, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.volume, 1))

        # Generate data
        for i,idx in enumerate(list_indexes):
            # Store sample
            subject, movie, frame = self.list_IDs[idx]
            for t in range(self.time_steps):
                X[i, t] = np.array(self.videos[int(movie)-1].get_data(int(frame) -12*self.fraction + 24*(t - (self.time_steps-1))))
            y[i,:,:,:,0] = np.load(os.path.join(self.vol_root, subject, 'MOVIE'+ movie+'_MNI.npy'), mmap_mode='r')[2:,4:-5,:-2,int(int(frame)/24)+self.delay] 

        return X, y  
    
    
