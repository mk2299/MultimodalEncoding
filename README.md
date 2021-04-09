# Cortical response to naturalistic stimuli is largely predictable with deep neural networks
This code provides a python Keras based implementation of the following paper (to appear in Science Advances):\
Meenakshi Khosla, Gia H. Ngo, Keith W. Jamison, Amy Kuceyeski, Mert R. Sabuncu\
"Cortical response to naturalistic stimuli is largely predictable with deep neural networks"\
Biorxiv: https://www.biorxiv.org/content/10.1101/2020.09.11.293878v1 

keywords: Naturalistic neuroscience, fMRI, Convolutional Neural Networks, Auditory perception, Visual perception

This implementation is catered to the Human Connectome Project dataset where we focus on predicting the fMRI response to naturalistic stimuli like movies. The code can be used to train 6 types of models that differ in the types and amounts of information they receive.\ 
(1) Audio-1sec and (2) Audio-20sec models, which are trained on single audio spectrograms extracted over 1-second epochs and contiguous sequences of 20 spectrograms spanning 20 seconds respectively \
(3) Visual-1sec and (4) Visual-20sec models, trained with last frames of 1-second epochs and sequences of 20 evenly spaced frames within 20-second clips respectively \
(5) Audiovisual-1sec and (6) Audiovisual-20sec models, which employ audio and visual input as described above, jointly. 

__Data__  \
All experiments in this study are based on the Human Connectome Project movie-watching database. The dataset is publically available for download through the ConnectomeDB software [https://db.humanconnectome.org/]. Here, we utilized 7T fMRI data from the 'Movie Task fMRI 1.6mm/59k FIX-Denoised' package. 
Correct path to stimulus files should be provided in the dataloaders. Note that for visual and audiovisual models, the stimulus is extracted automatically from the .mp4 files (['7T_MOVIE1_CC1_v2.mp4', '7T_MOVIE2_HO1_v2.mp4', '7T_MOVIE3_CC2_v2.mp4', '7T_MOVIE4_HO2_v2.mp4']) in the dataloader. \
For the preprocessing of video clips to extract auditory stimuli, refer to instructions provided here: https://github.com/mk2299/SharedEncoding_MICCAI. The audio spectrogram extraction is based on parameters set in the Audioset library (https://github.com/tensorflow/models/tree/master/research/audioset). 

__Requirements__ \
Tensorflow (>1.3.0) \
Keras (=2.0.8) \
NumPy \
NiBabel \
Scikit-learn \

__Usage__ \
cd to respective directory for training audio/visual/audiovisual models. 
e.g., To learn about the parameters, use \
python train_audio_visual_context.py --help \






