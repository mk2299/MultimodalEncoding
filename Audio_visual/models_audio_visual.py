from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose, Conv2DTranspose, Reshape, Permute, concatenate, UpSampling2D, Cropping2D, Concatenate, TimeDistributed, LSTM
from keras.models import Sequential
from keras import layers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.applications.resnet50 import ResNet50
from base_models import audio_FPN, visual_FPN

def pretrained_volumetric_audio_visual_singleframe_FPN(pretrained = True):
        
        audio_model = audio_FPN()
        visual_model = visual_FPN(pretrained = pretrained)
        
        
        for layer in visual_model.layers:
            layer.name = layer.name + str("_2")

        x1 = audio_model.output
        x2 = visual_model.output
 
        x = concatenate([x1,x2])
        x = Dense(1024, activation = 'elu')(x)
        x = Dense(6*7*6*1024, activation='elu')(x)
        y = Reshape((6,7,6,1024))(x)

        y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
        y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
        y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
        out = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
        
        model = Model(inputs = [audio_model.input, visual_model.input], outputs = out)
        
        return model   


    
def pretrained_volumetric_audio_visual_context_FPN(img_shape1 = (20,96,64,1), img_shape2 = (20,720,1024,3), pretrained = True):
        
        audio_model = audio_FPN()
        visual_model = visual_FPN(pretrained = pretrained)
        
        
        for layer in visual_model.layers:
                layer.name = layer.name + str("_2")

        
        
        sequence1 = Input(shape = img_shape1, dtype='float32')
        sequence2 = Input(shape = img_shape2, dtype='float32')
        x1 = TimeDistributed(audio_model)(sequence1) 
        x1 = LSTM(512)(x1)
        
        x2 = TimeDistributed(visual_model)(sequence2)    
        x2 = LSTM(1024)(x2)
    
        x = concatenate([x1,x2])
        x = Dense(1024, activation = 'elu')(x)
        x = Dense(6*7*6*1024, activation='elu')(x)
        y = Reshape((6,7,6,1024))(x)

        y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
        y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
        y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
        out = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
        
        model = Model(inputs = [sequence1, sequence2], outputs = out)
        
        return model       




