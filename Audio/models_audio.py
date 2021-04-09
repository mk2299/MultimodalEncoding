from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose, Conv2DTranspose, Reshape, Permute, concatenate, UpSampling2D, Cropping2D, Concatenate, TimeDistributed, LSTM
from keras.models import Sequential
from keras import layers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.applications.resnet50 import ResNet50
import sys
sys.path.append('/home/mk2299/HCP_Movies/Audio_experiments/audioset/vggish')
from vggish_keras import get_vggish_keras



def pretrained_vggish_volumetric_FPN(pretrained = True):
 
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    
    def _upsample_add( x, y, crop = 0):
        #print(x.shape, y.shape)
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = load_model('./vggish/vggish_weights_keras.h5')
                 
    
    layer1 =  base_model.layers[3].output 
    layer2 =  base_model.layers[6].output 
    layer3 =  base_model.layers[9].output 
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    
    # Lateral layers
    toplayer  = Conv2D(128, kernel_size=1, strides=1) #(layer4)
    latlayer1 = Conv2D(128, kernel_size=1, strides=1) #(layer3)
    latlayer2 = Conv2D(128, kernel_size=1, strides=1) #(layer2)
    
    
    p5 = toplayer(layer3)
    p4 = _upsample_add(p5, latlayer1(layer2), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer1))
    p3 = smooth2(p3)
    
    
    z = concatenate([layer4, GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(p5)])
    x = Dense(6*7*6*1024, activation='elu')(z)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
   
    model = Model(inputs = base_model.input, outputs = y)
    return model

def pretrained_vggish_volumetric_FPN_frozen():
 
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    
    def _upsample_add( x, y, crop = 0):
        #print(x.shape, y.shape)
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = load_model('./vggish/vggish_weights_keras.h5')
    for layer in base_model.layers:
         layer.trainable = False
    
    layer1 =  base_model.layers[3].output 
    layer2 =  base_model.layers[6].output 
    layer3 =  base_model.layers[9].output 
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    
    # Lateral layers
    toplayer  = Conv2D(128, kernel_size=1, strides=1) #(layer4)
    latlayer1 = Conv2D(128, kernel_size=1, strides=1) #(layer3)
    latlayer2 = Conv2D(128, kernel_size=1, strides=1) #(layer2)
    
    
    p5 = toplayer(layer3)
    p4 = _upsample_add(p5, latlayer1(layer2), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer1))
    p3 = smooth2(p3)
    
    
    z = concatenate([layer4, GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(p5)])
    x = Dense(6*7*6*1024, activation='elu')(z)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
   
    model = Model(inputs = base_model.input, outputs = y)
    return model



def nonpretrained_vggish_volumetric_FPN(pretrained = False):
 
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    
    def _upsample_add( x, y, crop = 0):
        #print(x.shape, y.shape)
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = get_vggish_keras()     
    layer1 =  base_model.layers[3].output 
    layer2 =  base_model.layers[6].output 
    layer3 =  base_model.layers[9].output 
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    
    # Lateral layers
    toplayer  = Conv2D(128, kernel_size=1, strides=1) 
    latlayer1 = Conv2D(128, kernel_size=1, strides=1) 
    latlayer2 = Conv2D(128, kernel_size=1, strides=1) 
    
    
    p5 = toplayer(layer3)
    p4 = _upsample_add(p5, latlayer1(layer2), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer1))
    p3 = smooth2(p3)
    
    
    z = concatenate([layer4, GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(p5)])
    x = Dense(6*7*6*1024, activation='elu')(z)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
   
    model = Model(inputs = base_model.input, outputs = y)
    return model

def pretrained_vggish_volumetric(pretrained = True):
 
    
  
    base_model = load_model('./vggish/vggish_weights_keras.h5')
    if pretrained:           
        for layer in base_model.layers:
                   layer.trainable = False
   
    layer4 =  base_model.get_output_at(-1)
    
    x = Dense(6*7*6*1024, activation='elu')(layer4)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
   
    model = Model(inputs = base_model.input, outputs = y)
    return model

def nonpretrained_vggish_volumetric(pretrained = True):
 
    
  
    base_model = get_vggish_keras()                  
   
    layer4 =  base_model.get_output_at(-1)
    
    x = Dense(6*7*6*1024, activation='elu')(layer4)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
   
    model = Model(inputs = base_model.input, outputs = y)
    return model


#################################### Context model #########################################################


def pretrained_vggish_volumetric_context_FPN(img_shape = (20, 96, 64,1), pretrained = True):
 
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    
    def _upsample_add( x, y, crop = 0):
        #print(x.shape, y.shape)
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = load_model('./vggish/vggish_weights_keras.h5')
                 
    
    layer1 =  base_model.layers[3].output #base_model.get_layer('activation_10').output #(input2) #
    layer2 =  base_model.layers[6].output #base_model.get_layer('activation_22').output #(input2) #
    layer3 =  base_model.layers[9].output #base_model.get_layer('activation_40').output #(input2) #
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    
    # Lateral layers
    toplayer  = Conv2D(128, kernel_size=1, strides=1) #(layer4)
    latlayer1 = Conv2D(128, kernel_size=1, strides=1) #(layer3)
    latlayer2 = Conv2D(128, kernel_size=1, strides=1) #(layer2)
    
    
    p5 = toplayer(layer3)
    p4 = _upsample_add(p5, latlayer1(layer2), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer1))
    p3 = smooth2(p3)
    
    
    z = concatenate([layer4, GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(p5)])
    singleframe_model = Model(inputs = base_model.input, outputs = z)
    
    sequence = Input(shape = img_shape, dtype='float32') 
    x = TimeDistributed(singleframe_model)(sequence) 
    x = LSTM(512)(x)
    
        
    x = Dense(6*7*6*1024, activation='elu')(x)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    out = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = sequence , outputs = out)
    
    return model


def pretrained_vggish_volumetric_context_FPN_frozen(img_shape = (20, 96, 64,1), pretrained = True):
 
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    
    def _upsample_add( x, y, crop = 0):
        #print(x.shape, y.shape)
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = load_model('./vggish/vggish_weights_keras.h5')
                 
    for layer in base_model.layers:
        layer.trainable = False
        
    layer1 =  base_model.layers[3].output #base_model.get_layer('activation_10').output #(input2) #
    layer2 =  base_model.layers[6].output #base_model.get_layer('activation_22').output #(input2) #
    layer3 =  base_model.layers[9].output #base_model.get_layer('activation_40').output #(input2) #
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    
    # Lateral layers
    toplayer  = Conv2D(128, kernel_size=1, strides=1) #(layer4)
    latlayer1 = Conv2D(128, kernel_size=1, strides=1) #(layer3)
    latlayer2 = Conv2D(128, kernel_size=1, strides=1) #(layer2)
    
    
    p5 = toplayer(layer3)
    p4 = _upsample_add(p5, latlayer1(layer2), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer1))
    p3 = smooth2(p3)
    
    
    z = concatenate([layer4, GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(p5)])
    singleframe_model = Model(inputs = base_model.input, outputs = z)
    
    sequence = Input(shape = img_shape, dtype='float32') 
    x = TimeDistributed(singleframe_model)(sequence) 
    x = LSTM(512)(x)
    
        
    x = Dense(6*7*6*1024, activation='elu')(x)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    out = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = sequence , outputs = out)
    
    return model




def nonpretrained_vggish_volumetric_context_FPN(img_shape = (20, 96, 64,1), pretrained = True):
 
    
    def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    
    def _upsample_add( x, y, crop = 0):
        #print(x.shape, y.shape)
        out = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
        if crop==1:
            ch, cw = get_crop_shape(out, y)
            out = Cropping2D(cropping=(ch,cw))(out)
       
        return Add()([out, y]) 
    
    base_model = get_vggish_keras()
                 
    
    layer1 =  base_model.layers[3].output 
    layer2 =  base_model.layers[6].output 
    layer3 =  base_model.layers[9].output 
    layer4 =  base_model.get_output_at(-1)
    
    smooth1 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    smooth2 = Conv2D(128, kernel_size=3, strides=1, padding="same")
    
    # Lateral layers
    toplayer  = Conv2D(128, kernel_size=1, strides=1) #(layer4)
    latlayer1 = Conv2D(128, kernel_size=1, strides=1) #(layer3)
    latlayer2 = Conv2D(128, kernel_size=1, strides=1) #(layer2)
    
    
    p5 = toplayer(layer3)
    p4 = _upsample_add(p5, latlayer1(layer2), crop = 1)
    p4 = smooth1(p4)  
    p3 = _upsample_add(p4, latlayer2(layer1))
    p3 = smooth2(p3)
    
    
    z = concatenate([layer4, GlobalAveragePooling2D()(p3), GlobalAveragePooling2D()(p4), GlobalAveragePooling2D()(p5)])
    singleframe_model = Model(inputs = base_model.input, outputs = z)
    
    sequence = Input(shape = img_shape, dtype='float32') 
    x = TimeDistributed(singleframe_model)(sequence) 
    x = LSTM(512)(x)
    
        
    x = Dense(6*7*6*1024, activation='elu')(x)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    out = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = sequence , outputs = out)
    
    return model



def pretrained_vggish_volumetric_context(img_shape = (20, 96, 64,1), pretrained = True):
 
    base_model = load_model('./vggish/vggish_weights_keras.h5')
    
    if pretrained:           
        for layer in base_model.layers:
                   layer.trainable = False
    
    layer4 =  base_model.get_output_at(-1)
    
    singleframe_model = Model(inputs = base_model.input, outputs = layer4)
    
    sequence = Input(shape = img_shape, dtype='float32') 
    x = TimeDistributed(singleframe_model)(sequence) 
    x = LSTM(512)(x)
    
        
    x = Dense(6*7*6*1024, activation='elu')(x)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    out = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = sequence , outputs = out)
    
    return model


def nonpretrained_vggish_volumetric_context(img_shape = (20, 96, 64,1), pretrained = True):
 
    base_model = get_vggish_keras()

    layer4 =  base_model.get_output_at(-1)
    
    singleframe_model = Model(inputs = base_model.input, outputs = layer4)
    
    sequence = Input(shape = img_shape, dtype='float32') 
    x = TimeDistributed(singleframe_model)(sequence) 
    x = LSTM(512)(x)
    
        
    x = Dense(6*7*6*1024, activation='elu')(x)
    y = Reshape((6,7,6,1024))(x)
    
    y = Conv3DTranspose(512,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(256,(3,3,3), (2,2,2), activation='elu')(y)
    y = Conv3DTranspose(128,(3,3,3), (2,2,2), activation='elu')(y)
    out = Conv3DTranspose(1,(3,3,3), (2,2,2), activation='elu')(y)
    model = Model(inputs = sequence , outputs = out)
    
    return model


