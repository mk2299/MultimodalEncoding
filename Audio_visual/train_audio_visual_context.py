import os

import argparse

def main():
    parser = argparse.ArgumentParser(description='Single frame model')
    
    parser.add_argument('--lrate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default = 1, type=int) 
    parser.add_argument('--model_file', default = None, help = 'Location for saving model')
    parser.add_argument('--lastckpt_file', default = None, help = 'Location for saving last model')
    parser.add_argument('--log_file', default = None, help = 'Location for saving logs')
    parser.add_argument('--gpu_devices', default = "0", type = str, help = 'Device IDs')
    parser.add_argument('--gpu_count', default = None, type =int, help = 'Device count')
    parser.add_argument('--pretrained', default = 1, type = int, help = 'Freeze ResNet weights')
    parser.add_argument('--delay', default = None, type = int, help = 'HR')
    
    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

    import numpy as np

    from models_audio_visual import pretrained_volumetric_audio_visual_context_FPN
    from dataloader_audio_visual import DataGenerator_volumetric_audio_visual_context
    from keras.callbacks import ModelCheckpoint, EarlyStopping,  CSVLogger
    from keras import optimizers
    from keras.utils import multi_gpu_model
    from keras.models import load_model
    from keras import optimizers
    from keras.models import Model
    from losses import LossHistory
    
    
    IDs_train = np.genfromtxt('../preprocess/ListIDs_train.txt', dtype = 'str') 
    IDs_val = np.genfromtxt('../preprocess/ListIDs_val.txt', dtype = 'str') 
    
    train_generator = DataGenerator_volumetric_audio_visual_context(IDs_train,  batch_size = args.batch_size, train = True, delay = args.delay)
    val_generator = DataGenerator_volumetric_audio_visual_context(IDs_val,  batch_size = args.batch_size, train = False, delay = args.delay)

    history = LossHistory()
    filepath = args.model_file + "-{epoch:02d}.h5"
    callback_save = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)
    
    saver = CSVLogger(args.log_file)
    print('Pretraining parameter: ', bool(args.pretrained))
    model = pretrained_volumetric_audio_visual_context_FPN()#sf_audio = args.sf_audio, sf_visual = args.sf_visual)
    if args.gpu_count>1:
        model = multi_gpu_model(model, gpus = args.gpu_count)
    print(model.summary())
    model.compile(optimizer=optimizers.Adam(lr=args.lrate, amsgrad=True), loss='mean_squared_error',metrics=['mean_squared_error'])
    model.fit_generator(
        train_generator,
        validation_data=val_generator,
        callbacks = [history, saver, callback_save], steps_per_epoch = 2000, validation_steps = 100,
        epochs = args.epochs)
    model.save(args.lastckpt_file)
if __name__ == '__main__':
    main()
