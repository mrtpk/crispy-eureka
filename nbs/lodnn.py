import keras
import numpy as np

def get_model(shape=(400,200, 6)):
    '''
    Returns LoDNN model.
    Note: Instead of max-unpooling layer, deconvolution is used[see d_layer_01].
    '''
    # encoder
    e_layer_01 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu', input_shape=shape)
    e_layer_02 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu')

    e_layer_03 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')

    # context module
    c_layer_01 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation='elu')
    c_layer_01_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_02 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 2), activation='elu')
    c_layer_02_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_03 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 4), activation='elu')
    c_layer_03_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_04 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(4, 8), activation='elu')
    c_layer_04_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_05 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(8, 16), activation='elu')
    c_layer_05_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_06 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(16, 32), activation='elu')
    c_layer_06_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_07 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(32, 64), activation='elu')
    c_layer_07_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_08 = keras.layers.Conv2D(filters=32, kernel_size=(1,1),  strides=1, padding='same')

    # decoder
    # d_layer_01: Deconvolution layer is instead of max unpooling layer 
    d_layer_01 = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')
    d_layer_02 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu')
    d_layer_03 = keras.layers.Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu')
    d_layer_04 = keras.layers.Activation('softmax')
    model = keras.models.Sequential([
                    e_layer_01,
                    e_layer_02,
                    e_layer_03,
                    c_layer_01,
                    c_layer_01_dropout,
                    c_layer_02,
                    c_layer_02_dropout,
                    c_layer_03,
                    c_layer_03_dropout,
                    c_layer_04,
                    c_layer_04_dropout,
                    c_layer_05,
                    c_layer_05_dropout,
                    c_layer_06,
                    c_layer_06_dropout,
                    c_layer_07,
                    c_layer_07_dropout,
                    c_layer_08,
                    d_layer_01,
                    d_layer_02,
                    d_layer_03,
                    d_layer_04
                   ])
    return model
