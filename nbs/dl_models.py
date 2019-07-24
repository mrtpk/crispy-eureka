import keras
from keras_drop_block import DropBlock2D
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Input, Dropout, concatenate, UpSampling2D
from keras import backend as K

def get_lodnn_model(shape=(400,200, 6)):
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

    c_layer_02 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 1), activation='elu')
    c_layer_02_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_03 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(4, 2), activation='elu')
    c_layer_03_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_04 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(8, 4), activation='elu')
    c_layer_04_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_05 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(16, 8), activation='elu')
    c_layer_05_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_06 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(32, 16), activation='elu')
    c_layer_06_dropout = keras.layers.SpatialDropout2D(rate=0.25)

    c_layer_07 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(64, 32), activation='elu')
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

def get_unet_model(input_size = (400,200,6)):
    inputs = keras.layers.Input(input_size)
    conv1 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv4 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    drop5 = keras.layers.Dropout(0.5)(pool4)

    up6 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    
    up7 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = keras.layers.concatenate([conv2,up7], axis = 3)
    conv7 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    up8 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = keras.layers.concatenate([conv1,up8], axis = 3)
    conv8 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv9 = keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = keras.models.Model(input = inputs, output = conv10)
    
    return model


def conv2dLeakyDownProj(layer_input, filters, output,f_size=3):
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(output, kernel_size=1, padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d


def conv2dLeakyProj(layer_input, filters,output, f_size=3):
    d = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(output, kernel_size=1, padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d


def conv2dLeaky(layer_input, filters, f_size=3):
    d = Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    return d


def u_net6(shape, filters=4, f_size=3, int_space=4, output_channels=4, rate=.3):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    inputs = Input(shape)
    #  inputN = GaussianNoise(rate)(inputs)

    dblock0 = DropBlock2D(block_size=5, keep_prob=0.8, name="dropblock0")(inputs)

    pool1 = conv2dLeakyDownProj(dblock0, filters, int_space, f_size=f_size)
    pool1 = Dropout(rate)(pool1)
    pool2 = conv2dLeakyDownProj(pool1, filters, int_space, f_size=f_size)
    pool2 = Dropout(rate)(pool2)
    pool3 = conv2dLeakyDownProj(pool2, filters, int_space, f_size=f_size)
    pool3 = Dropout(rate)(pool3)
    pool4 = conv2dLeakyDownProj(pool3, filters, int_space, f_size=f_size)
    pool4 = Dropout(rate)(pool4)
    pool5 = conv2dLeakyDownProj(pool4, filters, int_space, f_size=f_size)
    pool5 = Dropout(rate)(pool5)
    up4 = concatenate([UpSampling2D(size=(2, 2))(pool5), pool4], axis=channel_axis)
    conv4 = conv2dLeakyProj(up4, filters, int_space, f_size=f_size)
    conv4 = Dropout(rate)(conv4)
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv4), pool3], axis=channel_axis)
    conv3 = conv2dLeakyProj(up3, filters, int_space, f_size=f_size)
    conv3 = Dropout(rate)(conv3)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv3), pool2], axis=channel_axis)
    conv2 = conv2dLeakyProj(up2, filters, int_space, f_size=f_size)
    conv2 = Dropout(rate)(conv2)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv2), pool1], axis=channel_axis)
    conv1 = conv2dLeakyProj(up1, filters, int_space, f_size=f_size)
    conv1 = Dropout(rate)(conv1)
    up0 = concatenate([UpSampling2D(size=(2, 2))(conv1), inputs], axis=channel_axis)
    conv0 = conv2dLeaky(up0, output_channels, f_size=f_size)
    #  conv0  = GaussianNoise(rate)(conv0)
    return Model(inputs, conv0)
