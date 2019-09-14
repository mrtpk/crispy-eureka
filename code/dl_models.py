import keras
from keras_drop_block import DropBlock2D
from keras.models import Model
from keras.layers import Conv2D, LeakyReLU, Input, Dropout, concatenate, \
    UpSampling2D, BatchNormalization, MaxPooling2D,  GlobalAveragePooling2D, Activation
from keras import backend as K
import larq as lq
import tensorflow as tf 

def get_Q_lodnn_model(shape=(400,200, 6)):
    '''
    Returns LoDNN model.
    Note: Instead of max-unpooling layer, deconvolution is used[see d_layer_01].
    '''
    model = tf.keras.Sequential([
    lq.layers.QuantConv2D(32, 3, strides=(1, 1), 
                        padding='same', kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip", input_shape=shape), #activation='elu'
    
    lq.layers.QuantConv2D(32, 3, strides=(1, 1), 
                        padding='same', kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'),

    # context module
    lq.layers.QuantConv2D(128, 3, strides=(1, 1), padding='same', 
                        dilation_rate=(1, 1), kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    
    tf.keras.layers.SpatialDropout2D(rate=0.25),
    lq.layers.QuantConv2D(128, 3, strides=(1, 1), padding='same', 
                        dilation_rate=(2, 1), kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    
    tf.keras.layers.SpatialDropout2D(rate=0.25),
    lq.layers.QuantConv2D(128, 3, strides=(1, 1), padding='same', 
                        dilation_rate=(4, 2), kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    
    tf.keras.layers.SpatialDropout2D(rate=0.25),
    lq.layers.QuantConv2D(128, 3, strides=(1, 1), padding='same', 
                        dilation_rate=(8, 4), kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    
    tf.keras.layers.SpatialDropout2D(rate=0.25),
    lq.layers.QuantConv2D(128, 3, strides=(1, 1), padding='same', 
                        dilation_rate=(16, 8), kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    
    tf.keras.layers.SpatialDropout2D(rate=0.25),
    lq.layers.QuantConv2D(128, 3, strides=(1, 1), padding='same', 
                        dilation_rate=(32, 16), kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    
    tf.keras.layers.SpatialDropout2D(rate=0.25),
    lq.layers.QuantConv2D(128, 3, strides=(1, 1), padding='same', 
                        dilation_rate=(64, 32), kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    
    tf.keras.layers.SpatialDropout2D(rate=0.25),
    lq.layers.QuantConv2D(32, 1, strides=1, padding='same', 
                        kernel_quantizer="ste_sign", 
                        kernel_constraint="weight_clip"), #activation='elu'
    

    # decoder
    lq.layers.QuantConv2DTranspose(32,3,strides=(2,2),padding='same'),
    lq.layers.QuantConv2D(32, 1, strides=(1,1), padding='same', kernel_quantizer="ste_sign", 
                          kernel_constraint="weight_clip"), #activation='elu'
    lq.layers.QuantConv2D(2, 3, strides=(1,1), padding='same', kernel_quantizer="ste_sign", 
                          kernel_constraint="weight_clip"), #activation='elu'
    tf.keras.layers.Activation("softmax")
    ])
    return model

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

def get_unet_model(input_size = (400,200,6), subsample_ratio=1):
    inputs = keras.layers.Input(input_size)
    conv1 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1 = keras.layers.MaxPooling2D(pool_size=(1, 2))(conv1)
    conv2 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    pool3 = keras.layers.MaxPooling2D(pool_size=(1, 2))(conv2)
    conv4 = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    drop4 = keras.layers.Dropout(0.5)(conv4)
    pool4 = keras.layers.MaxPooling2D(pool_size=(1, 2))(drop4)
    drop5 = keras.layers.Dropout(0.5)(pool4)

    up6 = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (1,2))(drop5))
    merge6 = keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    
    up7 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (1,2))(conv6))
    merge7 = keras.layers.concatenate([conv2,up7], axis = 3)
    conv7 = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    up8 = keras.layers.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling2D(size = (1,2))(conv7))
    merge8 = keras.layers.concatenate([conv1,up8], axis = 3)
    conv8 = keras.layers.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    if subsample_ratio % 2 == 0:
        conv8 = keras.layers.Conv2D(8, 3, activation= 'relu', padding='same', kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2,1))(conv8))

    if subsample_ratio % 4 == 0:
        conv8 = keras.layers.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 1))(conv8))

    conv9 = keras.layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    conv10 = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = keras.models.Model(input = inputs, output = conv10)
    
    return model


def conv2dLeakyDownProj(layer_input, filters, output,f_size=3):
    d = Conv2D(filters, kernel_size=f_size, strides=(1, 2), padding='same')(layer_input)
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


def u_net6(shape, filters=4, f_size=3, int_space=4, output_channels=4, rate=.3, subsample_ratio=1):
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
    up4 = concatenate([UpSampling2D(size=(1, 2))(pool5), pool4], axis=channel_axis)
    conv4 = conv2dLeakyProj(up4, filters, int_space, f_size=f_size)
    conv4 = Dropout(rate)(conv4)
    up3 = concatenate([UpSampling2D(size=(1, 2))(conv4), pool3], axis=channel_axis)
    conv3 = conv2dLeakyProj(up3, filters, int_space, f_size=f_size)
    conv3 = Dropout(rate)(conv3)
    up2 = concatenate([UpSampling2D(size=(1, 2))(conv3), pool2], axis=channel_axis)
    conv2 = conv2dLeakyProj(up2, filters, int_space, f_size=f_size)
    conv2 = Dropout(rate)(conv2)
    up1 = concatenate([UpSampling2D(size=(1, 2))(conv2), pool1], axis=channel_axis)
    conv1 = conv2dLeakyProj(up1, filters, int_space, f_size=f_size)
    conv1 = Dropout(rate)(conv1)
    up0 = concatenate([UpSampling2D(size=(1, 2))(conv1), inputs], axis=channel_axis)
    conv0 = conv2dLeaky(up0, 2, f_size=f_size)

    if subsample_ratio % 2 == 0:
        conv0 = keras.layers.Conv2D(8, 3, activation= 'relu', padding='same', kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 1))(conv0))

    if subsample_ratio % 4 == 0:
        conv0 = keras.layers.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            keras.layers.UpSampling2D(size=(2, 1))(conv0))

    conv0 = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv0)
    #  conv0  = GaussianNoise(rate)(conv0)
    return Model(inputs, conv0)


def FireModule(s_1x1, e_1x1, e_3x3, name):
    """FireModule
        Fire module for the SqueezeNet model.
        Implements the expand layer, which has a mix of 1x1 and 3x3 filters,
        by using two conv layers concatenated in the channel dimension.
    :param s_1x1: Number of 1x1 filters in the squeeze layer
    :param e_1x1: Number of 1x1 filters in the expand layer
    :param e_3x3: Number of 3x3 filters in the expand layer
    :param name: Name of the fire module
    :return:
        Returns a callable function
    """
    # Concat on the channel axis. TensorFlow uses (rows, cols, channels), while
    # Theano uses (channels, rows, cols).
    if K.image_dim_ordering() == 'tf':
        concat_axis = 3
    else:
        concat_axis = 1

    def layer(x):
        squeeze = Conv2D(filters=s_1x1, kernel_size=1, activation='relu', kernel_initializer='glorot_uniform', name=name + '/squeeze1x1')(x)
        squeeze = BatchNormalization(name=name + '/squeeze1x1_bn')(squeeze)

        # Needed to merge layers expand_1x1 and expand_3x3.
        expand_1x1 = Conv2D(filters=e_1x1, kernel_size=1, activation='relu', kernel_initializer='glorot_uniform', name=name + '/expand1x1')(
            squeeze)

        # Pad the border with zeros. Not needed as border_mode='same' will do the same.
        # expand_3x3 = ZeroPadding2D(padding=(1, 1), name=name+'_expand_3x3_padded')(squeeze)
        expand_3x3 = Conv2D(filters=e_3x3, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform',
                                   name=name + '/expand3x3')(squeeze)
        # Concat in the channel dim
        expand_merge = concatenate([expand_1x1, expand_3x3], axis=concat_axis, name=name + '/concat')
        return expand_merge

    return layer


def SqueezeNet(nb_classes, input_shape):
    """
        SqueezeNet v1.1 implementation
    :param nb_classes: Number of classes
    :param rows: Amount of rows in the input
    :param cols: Amount of cols in the input
    :param channels: Amount of channels in the input
    :returns: SqueezeNet model
    """
    # if K.image_dim_ordering() == 'tf':
    #     input_shape = (rows, cols, channels)
    # else:
    #     input_shape = (channels, rows, cols)

    input_image = Input(shape=input_shape)
    # conv1 output shape = (113, 113, 64)

    conv1 = Conv2D(filters=64, kernel_size=3, activation='relu', strides=(2, 2), kernel_initializer='glorot_uniform', name='conv1')(
        input_image)
    # maxpool1 output shape = (56, 56, 64)
    maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)
    # fire2 output shape = (?, 56, 56, 128)
    fire2 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire2')(maxpool1)
    # fire3 output shape = (?, 56, 56, 128)
    fire3 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire3')(fire2)
    # maxpool3 output shape = (?, 27, 27, 128)
    maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(fire3)
    # fire4 output shape = (?, 56, 56, 256)
    fire4 = FireModule(s_1x1=32, e_1x1=128, e_3x3=128, name='fire4')(fire3)
    # fire5 output shape = (?, 56, 56, 256)
    fire5 = FireModule(s_1x1=32, e_1x1=128, e_3x3=128, name='fire5')(fire4)
    # maxpool5 output shape = (?, 27, 27, 256)
    maxpool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(fire5)
    # fire6 output shape = (?, 27, 27, 384)
    fire6 = FireModule(s_1x1=48, e_1x1=192, e_3x3=192, name='fire6')(maxpool5)
    # fire7 output shape = (?, 27, 27, 384)
    fire7 = FireModule(s_1x1=48, e_1x1=192, e_3x3=192, name='fire7')(fire6)
    # fire8 output shape = (?, 27, 27, 512)
    fire8 = FireModule(s_1x1=64, e_1x1=256, e_3x3=256, name='fire8')(fire7)
    # fire9 output shape = (?, 27, 27, 512)
    fire9 = FireModule(s_1x1=64, e_1x1=256, e_3x3=256, name='fire9')(fire8)
    # Dropout after fire9 module.
    dropout9 = Dropout(0.5, name='dropout9')(fire9)
    # conv10 output shape = (?, 27, 27, nb_classes)
    conv10 = Conv2D(filters=nb_classes, kernel_size=1, activation='relu', kernel_initializer='he_normal', name='conv10')(dropout9)
    conv10 = BatchNormalization(name='conv10_bn')(conv10)
    # avgpool10, softmax output shape = (?, nb_classes)
    avgpool10 = GlobalAveragePooling2D(name='pool10')(conv10)
    softmax = Activation('softmax', name='loss')(avgpool10)

    model = Model(input=input_image, output=[softmax])
    return model