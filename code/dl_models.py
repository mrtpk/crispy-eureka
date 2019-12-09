import keras
from keras_drop_block import DropBlock2D
from keras.models import Model, Sequential
import keras.layers as L
from keras import backend as K

def upsampler_model_16_64(shape):
    """ Simple resize with interpolation """
    model = Sequential()
    model.add(L.UpSampling2D(input_shape=shape, interpolation='bilinear'))
    model.add(L.UpSampling2D(input_shape=shape, interpolation='bilinear'))
    model.summary()
    return model 

def upsampler_trainable_model_16_64(shape):
    """ Trainable Deconv2D Layers 
    Goal is to train the upsamplers independently over input pointclouds in Spherical view
    """
    model = Sequential()
    model.add(L.Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=shape))
    model.add(L.Conv2DTranspose(1, (1,1), strides=(2,2), input_shape=shape))
    model.summary()
    return model 


def get_lodnn_model(shape=(400, 200, 6)):
    """
    Returns LoDNN model.
    Note: Instead of max-unpooling layer, deconvolution is used[see d_layer_01].
    """
    # encoder
    e_layer_01 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu', input_shape=shape)
    e_layer_02 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu')

    e_layer_03 = L.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')

    # context module
    c_layer_01 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation='elu')
    c_layer_01_dropout = L.SpatialDropout2D(rate=0.25)

    c_layer_02 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 1), activation='elu')
    c_layer_02_dropout = L.SpatialDropout2D(rate=0.25)

    c_layer_03 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(4, 2), activation='elu')
    c_layer_03_dropout = L.SpatialDropout2D(rate=0.25)

    c_layer_04 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(8, 4), activation='elu')
    c_layer_04_dropout = L.SpatialDropout2D(rate=0.25)

    c_layer_05 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(16, 8), activation='elu')
    c_layer_05_dropout = L.SpatialDropout2D(rate=0.25)

    c_layer_06 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(32, 16), activation='elu')
    c_layer_06_dropout = L.SpatialDropout2D(rate=0.25)

    c_layer_07 = L.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(64, 32), activation='elu')
    c_layer_07_dropout = L.SpatialDropout2D(rate=0.25)

    c_layer_08 = L.Conv2D(filters=32, kernel_size=(1,1),  strides=1, padding='same')

    # decoder
    # d_layer_01: Deconvolution layer is instead of max unpooling layer 
    d_layer_01 = L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')
    d_layer_02 = L.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu')
    d_layer_03 = L.Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu')
    d_layer_04 = L.Activation('softmax')
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


def get_unet_model(shape,
                   nb_filters_0=8,
                   exp=1,
                   conv_size=3,
                   initialization='he_normal',
                   activation="relu",
                   subsample_ratio=1,
                   output_channels=1):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    inputs = L.Input(shape)

    conv1 = L.Conv2D(nb_filters_0, conv_size,
                   activation=activation, padding='same', kernel_initializer=initialization)(inputs)

    bn1 = L.BatchNormalization()(conv1)

    drop1 = L.SpatialDropout2D(0.3)(bn1)

    pool1 = L.MaxPooling2D(pool_size=(1, 2))(drop1)

    conv2 = L.Conv2D(nb_filters_0 * 2 ** (1 * exp), conv_size,
                   activation=activation, padding='same', kernel_initializer=initialization)(pool1)

    bn2 = L.BatchNormalization()(conv2)

    drop2 = L.SpatialDropout2D(0.3)(bn2)

    pool2 = L.MaxPooling2D(pool_size=(1, 2))(drop2)

    conv3 = L.Conv2D(nb_filters_0 * 2 ** (2 * exp), conv_size,
                   activation=activation, padding='same', kernel_initializer=initialization)(pool2)

    bn3 = L.BatchNormalization()(conv3)

    drop3 = L.SpatialDropout2D(0.3)(bn3)

    pool3 = L.MaxPooling2D(pool_size=(1, 2))(drop3)

    drop4 = L.SpatialDropout2D(0.3)(pool3)

    up5 = L.Conv2D(nb_filters_0 * 2 ** (2 * exp), conv_size,
                 activation=activation, padding='same',
                 kernel_initializer=initialization)(L.UpSampling2D(size=(1, 2))(drop4))

    merge5 = L.concatenate([drop3, up5], axis=channel_axis)

    conv5 = L.Conv2D(nb_filters_0 * 2 ** (2 * exp), conv_size,
                   activation=activation, padding='same', kernel_initializer=initialization)(merge5)

    bn5 = L.BatchNormalization()(conv5)

    up6 = L.Conv2D(nb_filters_0 * 2 ** (2 * exp), conv_size,
                 activation=activation, padding='same',
                 kernel_initializer=initialization)(L.UpSampling2D(size=(1, 2))(bn5))

    merge6 = L.concatenate([conv2, up6], axis=channel_axis)

    conv6 = L.Conv2D(nb_filters_0 * 2 ** (1 * exp), conv_size,
                   activation=activation, padding='same', kernel_initializer=initialization)(merge6)

    bn6 = L.BatchNormalization()(conv6)

    up7 = L.Conv2D(nb_filters_0 * 2 ** (1 * exp), conv_size,
                 activation=activation, padding='same',
                 kernel_initializer=initialization)(L.UpSampling2D(size=(1, 2))(bn6))

    merge7 = L.concatenate([conv1, up7], axis=channel_axis)

    conv7 = L.Conv2D(nb_filters_0, conv_size,
                   activation=activation, padding='same', kernel_initializer=initialization)(merge7)

    if subsample_ratio % 2 == 0:
        conv7 = L.Conv2D(nb_filters_0, conv_size,
                                    activation=activation,
                                    padding='same',
                                    kernel_initializer=initialization)(L.UpSampling2D(size=(2, 1))(conv7))

    if subsample_ratio % 4 == 0:
        conv7 = L.Conv2D(nb_filters_0, conv_size,
                                    activation=activation,
                                    padding='same',
                                    kernel_initializer=initialization)(L.UpSampling2D(size=(2, 1))(conv7))

    bn7 = L.BatchNormalization()(conv7)

    conv8 = L.Conv2D(nb_filters_0, conv_size,
                   activation=activation, padding='same', kernel_initializer=initialization)(bn7)

    bn8 = L.BatchNormalization()(conv8)

    conv9 = L.Conv2D(output_channels, conv_size, padding='same', activation='sigmoid')(bn8)

    model = Model(inputs=inputs, outputs=conv9)

    return model

# def get_unet_model(shape = (400, 200, 6), subsample_ratio=1):
#     inputs = L.Input(shape)
#     conv1 = L.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
#     pool1 = L.MaxPooling2D(pool_size=(1, 2))(conv1)
#     conv2 = L.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
#     pool3 = L.MaxPooling2D(pool_size=(1, 2))(conv2)
#     conv4 = L.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#     drop4 = L.Dropout(0.5)(conv4)
#     pool4 = L.MaxPooling2D(pool_size=(1, 2))(drop4)
#     drop5 = L.Dropout(0.5)(pool4)
#
#     up6 = L.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(L.UpSampling2D(size = (1,2))(drop5))
#     merge6 = L.concatenate([drop4,up6], axis = 3)
#     conv6 = L.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#
#     up7 = L.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(L.UpSampling2D(size = (1,2))(conv6))
#     merge7 = L.concatenate([conv2,up7], axis = 3)
#     conv7 = L.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#     up8 = L.Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(L.UpSampling2D(size = (1,2))(conv7))
#     merge8 = L.concatenate([conv1,up8], axis = 3)
#     conv8 = L.Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#
#     if subsample_ratio % 2 == 0:
#         conv8 = L.Conv2D(8, 3, activation= 'relu', padding='same', kernel_initializer='he_normal')(
#             L.UpSampling2D(size=(2,1))(conv8))
#
#     if subsample_ratio % 4 == 0:
#         conv8 = L.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
#             L.UpSampling2D(size=(2, 1))(conv8))
#
#     conv9 = L.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#
#     conv10 = L.Conv2D(1, 1, activation = 'sigmoid')(conv9)
#
#     model = keras.models.Model(input = inputs, output = conv10)
#
#     return model

def conv2dLeakyDownProj(layer_input, filters, output,f_size=3):
    d = L.Conv2D(filters, kernel_size=f_size, strides=(1, 2), padding='same')(layer_input)
    d = L.LeakyReLU(alpha=0.2)(d)
    d = L.Conv2D(output, kernel_size=1, padding='same')(d)
    d = L.LeakyReLU(alpha=0.2)(d)
    return d


def conv2dLeakyProj(layer_input, filters,output, f_size=3):
    d = L.Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
    d = L.LeakyReLU(alpha=0.2)(d)
    d = L.Conv2D(output, kernel_size=1, padding='same')(d)
    d = L.LeakyReLU(alpha=0.2)(d)
    return d


def conv2dLeaky(layer_input, filters, f_size=3):
    d = L.Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
    d = L.LeakyReLU(alpha=0.2)(d)
    return d


def u_net6(shape, filters=4, f_size=3, int_space=4, output_channels=4, rate=.3, subsample_ratio=1):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    inputs = L.Input(shape)
    #  inputN = GaussianNoise(rate)(inputs)

    dblock0 = DropBlock2D(block_size=5, keep_prob=0.8, name="dropblock0")(inputs)

    pool1 = conv2dLeakyDownProj(dblock0, filters, int_space, f_size=f_size)
    pool1 = L.Dropout(rate)(pool1)
    pool2 = conv2dLeakyDownProj(pool1, filters, int_space, f_size=f_size)
    pool2 = L.Dropout(rate)(pool2)
    pool3 = conv2dLeakyDownProj(pool2, filters, int_space, f_size=f_size)
    pool3 = L.Dropout(rate)(pool3)
    pool4 = conv2dLeakyDownProj(pool3, filters, int_space, f_size=f_size)
    pool4 = L.Dropout(rate)(pool4)
    pool5 = conv2dLeakyDownProj(pool4, filters, int_space, f_size=f_size)
    pool5 = L.Dropout(rate)(pool5)
    up4 = L.concatenate([L.UpSampling2D(size=(1, 2))(pool5), pool4], axis=channel_axis)
    conv4 = conv2dLeakyProj(up4, filters, int_space, f_size=f_size)
    conv4 = L.Dropout(rate)(conv4)
    up3 = L.concatenate([L.UpSampling2D(size=(1, 2))(conv4), pool3], axis=channel_axis)
    conv3 = conv2dLeakyProj(up3, filters, int_space, f_size=f_size)
    conv3 = L.Dropout(rate)(conv3)
    up2 = L.concatenate([L.UpSampling2D(size=(1, 2))(conv3), pool2], axis=channel_axis)
    conv2 = conv2dLeakyProj(up2, filters, int_space, f_size=f_size)
    conv2 = L.Dropout(rate)(conv2)
    up1 = L.concatenate([L.UpSampling2D(size=(1, 2))(conv2), pool1], axis=channel_axis)
    conv1 = conv2dLeakyProj(up1, filters, int_space, f_size=f_size)
    conv1 = L.Dropout(rate)(conv1)
    up0 = L.concatenate([L.UpSampling2D(size=(1, 2))(conv1), inputs], axis=channel_axis)
    conv0 = conv2dLeaky(up0, 2, f_size=f_size)

    if subsample_ratio % 2 == 0:
        conv0 = L.Conv2D(8, 3, activation= 'relu', padding='same', kernel_initializer='he_normal')(
            L.UpSampling2D(size=(2, 1))(conv0))

    if subsample_ratio % 4 == 0:
        conv0 = L.Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
            L.UpSampling2D(size=(2, 1))(conv0))

    conv0 = L.Conv2D(1, 1, activation='sigmoid')(conv0)
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
        squeeze = L.Conv2D(filters=s_1x1, kernel_size=1, activation='relu', kernel_initializer='glorot_uniform', name=name + '/squeeze1x1')(x)
        squeeze = L.BatchNormalization(name=name + '/squeeze1x1_bn')(squeeze)

        # Needed to merge layers expand_1x1 and expand_3x3.
        expand_1x1 = L.Conv2D(filters=e_1x1, kernel_size=1, activation='relu', kernel_initializer='glorot_uniform', name=name + '/expand1x1')(
            squeeze)

        # Pad the border with zeros. Not needed as border_mode='same' will do the same.
        # expand_3x3 = ZeroPadding2D(padding=(1, 1), name=name+'_expand_3x3_padded')(squeeze)
        expand_3x3 = L.Conv2D(filters=e_3x3, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform',
                                   name=name + '/expand3x3')(squeeze)
        # Concat in the channel dim
        expand_merge = L.concatenate([expand_1x1, expand_3x3], axis=concat_axis, name=name + '/concat')
        return expand_merge

    return layer


def SqueezeNet(shape, nb_classes):
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

    input_image = L.Input(shape=shape)
    # conv1 output shape = (113, 113, 64)

    conv1 = L.Conv2D(filters=64, kernel_size=3, activation='relu', strides=(2, 2), kernel_initializer='glorot_uniform', name='conv1')(
        input_image)
    # maxpool1 output shape = (56, 56, 64)
    maxpool1 = L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(conv1)
    # fire2 output shape = (?, 56, 56, 128)
    fire2 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire2')(maxpool1)
    # fire3 output shape = (?, 56, 56, 128)
    fire3 = FireModule(s_1x1=16, e_1x1=64, e_3x3=64, name='fire3')(fire2)
    # maxpool3 output shape = (?, 27, 27, 128)
    maxpool3 = L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(fire3)
    # fire4 output shape = (?, 56, 56, 256)
    fire4 = FireModule(s_1x1=32, e_1x1=128, e_3x3=128, name='fire4')(maxpool3)
    # fire5 output shape = (?, 56, 56, 256)
    fire5 = FireModule(s_1x1=32, e_1x1=128, e_3x3=128, name='fire5')(fire4)
    # maxpool5 output shape = (?, 27, 27, 256)
    maxpool5 = L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(fire5)
    # fire6 output shape = (?, 27, 27, 384)
    fire6 = FireModule(s_1x1=48, e_1x1=192, e_3x3=192, name='fire6')(maxpool5)
    # fire7 output shape = (?, 27, 27, 384)
    fire7 = FireModule(s_1x1=48, e_1x1=192, e_3x3=192, name='fire7')(fire6)
    # fire8 output shape = (?, 27, 27, 512)
    fire8 = FireModule(s_1x1=64, e_1x1=256, e_3x3=256, name='fire8')(fire7)
    # fire9 output shape = (?, 27, 27, 512)
    fire9 = FireModule(s_1x1=64, e_1x1=256, e_3x3=256, name='fire9')(fire8)
    # Dropout after fire9 module.
    dropout9 = L.Dropout(0.5, name='dropout9')(fire9)
    # conv10 output shape = (?, 27, 27, nb_classes)
    conv10 = L.Conv2D(filters=nb_classes, kernel_size=1, activation='relu', kernel_initializer='he_normal', name='conv10')(dropout9)
    conv10 = L.BatchNormalization(name='conv10_bn')(conv10)
    # avgpool10, softmax output shape = (?, nb_classes)
    avgpool10 = L.GlobalAveragePooling2D(name='pool10')(conv10)
    softmax = L.Activation('softmax', name='loss')(avgpool10)

    model = Model(input=input_image, output=[softmax])
    return model