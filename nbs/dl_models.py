import keras

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