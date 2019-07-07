import keras
import numpy as np

def get_model():
    '''
    Returns LoDNN model.
    Note: Instead of max-unpooling layer, deconvolution is used[see d_layer_01].
    '''
    # encoder
    e_layer_01 = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='elu', input_shape=(400, 200, 6))
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

def get_features(points):
    '''
    Returns features of the point cloud as stacked grayscale images.
    Shape of the output is (400x200x6).
    '''
    side_range=(-10, 10)
    fwd_range=(6, 46)
    res=.1

    # calculate the image dimensions
    img_width = int((side_range[1] - side_range[0])/res)
    img_height = int((fwd_range[1] - fwd_range[0])/res)
    number_of_grids = img_height * img_width

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3]

    norm_z_lidar = z_lidar # assumed that the z values are normalised
    
    # MAPPING
    # Mappings from one point to grid 
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution(grid size)
    x_img_mapping = (-y_lidar/res).astype(np.int32) # x axis is -y in LIDAR
    y_img_mapping = (x_lidar/res).astype(np.int32)  # y axis is -x in LIDAR; will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img_mapping -= int(np.floor(side_range[0]/res))
    y_img_mapping -= int(np.floor(fwd_range[0]/res))

    # Linerize the mappings to 1D
    lidx = ((-y_img_mapping) % img_height) * img_width + x_img_mapping

    # Feature extraction
    # count of points per grid
    count_input = np.ones_like(norm_z_lidar)
    binned_count = np.bincount(lidx, count_input, minlength = number_of_grids)
    # sum reflectance
    binned_reflectance =  np.bincount(lidx, r_lidar, minlength = number_of_grids)

    # sum elevation 
    binned_elevation = np.bincount(lidx, norm_z_lidar, minlength = number_of_grids)

    # Finding mean!
    binned_mean_reflectance = np.divide(binned_reflectance, binned_count, out=np.zeros_like(binned_reflectance), where=binned_count!=0.0)
    binned_mean_elevation = np.divide(binned_elevation, binned_count, out=np.zeros_like(binned_elevation), where=binned_count!=0.0)
    o_mean_elevation = binned_mean_elevation.reshape(img_height, img_width)

    # Standard devation stuff
    binned_sum_var_elevation = np.bincount(lidx, np.square(norm_z_lidar - o_mean_elevation[-y_img_mapping, x_img_mapping]), minlength = number_of_grids)
    binned_divide = np.divide(binned_sum_var_elevation, binned_count, out=np.zeros_like(binned_sum_var_elevation), where=binned_count!=0.0)
    binned_std_elevation = np.sqrt(binned_divide)

    # minimum and maximum
    sidx = lidx.argsort()
    idx = lidx[sidx]
    val = norm_z_lidar[sidx]

    m_idx = np.flatnonzero(np.r_[True,idx[:-1] != idx[1:]])
    unq_ids = idx[m_idx]

    o_max_elevation = np.zeros([img_height, img_width], dtype=np.float64)
    o_min_elevation = np.zeros([img_height, img_width], dtype=np.float64)

    o_max_elevation.flat[unq_ids] = np.maximum.reduceat(val, m_idx)
    o_min_elevation.flat[unq_ids] = np.minimum.reduceat(val, m_idx)

    norm_binned_count = binned_count # normalise the output
    # reshape all other things
    o_count            = norm_binned_count.reshape(img_height, img_width)
    o_mean_reflectance = binned_mean_reflectance.reshape(img_height, img_width)
    o_std_elevation    = binned_std_elevation.reshape(img_height, img_width)
    return np.dstack([o_count, o_mean_reflectance, o_max_elevation, o_min_elevation, o_mean_elevation, o_std_elevation])
