# %matplotlib inline
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from helpers import data_loaders as dls
from helpers.viz import plot_history  # plot
from helpers.logger import Logger
import utils
import dl_models
import keras  # this is required
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as k
from keras_custom_loss import binary_focal_loss  # weightedLoss2
# import helpers.generate_gt as generate_gt
# cyclic lr
import keras_contrib
# from helpers.sgdr import SGDRScheduler
# from helpers.lr_finder import LRFinder, get_lr_for_model


# def test_feature(model_name, feature, view, dataset, sequences=None):
#     for subsample_ratio, s in zip([1, 2, 4], ['', 'Subsampled_32', 'Subsampled_16']):
    

#     return

def test_all(model_name, view, dataset, sequences=None):
    """ 
    All tests simple features, geomtric features with and without subsampling at 16, 32, 64 if possible
    """

    all_results = {}
    test_name = {}
    prefix = 'Classical_'

    for subsample_ratio, s in zip([1, 2, 4], ['', 'Subsampled_32', 'Subsampled_16']):
        for add_geometrical_features, g in zip([False, True], ['', 'Geometric_']):
            for compute_HOG, h in zip([False], ['']):  # change one of them to true when testing all
                for compute_eigen, e in zip([100, 0], ['Eigen_', '']): #training for 100 kneighbors
                    subsample_flag = True if subsample_ratio % 2 == 0 else False
                    key = (add_geometrical_features, subsample_flag, compute_HOG, compute_eigen)
                    test_name[key] = prefix+g+s+h+e
                    feature_flags = {}
                    feature_flags['add_geometrical_features']=add_geometrical_features
                    feature_flags['subsample_flag']=subsample_flag
                    feature_flags['subsample_ratio']=subsample_ratio
                    feature_flags['compute_HOG']=compute_HOG
                    feature_flags['compute_eigen'] = compute_eigen
                    print(test_name[key])
                    all_results[key] = test_road_segmentation(model_name,
                                                            feature_flags=feature_flags,
                                                            view=view,                                        
                                                            test_name=test_name[key],
                                                            dataset=dataset,
                                                            sequences=sequences)

    return all_results


def test_road_segmentation(model_name, feature_flags,
                           view='bev', test_name='Test_',
                           dataset='kitti', sequences=None):

    feature_flags['model_name'] = model_name
    KPC = utils.KittiPointCloudClass(feature_flags=feature_flags,
                                     view=view,
                                     dataset=dataset,
                                     sequences=sequences)
    NAME = 'experiment0'  # name of experiment
    # It is better to create a folder with runid in the experiment folder
    _EXP, _LOG, _TMP, _RUN_PATH = dls.create_dir_struct(KPC.dataset_path, NAME)
    logger = Logger('EXP0', _LOG + 'experiment0.log')
    logger.debug('Logger EXP0 int')
    # paths
    run_id = model_name + '_' + utils.get_unique_id()
    path = utils.create_run_dir(_RUN_PATH, run_id)
    callbacks = utils.get_basic_callbacks(path)
    
    #KPC.write_all_features()
    n_channels = KPC.n_channels

    #get dataset should load all paths/write out txt file to write output feature maps
    f_train, f_valid, f_test, gt_train, gt_valid, gt_test = KPC.get_dataset()

    _, n_row, n_col, _ = f_train.shape

    if 'unet' in model_name:
        multiple_of = 16 if model_name == 'unet' else 32
        nrpad = multiple_of - n_row % multiple_of if n_row % multiple_of != 0 else 0
        ncpad = multiple_of - n_col % multiple_of if n_col % multiple_of != 0 else 0

        f_train = np.pad(f_train, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        f_valid = np.pad(f_valid, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        f_test = np.pad(f_test, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')

        gt_train = np.pad(gt_train, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        gt_train[:, n_col:, n_row:, 1] = 1  # set the new pixels to non-road
        gt_valid = np.pad(gt_valid, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        gt_valid[:, n_col:, n_row:, 1] = 1  # set the new pixels to non-road
        gt_test = np.pad(gt_test, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        gt_test[:, n_col:, n_row:, 1] = 1  # set the new pixels to non-road

        gt_train = utils.remove_bg_channel(gt_train)
        gt_valid = utils.remove_bg_channel(gt_valid)
        gt_test = utils.remove_bg_channel(gt_test)

    print(f_test.shape, gt_test.shape)

    # All training params to be added here
    training_config = {
        "loss_function": "binary_crossentropy",
        "learning_rate": 1e-3,
        "batch_size": 3,
        "epochs": 120,
        "optimizer": "keras.optimizers.Adam"  # "keras.optimizers.Nadam"
    }
    # we create two instances with the same arguments
    dict_aug_args = dict(horizontal_flip=True)
    seed = 1
    train_datagen = ImageDataGenerator(**dict_aug_args)
    mask_datagen = ImageDataGenerator(**dict_aug_args)
    # this is the augmentation configuration we will use for training
    read_all_into_memory = True
    if read_all_into_memory:
        # Provide the same seed and keyword arguments to the fit and flow methodsf_
        train_generator = train_datagen.flow(f_train,
                                            batch_size=training_config['batch_size'], shuffle=True, seed=seed)
        mask_generator = mask_datagen.flow(gt_train, batch_size=training_config['batch_size'], shuffle=True, seed=seed)

        train_iterator = zip(train_generator, mask_generator)
    else:  # read flow from directory
        """
        source : https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L405
        """
        train_path = '../dataset/KITTI/dataset/data_road_velodyne/training/feature_maps/'
        train_datagen = train_datagen.flow_from_directory(horizontal_flip=True,
                                                        directory=train_path,
                                                        target_size=(32, 32),
                                                        batch_size=32,
                                                        shuffle=False)
    # Validation
    seed = 2
    valid_datagen = ImageDataGenerator(**dict_aug_args)
    valid_mask_datagen = ImageDataGenerator(**dict_aug_args)
    valid_generator = valid_datagen.flow(f_valid, batch_size=1, shuffle=True, seed=seed)
    valid_mask_generator = valid_mask_datagen.flow(gt_valid, batch_size=1, shuffle=True, seed=seed)

    valid_iterator = zip(valid_generator, valid_mask_generator)
    # Test
    # test_datagen = ImageDataGenerator() #horizontal_flip=True #add this ?
    # test_iterator = test_datagen.flow(f_test, gt_test,
    #                 batch_size=1, shuffle=True)
    
    #TODO : Make sure that models are chosen based on the view (BEV, front)
    if model_name == 'lodnn':
        model = dl_models.get_lodnn_model(shape=(400, 200, n_channels))
    elif model_name == 'unet':
        model = dl_models.get_unet_model(input_size=(f_train.shape[1], f_train.shape[2], n_channels),
                                         subsample_ratio=subsample_ratio)

    elif model_name == 'unet6':
        model = dl_models.u_net6(shape=(f_train.shape[1], f_train.shape[2], n_channels),
                                 filters=512,
                                 int_space=32,
                                 output_channels=2)
    elif model_name == 'squeeze':
        #TODO this is not a squeezeSeg network, it is a simple classifier (encoder part only)
        model = dl_models.SqueezeNet(2, input_shape=(f_train.shape[1], f_train.shape[2], n_channels))
    else:
        raise ValueError("Acceptable values for model parameter are 'lodnn', 'unet', 'unet6'.")

    model.summary()

    # Add more callbacks
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
    # callbacks = callbacks + [early_stopping, reduce_lr]

    clr_custom = keras_contrib.callbacks.CyclicLR(base_lr=0.0001, max_lr=0.01,
                                                  mode='triangular', gamma=0.99994,
                                                  step_size=120000)
    # clr_triangular._reset(new_base_lr=0.003, new_max_lr=0.009)
    # clr_custom = SGDRScheduler(min_lr=1e-3, max_lr=1e-2, steps_per_epoch=1e4, lr_decay=0.9, cycle_length=1, mult_factor=2)

    callbacks = callbacks + [clr_custom]
    optimizer = keras.optimizers.Nadam()  # SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #eval(training_config["optimizer"])(lr=training_config["learning_rate"])

    # todo: move from class weight to focal loss
    if dataset == 'kitti':
        class_weight = [10.123203420301278, 1.0]
    else:#semantic kitti
        class_weight = [4.431072018928066, 1.0]

    model.compile(loss=binary_focal_loss(),
                  # loss=weightedLoss2(keras.losses.binary_crossentropy, class_weight),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # get learning rate plots
    # get_lr_for_model(model, train_iterator)
    m_history = model.fit_generator(train_iterator,
                                    samples_per_epoch=f_train.shape[0],
                                    nb_epoch=training_config["epochs"],
                                    steps_per_epoch=int(f_train.shape[0] // training_config["batch_size"]),
                                    verbose=1,
                                    callbacks=callbacks,
                                    validation_data=valid_iterator,
                                    validation_steps=int(f_valid.shape[0] / 1))

    model.save("{}/model/final_model.h5".format(path))
    plot_history(m_history)
    png_name = '{}.png'.format(path + '/' + test_name)

    plt.savefig(png_name)
    plt.close()
    # test set prediction

    all_pred = []
    for f in f_test:
        all_pred.append(model.predict(np.expand_dims(f, axis=0))[0, :, :, :])
    all_pred = np.array(all_pred)

    if 'unet' in model_name:
        result = utils.measure_perf(path, all_pred[:, :n_row, :n_col, :], gt_test[:, :n_row, :n_col, :])
    else:
        result = utils.measure_perf(path, all_pred, gt_test)
    print('------------------------------------------------')
    for key in result:
        print(key, result)

    result = {"name": _NAME,
              "test_name": test_name,
              "run_id": run_id,
              "dataset": "KITTI",
              "training_config": training_config,
              "z_min": str(KPC.z_min),
              "z_max": str(KPC.z_max),
              "COUNT_MAX": str(KPC.COUNT_MAX)
              }
    with open('{}/details.json'.format(path), 'w') as f:
        json.dump(result, f)
    return result

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Road Segmentation")
    parser.add_argument('--feature', default='classic', type=str, help='architecture to use for evaluation')
    parser.add_argument('--model', default='lodnn', type=str, help='architecture to use for evaluation')
    parser.add_argument('--cuda_device', default='0', type=str, help='GPU to use')
    parser.add_argument('--view', default='bev', type=str,
                        help='BEV or top view, different DNNs to choose based on view')
    parser.add_argument('--dataset', default='kitti', type=str, help='Dataset to use for training')
    parser.add_argument('--sequences', default='', type=str, help='Sequences to use in case of SemanticKitti. '
                                                                  'Values must be comma separated')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print(args.cuda_device)
    # Tensorflow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.85

    # create a session with the above option specified
    k.tensorflow_backend.set_session(tf.Session(config=config))
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    sequences = None
    if len(args.sequences) > 0:
        sequences = args.sequences.split(',')

    test_all(args.model, args.view, args.dataset, sequences)
