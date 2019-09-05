#%matplotlib inline
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from helpers import data_loaders as dls
from helpers.viz import plot, plot_history
from helpers.logger import Logger
import utils
import dl_models
import keras #this is required
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as k
from keras_custom_loss import weightedLoss2
#cyclic lr
import keras_contrib
from helpers.sgdr import SGDRScheduler
from helpers.lr_finder import LRFinder, get_lr_for_model

def test_all(model_name, view, dataset, sequences=None):
    """ 
    Test simple features, geomtric features with and without subsampling at 16, 32, 64 if possible
    """

    all_results = {}
    test_name = {}
    prefix = 'Classical_'
    for add_geometrical_features, g in zip([True, False], ['Geometric_', '']):
        for subsample_ratio, s in zip([1, 2, 4], ['', 'Subsampled_32', 'Subsampled_16']):
            for compute_HOG, h in zip([False], ['']): #change one of them to true when testing all
                subsample_flag = True if subsample_ratio % 2 == 0 else False
                key = (add_geometrical_features, subsample_flag, compute_HOG)
                test_name[key] = prefix+g+s+h
                print(test_name[key])
                all_results[key] = test_road_segmentation(model_name,
                                                          add_geometrical_features=add_geometrical_features,
                                                          subsample_flag =subsample_flag,
                                                          compute_HOG = compute_HOG,
                                                          view=view,
                                                          subsample_ratio=subsample_ratio,
                                                          test_name = test_name[key],
                                                          dataset=dataset,
                                                          sequences=sequences)
    return all_results

def get_KPC_setup(model_name,
                  add_geometrical_features = True,
                  subsample_flag = True,
                  compute_HOG = False,
                  view='bev',
                  subsample_ratio=1,
                  dataset='kitti',
                  sequences=None):
    PATH = '../' # path of the repo.
    _NAME = 'experiment0' # name of experiment
    # It is better to create a folder with runid in the experiment folder
    _EXP, _LOG, _TMP, _RUN_PATH = dls.create_dir_struct(PATH, _NAME)
    logger = Logger('EXP0', _LOG + 'experiment0.log')
    logger.debug('Logger EXP0 int')
    #paths
    run_id = model_name + '_' +utils.get_unique_id()
    path = utils.create_run_dir(_RUN_PATH, run_id)
    callbacks = utils.get_basic_callbacks(path)
    
    #create dataclass
    KPC = utils.KittiPointCloudClass(dataset_path=PATH, 
                                     add_geometrical_features=add_geometrical_features,
                                     subsample=subsample_flag,
                                     compute_HOG=compute_HOG,
                                     view=view,
                                     subsample_ratio=subsample_ratio,
                                     dataset=dataset,
                                     sequences=sequences)
    # number of channels in the images
    n_channels = 6
    if add_geometrical_features:
        n_channels += 3
    if compute_HOG:
        n_channels += 6

    return _NAME, run_id, path, callbacks, KPC, n_channels

def test_road_segmentation(model_name,
                           add_geometrical_features = True,
                           subsample_flag = True,
                           compute_HOG = False,
                           view='bev',
                           subsample_ratio=1,
                           test_name='Test_',
                           dataset='kitti',
                           sequences=None):

    _NAME, run_id, path, callbacks, KPC, n_channels = get_KPC_setup(model_name,
                                                                    add_geometrical_features = add_geometrical_features,
                                                                    subsample_flag = subsample_flag,
                                                                    view=view,
                                                                    compute_HOG = compute_HOG,
                                                                    subsample_ratio=subsample_ratio,
                                                                    dataset=dataset,
                                                                    sequences=sequences)

    #limit_index = -1 for all dataset while i > 0 for smaller #samples
    f_train, f_valid, f_test, gt_train, gt_valid, gt_test = KPC.get_dataset(limit_index = -1)
    
    _, n_row, n_col, _ = f_train.shape

    if 'unet' in model_name:
        multiple_of = 16 if model_name =='unet' else 32
        nrpad =  multiple_of - n_row % multiple_of if n_row % multiple_of != 0 else 0
        ncpad =  multiple_of - n_col % multiple_of if n_col % multiple_of != 0 else 0

        f_train = np.pad(f_train, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        f_valid = np.pad(f_valid, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        f_test = np.pad(f_test, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')

        gt_train = np.pad(gt_train, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        gt_train[:,n_col:, n_row:, 1] = 1 # set the new pixels to non-road
        gt_valid = np.pad(gt_valid, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        gt_valid[:, n_col:, n_row:, 1] = 1  # set the new pixels to non-road
        gt_test = np.pad(gt_test, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')
        gt_test[:, n_col:, n_row:, 1] = 1  # set the new pixels to non-road
    print(f_test.shape, gt_test.shape)
    
    # All training params to be added here
    training_config = {
        "loss_function" : "binary_crossentropy",
        "learning_rate" : 1e-3,
        "batch_size"    : 3,
        "epochs"        : 120,
        "optimizer"     : "keras.optimizers.Adam" #"keras.optimizers.Nadam"
    }

    #this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(horizontal_flip=True) 
    train_iterator = train_datagen.flow(f_train, gt_train, 
                    batch_size=training_config['batch_size'], shuffle=True)
    # Validation
    valid_datagen = ImageDataGenerator(horizontal_flip=True) 
    valid_iterator = valid_datagen.flow(f_valid, gt_valid, 
                    batch_size=1, shuffle=True)
    # Test 
    # test_datagen = ImageDataGenerator() #horizontal_flip=True #add this ?
    # test_iterator = test_datagen.flow(f_test, gt_test,
    #                 batch_size=1, shuffle=True)
    if model_name=='lodnn':
        model = dl_models.get_lodnn_model(shape=(400, 200, n_channels))
    elif model_name=='unet':
        model = dl_models.get_unet_model(input_size=(f_train.shape[1],f_train.shape[2], n_channels),
                                         subsample_ratio=subsample_ratio)

    elif model_name=='unet6':
        model = dl_models.u_net6(shape=(f_train.shape[1],f_train.shape[2], n_channels),
                                 filters=512,
                                 int_space=32,
                                 output_channels=2)
    elif model_name=='squeeze':
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
    optimizer = keras.optimizers.Nadam() #SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #eval(training_config["optimizer"])(lr=training_config["learning_rate"])

    # todo: move from class weight to focal loss
    if dataset == 'kitti':
        class_weight = [10.123203420301278, 1.0]
    else:
        class_weight = [4.431072018928066, 1.0]

    model.compile(loss=weightedLoss2(keras.losses.binary_crossentropy, class_weight),
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    #get learning rate plots
    #get_lr_for_model(model, train_iterator)
    m_history = model.fit_generator(train_iterator,
                      samples_per_epoch = f_train.shape[0],
                      nb_epoch=training_config["epochs"],
                      steps_per_epoch = int(f_train.shape[0] // training_config["batch_size"]),
                      verbose=1,
                      callbacks=callbacks,
                      validation_data=valid_iterator,
                      validation_steps = int(f_valid.shape[0]/1))
    
    model.save("{}/model/final_model.h5".format(path))
    plot_history(m_history)
    png_name = '{}.png'.format(path+'/'+test_name)
    
    plt.savefig(png_name)
    plt.close()
    #test set prediction

    all_pred = []
    for f in f_test:
        all_pred.append(model.predict(np.expand_dims(f, axis=0))[0,:,:,:])
    all_pred = np.array(all_pred)

    if 'unet' in model_name:
        result = utils.measure_perf(path, all_pred[:,:n_row,:n_col, :], gt_test[:,:n_row,:n_col, :])
    else:
        result = utils.measure_perf(path, all_pred, gt_test)
    print('------------------------------------------------')
    for key in result:
        print(key, result)
    
    result = {"name" : _NAME,
              "test_name" : test_name,
           "run_id" : run_id,
           "dataset": "KITTI",
           "training_config" : training_config,
           }
    with open('{}/details.json'.format(path), 'w') as f:
        json.dump(result, f)
    return result


if __name__ == "__main__":
    #ensure the front view ground truth exists
    # write_front_view_GT()
    
#    test_road_segmentation()
    parser = argparse.ArgumentParser(description="Road Segmentation")
    parser.add_argument('--model', default='lodnn', type=str, help='architecture to use for evaluation')
    parser.add_argument('--cuda_device', default='0', type=str, help='GPU to use')
    parser.add_argument('--view', default='bev', type=str, help='BEV or top view, different DNNs to choose based on view')
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
    # config.gpu_options.per_process_gpu_memory_fraction = 0.85

    # create a session with the above option specified
    k.tensorflow_backend.set_session(tf.Session(config=config))
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    sequences = None
    if len(args.sequences) > 0:
        sequences = args.sequences.split(',')

    test_all(args.model, args.view, args.dataset, sequences)