#%matplotlib inline
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

def dataload_gen(pathsX, pathsY):
    for pathpc, pathimg in pathsX, pathsY:
        pc = dls.load_bin_file((pathpc))
        X = KPC.get_features(pc)
        img = dls.get_image(pathimg, is_color=True, rgb=False)
        Y = utils.kitti_gt(img)
        yield X, Y

if __name__ == "__main__":
    PATH = '../' # path of the repo.
    _NAME = 'experiment0' # name of experiment
    # It is better to create a folder with runid in the experiment folder
    _EXP, _LOG, _TMP, _RUN_PATH = dls.create_dir_struct(PATH, _NAME)
    logger = Logger('EXP0', _LOG + 'experiment0.log')
    logger.debug('Logger EXP0 int')

    add_geometrical_features = True #add geometrical features flag
    subsample_flag = True 
    #create dataclass
    KPC = utils.KittiPointCloudClass(dataset_path=PATH, 
                                     add_geometrical_features=add_geometrical_features,
                                     subsample=subsample_flag)
    
    #limit_index = -1 for all dataset while i > 0 for smaller #samples
    f_train, f_valid, f_test, gt_train, gt_valid, gt_test = KPC.get_dataset(limit_index = -1)

    print(f_test.shape, gt_test.shape)
    
    # All training params to be added here
    training_config = {
        "loss_function" : "binary_crossentropy",
        "learning_rate" : 1e-4,
        "batch_size"    : 3,
        "epochs"        : 5,
        "optimizer"     : "keras.optimizers.Adam"
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
    test_datagen = ImageDataGenerator() #horizontal_flip=True #add this ?
    test_iterator = test_datagen.flow(f_test, gt_test, 
                    batch_size=1, shuffle=True)
    
    run_id = utils.get_unique_id()
    path = utils.create_run_dir(_RUN_PATH, run_id)
    
    if add_geometrical_features:
        model = dl_models.get_lodnn_model(shape=(400,200, 9))
        # model = dl_models.get_unet_model(shape=(400,200,9))
    else:
        model = dl_models.get_lodnn_model(shape=(400,200, 6))
        # model = dl_models.get_unet_model(shape=(400,200,6))
    
    model.summary()
    callbacks = utils.get_basic_callbacks(path)
    
    # Add more callbacks
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
    # callbacks = callbacks + [early_stopping, reduce_lr]
    
    optimizer = eval(training_config["optimizer"])(lr=training_config["learning_rate"])
    model.compile(loss=training_config["loss_function"],
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
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
    
    #test set prediction
    all_pred = model.predict_generator(test_iterator,
                                       steps = f_test.shape[0])
    all_pred = np.array(all_pred)

    result = utils.measure_perf(path, all_pred, gt_test)
    print('------------------------------------------------')
    for key in result:
        print(key, result)
    #plot single prediction
    single_pred = all_pred[1,:,:,:]
    #plot([[single_pred]])
    
    #argmax to obtain segmentation
    plot([[1-utils.apply_argmax(single_pred)]])
    plt.show()

    result = {"name" : _NAME,
           "run_id" : run_id,
           "dataset": "KITTI",
           "training_config" : training_config,
           }
    with open('{}/details.json'.format(path), 'w') as f:
        json.dump(result, f)
