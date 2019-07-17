#%matplotlib inline
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

import os
import numpy as np
import tensorflow as tf
import keras
import sklearn.model_selection as sk

from PIL import Image as IM
from keras.callbacks import TensorBoard
import cv2
import json

from glob import glob

## import helpers
from helpers import data_loaders as dls
from helpers import pointcloud as pc
from helpers.viz import plot, plot_history
from helpers.logger import Logger
import utils
## import networks
import lodnn
import unet

from keras.preprocessing.image import ImageDataGenerator


def create_run_dir(path, run_id):
    path = path.replace("*run_id", str(run_id))
    model_dir = "{}/model/".format(path)
    output_dir = "{}/output/".format(path)
    log_dir = "{}/log/".format(path)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return path

def get_basic_callbacks(path):
    # tensorboard
    # to visualise `tensorboard --logdir="./path_to_log_dir" --port 6006`
    log_path = "{}/log/".format(path)
    tensorboard = TensorBoard(log_dir="{}/{}".format(log_path, time()))
    # save best model
    best_model_path = "{}/model/best_model.h5".format(path) #? .hd5
    save_the_best = keras.callbacks.ModelCheckpoint(filepath=best_model_path,
                                                    verbose=1, save_best_only=True)
    # save models after few epochs
    epoch_save_path = "{}/model/*.h5".format(path)
    save_after_epoch = keras.callbacks.ModelCheckpoint(filepath=epoch_save_path.replace('*', 'e{epoch:02d}-val_acc{val_acc:.2f}'),
                                                       monitor='val_acc', verbose=1, period = 1)
    return [tensorboard, save_the_best, save_after_epoch]

def apply_argmax(res):
    return np.argmax(res, axis=2)

def measure_perf(path, all_pred, all_gt, is_viz=False):
    result_path = "{}/output/*".format(path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    # NOTE: Now computing for 8 samples (can this be done on samples from the train and validation set?)
    
    F1, P, R, ACC = [], [], [], []
    FN, FP, TP, TN = [], [], [], []
    counter = 0
    diplay_thresh = 3
    display = []
    labels = []
    
    for i in range(all_pred.shape[0]):
        _f, _gt = all_pred[i], all_gt[i]
        p_road = apply_argmax(_f)       
        # get metrics
        gt_road = _gt[:, :, 0]
        fn, fp, tp, tn = utils.get_metrics_count(pred=p_road, gt=gt_road)
        f1, recall, precision, acc = utils.get_metrics(gt=gt_road, pred=p_road)
        F1.append(f1)
        P.append(precision)
        R.append(recall)
        ACC.append(acc)
        TP.append(tp)
        FP.append(fp)
        TN.append(tn)
        FN.append(fn)
        
        # for viz
        if is_viz is True:
            display.append([gt_road, res[:,:,0], p_road])
            labels.append(["gt", "Acc: {}".format(str(np.round(acc,2))), "F1: {}".format(str(np.round(f1,2)))])
            if (i+1) % diplay_thresh == 0:
                plot(display, labels, fontsize=10)
                plt.savefig(result_path.replace("*", str(i+1) + ".jpg"))
                display, labels = [], []
    if is_viz is True:
        plot(display, labels, fontsize=10)
        plt.savefig(result_path.replace("*", str(i+1) + ".jpg"))

    eps = np.finfo(np.float32).eps        
    _acc = (sum(TP)+sum(TN))/(sum(TP)+sum(FP)+sum(TN)+sum(FN) + eps)
    _recall = sum(TP)/(sum(TP) + sum(FN)+eps)
    _precision = sum(TP)/(sum(TP) + sum(FP)+eps)
    _f1 = 2*((precision * recall)/(precision + recall))    
    return _acc, _recall, _precision, _f1


def dataload_static(limit_index = 3):
    # NOTE: change limit_index to -1 to train on the whole dataset
    f_train = dls.process_pc(train_set["pc"][0:limit_index], lambda x: KPC.get_features(x))
    f_valid = dls.process_pc(valid_set["pc"][0:limit_index], lambda x: KPC.get_features(x))
    f_test = dls.process_pc(test_set["pc"][0:limit_index], lambda x: KPC.get_features(x))
    gt_train = dls.process_img(train_set["gt_bev"][0:limit_index], func=lambda x: utils.kitti_gt(x))
    gt_valid = dls.process_img(valid_set["gt_bev"][0:limit_index], func=lambda x: utils.kitti_gt(x))
    gt_test = dls.process_img(train_set["gt_bev"][0:limit_index], func=lambda x: utils.kitti_gt(x))
    return np.array(f_train), np.array(f_valid), np.array(f_test), np.array(gt_train), np.array(gt_valid), np.array(gt_test)

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
    # get dataset
    train_set, valid_set, test_set = dls.get_dataset(PATH, is_training=True)
    #create dataclass
    KPC = utils.KittiPointCloudClass(train_set=train_set, 
                                     valid_set=valid_set)
    
    #limit_index = -1 for all dataset while i > 0 for smaller #samples
    f_train, f_valid, f_test, gt_train, gt_valid, gt_test = dataload_static(limit_index = 30)

    print(f_test.shape, gt_test.shape)
    print('===============')
    exit
    # All training params to be added here
    training_config = {
        "loss_function" : "binary_crossentropy",
        "learning_rate" : 1e-4,
        "batch_size"    : 2,
        "epochs"        : 1,
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
    path = create_run_dir(_RUN_PATH, run_id)
    
    model = lodnn.get_model()
    # model = unet.get_model()
    model.summary()
    callbacks = get_basic_callbacks(path)
    
    # Add more callbacks
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
    # callbacks = callbacks + [early_stopping, reduce_lr]
    
    optimizer = eval(training_config["optimizer"])(lr=training_config["learning_rate"])
    model.compile(loss=training_config["loss_function"],
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    # TODO: add fit_generator
    if 0:
        m_history = model.fit(x=f_train, y=gt_train,
                            batch_size=training_config["batch_size"],
                            epochs=training_config["epochs"],
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(f_valid, gt_valid)) 
    else:
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

    _acc, _recall, _precision, _f1 = measure_perf(path, all_pred, gt_test, is_viz=False)
    print(_acc, _recall, _precision, _f1)
    print('------------------------------------------------')
    #plot single prediction
    single_pred = all_pred[1,:,:,:]
    #plot([[single_pred]])
    
    #argmax to obtain segmentation
    plot([[1-apply_argmax(single_pred)]])
    plt.show()

    details = {"name" : _NAME,
           "run_id" : run_id,
           "dataset": "KITTI",
           "training_config" : training_config,
           #"threshold" : THRESHOLD,
           #"accuracy" : _acc,
           #"recall" : _recall,
           #"precision" : _precision,
           #"F1" : _f1
           }
    with open('{}/details.json'.format(path), 'w') as f:
        json.dump(details, f)