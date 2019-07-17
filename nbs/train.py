#%matplotlib inline
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import datetime
import os
import numpy as np
import tensorflow as tf
import keras
import sklearn.model_selection as sk
import copy
from PIL import Image as IM
from keras.callbacks import TensorBoard
import cv2
import json
import pathlib
from glob import glob

## import helpers
from helpers import data_loaders as dls
from helpers import pointcloud as pc
from helpers.viz import plot, plot_history
from helpers.logger import Logger
import utils
## import networks
import lodnn

def get_unique_id():
    return datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')

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

def apply_threshold(res, threshold=0.9):
    '''
    Apply thresholding on probability values.
    '''
    _tmp = res.copy()
    _tmp[res >= threshold] = 1
    _tmp[res < threshold] = 0
    return _tmp.astype(np.uint8)

def apply_argmax(res):
    return np.argmax(res, axis=2)

def get_output(path, model, dataset, threshold, is_viz=False):
    result_path = "{}/output/*".format(path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    # NOTE: Now computing for 8 samples (can this be done on samples from the train and validation set?)
    f = dls.process_pc(dataset["pc"][0:8], lambda x: _get_features(x))
    gt = dls.process_img(dataset["gt_bev"][0:8], func=lambda x: kitti_gt(x))
    F1, P, R, ACC = [], [], [], []
    FN, FP, TP, TN = [], [], [], []
    counter = 0
    diplay_thresh = 3
    display = []
    labels = []
    
    for i, datum in enumerate(zip(f, gt)):
        _f, _gt = datum
        res = get_prediction(model, _f)
        th_res = apply_threshold(res, threshold=threshold)
               
        # get metrics
        p_road = th_res[:, :, 0]
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

if __name__ == "__main__":
    PATH = '../' # path of the repo.
    _NAME = 'experiment0' # name of experiment
    #!ls $PATH
    # It is better to create a folder with runid in the experiment folder
    _EXP, _LOG, _TMP, _RUN_PATH = dls.create_dir_struct(PATH, _NAME)
    logger = Logger('EXP0', _LOG + 'experiment0.log')
    logger.debug('Logger EXP0 int')
    #!ls $_EXP
    # get dataset
    train_set, valid_set, test_set = dls.get_dataset(PATH, is_training=True)
    #create dataclass
    KPC = utils.KittiPointCloudClass(train_set=train_set, valid_set=valid_set)
    
    limit_index = -1
    # NOTE: change limit_index to -1 to train on the whole dataset
    f_train = dls.process_pc(train_set["pc"][0:limit_index], lambda x: KPC.get_features(x))
    f_valid = dls.process_pc(valid_set["pc"][0:limit_index], lambda x: KPC.get_features(x))
    f_test = dls.process_pc(test_set["pc"][0:limit_index], lambda x: KPC.get_features(x))
    gt_train = dls.process_img(train_set["gt_bev"][0:limit_index], func=lambda x: utils.kitti_gt(x))
    gt_valid = dls.process_img(valid_set["gt_bev"][0:limit_index], func=lambda x: utils.kitti_gt(x))
    gt_test = dls.process_img(train_set["gt_bev"][0:limit_index], func=lambda x: utils.kitti_gt(x))
    
    run_id = get_unique_id()
    path = create_run_dir(_RUN_PATH, run_id)
    
    model = lodnn.get_model()
    model.summary()
    callbacks = get_basic_callbacks(path)
    
    # Add more callbacks
    # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.000001)
    # callbacks = callbacks + [early_stopping, reduce_lr]
    
    # All training params to be added here
    training_config = {
        "loss_function" : "binary_crossentropy",
        "learning_rate" : 1e-4,
        "batch_size"    : 1,
        "epochs"        : 10,
        "optimizer"     : "keras.optimizers.Adam"
    }
    
    optimizer = eval(training_config["optimizer"])(lr=training_config["learning_rate"])
    model.compile(loss=training_config["loss_function"],
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    # TODO: add fit_generator
    m_history = model.fit(x=np.array(f_train),
                          y=np.array(gt_train),
                          batch_size=training_config["batch_size"],
                          epochs=training_config["epochs"],
                          verbose=1,
                          callbacks=callbacks,
                          validation_data=(np.array(f_valid), np.array(gt_valid))) 
    
    model.save("{}/model/final_model.h5".format(path))
    plot_history(m_history)
    
    f_test = np.array(f_test).squeeze()
    
    #test set prediction
    res = model.predict(f_test, verbose=0).squeeze()
    
    single_pred = res[:,:,1,:]
    #plot([[single_pred]])
    
    #argmax to obtain segmentation
    plot([[1-apply_argmax(single_pred)]])
    
    details = {"name" : _NAME,
           "run_id" : run_id,
           "dataset": "KITTI",
           "training_config" : training_config,
           "threshold" : THRESHOLD,
           "accuracy" : _acc,
           "recall" : _recall,
           "precision" : _precision,
           "F1" : _f1}
    with open('{}/details.json'.format(path), 'w') as f:
        json.dump(details, f)
    
