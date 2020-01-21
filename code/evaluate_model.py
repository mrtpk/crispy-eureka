import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from time import time
from data import *
import dl_models
from helpers.data_loaders import get_semantic_kitti_dataset, get_dataset, load_pc, process_calib, load_img, process_list
from helpers.data_loaders import load_filter_cloud, process_iter, process_img
from helpers.timer import timer
from helpers.viz import plot_confusion_matrix, ConfusionMatrix
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, jaccard_score, recall_score, precision_score, f1_score
import keras # this is required
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras import backend as k
from keras.optimizers import *
from keras_custom_loss import jaccard2_loss


def plot_features_map(x, y, geom=False, figsize=(8,8)):
    nr, nc, nz = x.shape

    if geom:
        num_fig = nz - 1
    else:
        num_fig = nz + 1

    _, ax = plt.subplots(num_fig, 1, figsize=figsize)

    for i in range(num_fig):
        if geom and i == num_fig - 2:
            ax[i].imshow(x[:, :, -3:])
        elif i == num_fig - 1:
            ax[i].imshow(y[: , :,0])
        else:
            ax[i].imshow(x[:,:,i])
        # ax[i].axes('off')

    plt.show()

class Experiment:
    def __init__(self, path, weights, model_name='unet', view='front', plot_result_flag=False, dataset='semantickitti', sequences=None):
        self.input = dict(
            model_name=model_name,
            weights=weights,
            view=view,
            plot_result_flag=plot_result_flag,
            dataset=dataset)

        self.model_name = model_name
        self.view = view
        self.weights = weights
        self.dataset = dataset
        weights_dir = os.path.abspath(os.path.dirname(weights))
        weights_par_dir = os.path.abspath(os.path.join(weights_dir, os.pardir))
        self.details = self.__load_details(os.path.join(weights_par_dir, 'details.json'))

        self.test_name = self.details['test_name'].lower()

        self.geometric, self.hog, self.sampled, self.sampled_ratio, \
        self.eigen, self.n_channels = self.get_info_from_test_name()

        self.feature_parameters = self.details.get('features', None)
        if self.feature_parameters is None:
            self.feature_parameters = dict(
                compute_classic=True,
                add_geometrical_features=self.geometric,
                subsample_ratio=self.sampled_ratio,
                compute_eigen=self.eigen,
                model_name=self.model_name
            )
        self.sampled_ratio = self.feature_parameters['subsample_ratio']
        self.model = self.__initialize_model()

        self.feature_parameters['z_max'] = float(self.details['z_max'])
        self.feature_parameters['z_min'] = float(self.details['z_min'])
        if self.view == 'bev':
            self.feature_parameters['COUNT_MAX'] = int(float(self.details['COUNT_MAX']))

        self.path = path
        if sequences is None and self.view != 'kitti':
            sequences = ['08']

        KPC = KITTIPointCloud(feature_parameters=self.feature_parameters,
                              path=self.path,
                              sequences=sequences,
                              is_training=False,
                              view=self.view,
                              dataset=self.dataset)
        self.KPC = KPC


    def __initialize_model(self):
        shape = (None, None, self.n_channels)
        print(shape)
        if self.model_name == 'lodnn':
            model = dl_models.get_lodnn_model(shape=shape)
        elif self.model_name == 'unet':
            if self.dataset == 'kitti':
                output_channels = 2
            else:
                output_channels = 1

            model = dl_models.get_unet_model(shape=shape, subsample_ratio=self.sampled_ratio, output_channels=1)
        elif self.model_name == 'unet6':
            model = dl_models.u_net6(shape=shape,
                                     filters=512,
                                     int_space=32,
                                     output_channels=2)
        else:
            raise ValueError("model_name nor lodnn neither unet")

        model.summary()

        model.load_weights(self.weights)
        training_config = self.details['training_config']
        # retrieve optimizer
        optimizer = eval(training_config["optimizer"])(lr=training_config["learning_rate"])

        # compile model
        model.compile(loss=jaccard2_loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def __load_details(self, filename):
        with open(filename) as f:
            d = json.load(f)

        return d

    def _fetch_data(self, test_set):
        """
        This reads the whole dataset into memory
        """
        print('Reading cloud')
        f_test = load_pc(test_set["pc"][0:])

        if self.hog:
            print('Reading calibration files')
            cal_test = process_calib(test_set["calib"][0:])

            print('Reading camera images')
            cam_img_test = load_img(test_set["imgs"][0:])

        if self.sampled:
            print('Read and Subsample cloud')
            t = time()
            f_test = process_list(f_test, subsample_pc, sub_ratio=self.sampled_ratio)
            print('Evaluated in : ' + repr(time() - t))

        print('Extracting features')
        t = time()

        if self.hog:
              f_cam_calib_test = [(f_t, img, calib) for f_t, img, calib in zip(f_test, cam_img_test, cal_test)]
        else:
            f_cam_calib_test = zip(f_test, len(f_test) * [None], len(f_test) * [None])

        f_test = process_list(f_cam_calib_test, self.KPC.get_features)

        print('Evaluated in : ' + repr(time() - t))

        gt_key = 'gt_bev' if self.view == 'bev' else 'gt_front'
        gt_test = process_img(test_set[gt_key][0:], func=lambda x: data.kitti_gt(x))

        return np.array(f_test), np.array(gt_test)


    def load_dataset(self, test_set=None):
        if test_set is None:
            test_set = self.KPC.test_set
        f_test, gt_test = self._fetch_data(test_set)

        return f_test, gt_test

    def get_info_from_test_name(self):
        '''
        Functions that reads the details.json file and return the number of features used during training, and if
        the point cloud has been sampled
        '''
        n_channels = 6 if self.view == 'bev' else 3
        n_channels = 0 if 'classic' not in self.test_name else n_channels
        geometric = False
        if 'geometric' in self.test_name:
            geometric = True
            n_channels += 3
        if 'height' in self.test_name:
            height = True
            n_channels += 4 if self.view == 'bev' else 1

        hog = False
        if 'hog' in self.test_name:
            hog = True
            n_channels += 6

        sampled = False
        sampled_ratio = 1
        if 'sampled' in self.test_name:
            sampled = True
            if '32' in self.test_name:
                sampled_ratio = 2
            if '16' in self.test_name:
                sampled_ratio = 4
        eigen = 0
        if 'eigen' in self.test_name:
            eigen = 100  # TODO: take it from config file
            n_channels += 6

        return geometric, hog, sampled, sampled_ratio, eigen, n_channels


    def run_pred(self, f_test):
        # image shape
        n_row, n_col, n_channels = f_test.shape[1:]

        if 'unet' in self.model_name:
            multiple_of = 16 if self.model_name == 'unet' else 32
            nrpad = multiple_of - (n_row % multiple_of) if (n_row % multiple_of) != 0 else 0
            ncpad = multiple_of - (n_col % multiple_of) if (n_col % multiple_of) != 0 else 0

            f_test = np.pad(f_test, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')


        pred, times = predict_test_set(self.model, f_test)

        # pred = pred[:, :(self.sampled_ratio * n_row), :n_col, :]  # setting same dimension as before

        return pred, times

    def save_predictions(self, gt, pred, ids):
        for i in ids:
            plot_prediction(gt, pred, i, save_fig=True, filename=self.test_name)

    @staticmethod
    def precision_recall_curve(gt, pred, return_f1=False):
        gt_flatten = gt[:, :, :, 0].flatten()
        pred_flatten = pred[:, :, :, 0].flatten()
        prec, rec, thresholds = precision_recall_curve(gt_flatten, pred_flatten)
        f1_scores = (2 * (prec*rec)/ (prec + rec))
        best_f1 = f1_scores.max()
        imax = f1_scores.argmax()
        print("Scores test: Best F1: {}, Rec: {}, Prec: {}, threshold: {}".format(best_f1, rec[imax],
                                                                                     prec[imax], thresholds[imax]))
        if return_f1:
            return prec, rec, best_f1

        return prec, rec

    @staticmethod
    def average_precision_score(gt, pred):
        gt_flatten = gt[:, :, :, 0].flatten()
        pred_flatten = pred[:, :, :, 0].flatten()
        score = average_precision_score(gt_flatten, pred_flatten)

        return score


def evaluate_single_experiment(path,
                               weights,
                               min_id=0,
                               max_id=-1,
                               step=1,
                               model_name='lodnn',
                               view='bev',
                               dataset='kitti',
                               sequences=None,
                               bp_func=None):
    exp = Experiment(path, weights=weights, model_name=model_name, view=view, dataset=dataset, sequences=sequences)
    # load images
    f_test, gt_test = exp.KPC.load_dataset(set_type='test', min_id=min_id, max_id=max_id, step=step)
    print(f_test.shape, gt_test.shape)
    print("Experiment {}: Features maps shape: {}, GT shape: {}".format(exp.test_name, f_test.shape, gt_test.shape))
    print("Running Prediction")
    pred = exp.model.predict(f_test)
    if bp_func is not None:
        print("Loading 3D GT")
        # load clouds and gt
        kwargs = {} if view == 'bev' else {'add_layers': True}
        clouds = process_list(exp.KPC.test_set['pc'][min_id:max_id:step], load_filter_cloud, **kwargs)
        y_true = np.concatenate([cloud.points['road'].values for cloud in clouds])
        print("Backprojecting prediction to 3D")
        # back project pred
        proj = exp.KPC.proj if view == 'bev' else exp.KPC.aux_proj
        y_pred = process_iter(zip(clouds, pred[:, :, :, 0]), bp_func, proj=proj)
        y_pred = np.concatenate(y_pred)
        print("Computing Scores")
        # evaluate 3D scores
        avg_prec = average_precision_score(y_true, y_pred)
        prec, recall, _ = precision_recall_curve(y_true, y_pred)
    else:
        avg_prec = exp.average_precision_score(gt_test, pred)
        prec, recall = exp.precision_recall_curve(gt_test, pred)

    print("Average Precision: ", avg_prec)
    out = {}
    out['avg_prec'] = avg_prec
    out['precision'] = prec
    out['recall'] = recall
    out['f_test'] = f_test
    out['prediction'] = pred
    out['clouds'] = clouds
    return out


def evaluate_experiments(path,
                         weights,
                         min_id=0,
                         max_id=-1,
                         step=1,
                         model_name='lodnn',
                         view='bev',
                         dataset='kitti',
                         sequences=None,
                         bp_func=None,
                         kittionsemkitti=False):
    prec_scores = {}
    recall_scores = {}
    avg_prec = {}

    for k in weights:
        print(k)
        if kittionsemkitti:
            exp = Experiment(path, weights=weights[k], model_name=model_name, view=view,
                             dataset='kitti', sequences=None)
            exp.KPC.test_set = get_semantic_kitti_dataset(path, is_training=False, sequences=sequences, shuffle=True)
        else:
            exp = Experiment(path, weights=weights[k], model_name=model_name, view=view,
                             dataset=dataset, sequences=sequences)
        # load images
        f_test, gt_test = exp.KPC.load_dataset(set_type='test', min_id=min_id, max_id=max_id, step=step)
        print(f_test.shape, gt_test.shape)
        print("Experiment {}. Features maps shape: {}, GT shape: {}".format(k, f_test.shape, gt_test.shape))
        print("Running Prediction")
        pred = exp.model.predict(f_test)
        if bp_func is not None:
            print("Loading 3D GT")
            # load clouds and gt
            kwargs = {} if view=='bev' else {'add_layers': True}
            clouds = process_list(exp.KPC.test_set['pc'][min_id:max_id:step], load_filter_cloud, **kwargs)
            y_true = np.concatenate([cloud.points['road'].values for cloud in clouds])
            print("Backprojecting prediction to 3D")
            # back project pred
            proj = exp.KPC.proj if view=='bev' else exp.KPC.aux_proj
            print(proj.projector)
            y_pred = process_iter(zip(clouds, pred[:, :, :, 0]), bp_func, proj=proj)
            y_pred = np.concatenate(y_pred)
            print("Computing Scores")
            # evaluate 3D scores
            avg_prec[k] = average_precision_score(y_true, y_pred)
            prec, recall, _ = precision_recall_curve(y_true, y_pred)
        else:
            avg_prec[k] = exp.average_precision_score(gt_test, pred)
            prec, recall = exp.precision_recall_curve(gt_test, pred)

        prec_scores[k] = prec
        recall_scores[k] = recall

    # Create a list of tuples sorted by index 1 i.e. value field
    listofTuples = sorted(avg_prec.items(), key=lambda x: x[1])

    print("Ranking Exp by Accuracy")
    # Iterate over the sorted sequence
    for elem in listofTuples:
        print(elem[0], " ::", elem[1])

    return avg_prec, prec_scores, recall_scores


def plot_prediction(gt, pred, n=-1, save_fig=False, filename=''):
    if n < 0:
        n = np.random.randint(len(gt))
    print(n)

    plt.style.use('classic')
    f, ax = plt.subplots(3, 1, figsize=(20, 7))
    ax[0].imshow(gt[n, :, :, 0])
    ax[0].set_title('Ground Truth {:04}'.format(n))
    ax[0].axis('off')
    ax[1].imshow(pred[n, :, :, 0])
    ax[1].set_title('Road class heatmap {:04}'.format(n))
    ax[1].axis('off')
    threshold = 0.5
    ax[2].imshow((pred[n] > threshold)[:, :, 0])
    ax[2].set_title('Prediction {:04} with threshold={}'.format(n, threshold))
    ax[2].axis('off')

    if save_fig:
        plt.tight_layout()
        plt.savefig('pred/pred_{}{:04}.png'.format(filename, n), dpi=90)

    plt.show()

def store_npz(filename, arrays):
    np.savez(filename, **arrays)


def remove_bg(gt_train):
    gt_train = gt_train[:,:,:,0]
    gt_train = gt_train[:,:,:,np.newaxis]
    return gt_train

def load_dataset(add_geometrical_features=True,
                 subsample_flag=True,
                 compute_HOG=False,
                 compute_eigen=0,
                 view='bev',
                 dataset='kitti',
                 subsample_ratio=1):

    PATH = '../'  # path of the repo.
    _NAME = 'experiment0'  # name of experiment
    sequences = None
    # create dataclass
    KPC = utils.KittiPointCloudClass(dataset_path=PATH,
                                     add_geometrical_features=add_geometrical_features,
                                     subsample=subsample_flag,
                                     compute_HOG=compute_HOG,
                                     eigen_neighbors=compute_eigen,
                                     view=view,
                                     subsample_ratio=subsample_ratio,
                                     dataset=dataset,
                                     sequences=sequences)


    f_train, f_valid, f_test, gt_train, gt_valid, gt_test = KPC.get_dataset()
    gt_test = remove_bg(gt_test)
    return f_test, gt_test


def get_info_from_test_name(filename):
    '''
    Functions that reads the details.json file and return the number of features used during training, and if
    the point cloud has been sampled
    '''
    with open(filename) as f:
        d = json.load(f)

    test_name = d['test_name'].lower()

    geometric = False
    if 'geometric' in test_name:
        geometric = True

    hog = False
    if 'hog' in test_name:
        hog = True

    sampled = False
    sampled_ratio = 1
    if 'sampled' in test_name:
        sampled = True
        if '32' in test_name:
            sampled_ratio = 2
        if '16' in test_name:
            sampled_ratio = 4
    eigen = 0
    if 'eigen' in test_name:
        eigen = 100 # TODO: take it from config file

    print(d)

    return geometric, hog, sampled, sampled_ratio, eigen, d['training_config']


def initialize_model(model_name, weights, training_config, shape, subsample_ratio=1):
    # initialize model
    if model_name == 'lodnn':
        model = dl_models.get_lodnn_model(shape=shape)
    elif model_name == 'unet':
        model = dl_models.get_unet_model(input_size=shape, subsample_ratio=subsample_ratio)
    elif model_name == 'unet6':
        model = dl_models.u_net6(shape=shape,
                                 filters=512,
                                 int_space=32,
                                 output_channels=2)
    else:
        raise ValueError("model_name nor lodnn neither unet")

    model.summary()

    model.load_weights(weights)

    # retrieve optimizer
    optimizer = eval(training_config["optimizer"])(lr=training_config["learning_rate"])

    # compile model
    model.compile(loss=jaccard2_loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


@timer
def predict_sample(model, sample):
    prediction = model.predict(sample)
    return prediction[0, :, :, :]


def predict_test_set(model, f_test):
    all_pred = []
    all_times =[]
    for f in f_test:
        prediction, time = predict_sample(model, np.expand_dims(f, axis=0))
        all_pred.append(prediction)
        all_times.append(time)

    all_pred = np.array(all_pred)
    all_times = np.array(all_times)
    print("Mean prediction time: {}. Std prediction time: {}".format(all_times.mean(), all_times.std()))

    return all_pred, all_times


def compute_scores(pred, gt, threshold=0.5):

    gt_all = gt[:,:,:,0].flatten()
    # argmax_pred = 1 - np.argmax(pred, axis=3)
    # argmax_pred = argmax_pred.flatten()
    argmax_pred = (pred > threshold).flatten()
    acc = accuracy_score(gt_all, argmax_pred)
    recall = recall_score(gt_all, argmax_pred)
    precision = precision_score(gt_all, argmax_pred)
    f1 = f1_score(gt_all, argmax_pred)
    jaccard = jaccard_score(gt_all, argmax_pred)

    print("Scores: "
          "F1 -> {},"
          " Recall -> {},"
          " Precision -> {},"
          " Accuracy -> {},"
          " Jaccard -> {}".format(f1, recall, precision, acc, jaccard))


    print('------------------------------------------------')

    scores = {'f1': f1,
              'recall': recall,
              'precision': precision,
              'acc': acc,
              'jaccard': jaccard}

    return scores


def evaluate_model(model_name, weights, view, plot_result_flag, dataset):
    # getting dir
    weights_dir = os.path.abspath(os.path.dirname(weights))

    # getting parend directory
    weights_par_dir = os.path.abspath(os.path.join(weights_dir, os.pardir))

    # retrieve basic info on trained model
    geometric, hog, sampled, subsample_ratio, eigen, training_config = get_info_from_test_name(os.path.join(weights_par_dir, 'details.json'))
    print(training_config)

    f_test, gt_test = load_dataset(add_geometrical_features=geometric,
                                   subsample_flag=sampled,
                                   compute_HOG=hog,
                                   compute_eigen=eigen,
                                   view=view,
                                   subsample_ratio=subsample_ratio,
                                   dataset=dataset)

    print("Test set shape {}".format(f_test.shape))
    print("GT set shape {}".format(gt_test.shape))

    # image shape
    n_row, n_col, n_channels = f_test.shape[1:]

    if 'unet' in model_name:
        multiple_of = 16 if model_name =='unet' else 32
        nrpad = multiple_of - (n_row % multiple_of) if (n_row % multiple_of) != 0 else 0
        ncpad = multiple_of - (n_col % multiple_of) if (n_col % multiple_of) != 0 else 0

        f_test = np.pad(f_test, ((0, 0), (0, nrpad), (0, ncpad), (0, 0)), 'constant')

    # initializing model
    model = initialize_model(model_name, weights, training_config, f_test.shape[1:], subsample_ratio=subsample_ratio)

    pred, times = predict_test_set(model, f_test)

    pred = pred[:, :(subsample_ratio*n_row), :n_col, :]  # setting same dimension as before

    print(pred.shape, gt_test.shape)
    if 'unet' in model_name:
        scores = compute_scores(pred[:, :(subsample_ratio*n_row), :n_col, :], gt_test)
    else:
        scores = compute_scores(pred, gt_test)

    if plot_result_flag:
        for n in range(len(pred)):
            f, ax = plt.subplots(1, 4)
            ax[0].imshow(gt_test[n, :, :, 0])
            ax[0].set_title('gt_{}'.format(n))
            ax[1].imshow(pred[n, :, :, 0])
            ax[1].set_title('pred_0_{}'.format(n))
            ax[2].imshow(pred[n, :, :, 1])
            ax[2].set_title('pred_1_{}'.format(n))
            ax[3].imshow(1 - np.argmax(pred[n, :, :, :], axis=2))
            ax[3].set_title('pred_argmax_{}'.format(n))
            plt.show()
    print(gt_test.shape)
    output = np.zeros_like(gt_test[:, :, :, 0])
    print(pred.shape)
    for i in range(pred.shape[0]):
        output[i] = 1 - np.argmax(pred[i], axis=2)

    cm = confusion_matrix(gt_test[:, :, :, 0].flatten(), output.flatten())
    plot_confusion_matrix(cm, ['not_road', 'road'], normalize=True)
    conf_mat = ConfusionMatrix(number_of_labels=2)
    conf_mat.confusion_matrix = cm
    print("Overall Accuracy: {}".format(conf_mat.get_overall_accuracy()))
    print("IoU: {}".format(conf_mat.get_average_intersection_union()))
    out = dict(f_test=f_test,
             gt_test=gt_test,
             model=model,
             scores=scores,
             pred=pred,
             times=times,
             conf_mat=conf_mat)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Segmentation")
    parser.add_argument('--model_weights', type=str, help='path to model weights')
    parser.add_argument('--model', default='lodnn', type=str, help='architecture to use for evaluation')
    parser.add_argument('--cuda_device', default='0', type=str, help='GPU to use')
    parser.add_argument('--plot_result_flag', default=False, help='flag to show images of road segmentation')
    parser.add_argument('--view', type=str, default='bev', help='projection to use')
    parser.add_argument('--dataset', default='kitti', type=str, help='Dataset to use for training')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Tensorflow wizardry
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    # config.gpu_options.per_process_gpu_memory_fraction = 0.85

    # create a session with the above option specified
    k.tensorflow_backend.set_session(tf.Session(config=config))
    run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    evaluate_model(args.model, args.model_weights, args.view, args.plot_result_flag, args.dataset)
