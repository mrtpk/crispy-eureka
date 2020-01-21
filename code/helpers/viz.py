# Contains helper functions for visualization.
# Helpers for images
# plot : for plotting multiple images
# process_stream : to process a video stream
# chunks : converts a list into sub lists
# _axis_mapping : to create a maping between sub plot and a viz function
# Helpers for training
# plot_history : plots loss and accuracy


##########
# Helpers for images
##########

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyvista as pv
from pyntcloud import PyntCloud

# %matplotlib qt # for process_stream
import cv2


def plot(imgs, labels=None, cmap='gray', figsize=(20,10), fontsize=30, show_axis=True):
    '''
    Displays images and labels in a plot.
    Usage:
    The input imgs should be in the format "[[img]]""
    '''

    nrows, ncols = len(imgs), len(imgs[0])
    if nrows > 100 or ncols > 50:
        print("Uh-oh, cannot plot these many images")
        return None
    f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1 or ncols == 1:
        axes = [axes]
    if nrows == 1 and ncols == 1:
        axes = [axes]
    for i in range(0, nrows):
        for j in range(0, ncols):
            try:
                if show_axis is False:
                    axes[i][j].axis('off')
                axes[i][j].imshow(imgs[i][j], cmap=cmap)
                if labels is not None:
                    axes[i][j].set_title(labels[i][j], fontsize=fontsize)
            except IndexError as err:
                continue
    return axes

def process_stream(path, process_frame, interval=1):
    '''
    Process and display a video stream at @param path using @param process_frame function.
    The @param interval specifies the interval at which the frame to be processed.

    Usage:
    count = 0
    def save_frame(img):
        global count
        count += 1
        cv2.imwrite('data/eq/#.jpg'.replace('#', str(count)), img)
        return img
    process_stream(path='sample.mp4', process_frame=save_frame, interval=10)
    '''
    cap = cv2.VideoCapture(path)
    counter = 1
    while(True):
        ret,frame = cap.read()
        if ret is True and counter % interval == 0:
            frame = process_frame(frame)
            cv2.imshow('frame',frame)
        counter += 1
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def chunks(data, n):
    return [data[x:x+n] for x in range(0, len(data), n)]

# Testing below code

def plot1D(data, labels=None, figsize=(20,10), fontsize=30):
    return _axis_mapping(func=_plot_1D, data=data, labels=labels, figsize=figsize, fontsize=fontsize)
    
def _plot_1D(axis, data):
    axis.plot(data)

def _axis_mapping(func, data, labels=None, figsize=(20,10), fontsize=30):
    '''
    To be merged with plot
    
    def act(axis, data):
        print(data.shape)
        axis.imshow(data, cmap='gray')

    def graphs(axis, data):
        axis.plot(data)

    '''
    nrows, ncols = len(data), len(data[0])
    if nrows > 100 or ncols > 50:
        print("Uh-oh, cannot plot these many plots")
        return None
    f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1 or ncols == 1:
        axes = [axes]
    if nrows == 1 and ncols == 1:
        axes = [axes]
    for i in range(0, nrows):
        for j in range(0, ncols):
            try:
                k = data[i][j]
                func(axes[i][j], k)
                if labels is not None:
                    axes[i][j].set_title(labels[i][j], fontsize=fontsize)
            except IndexError as err:
                continue
    return axes

########
# Training related helpers
########

def plot_history(history):
    '''
    Plots accuracy and loss from history object
    from https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
    '''
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()


class ConfusionMatrix:
    """Streaming interface to allow for any source of predictions. Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""

    def __init__(self, number_of_labels=2):
        self.number_of_labels = number_of_labels
        self.confusion_matrix = np.zeros(shape=(self.number_of_labels, self.number_of_labels))

    def count_predicted(self, ground_truth, predicted, number_of_added_elements=1):
        self.confusion_matrix[ground_truth][predicted] += number_of_added_elements

    """labels are integers from 0 to number_of_labels-1"""

    def get_count(self, ground_truth, predicted):
        return self.confusion_matrix[ground_truth][predicted]

    """returns list of lists of integers; use it as result[ground_truth][predicted]
       to know how many samples of class ground_truth were reported as class predicted"""

    def get_confusion_matrix(self):
        return self.confusion_matrix

    """returns list of 64-bit floats"""

    def get_intersection_union_per_class(self):
        matrix_diagonal = [self.confusion_matrix[i][i] for i in range(self.number_of_labels)]
        errors_summed_by_row = [0] * self.number_of_labels
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_row[row] += self.confusion_matrix[row][column]
        errors_summed_by_column = [0] * self.number_of_labels
        for column in range(self.number_of_labels):
            for row in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_column[column] += self.confusion_matrix[row][column]

        divisor = [0] * self.number_of_labels
        for i in range(self.number_of_labels):
            divisor[i] = matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
            if matrix_diagonal[i] == 0:
                divisor[i] = 1

        return [float(matrix_diagonal[i]) / divisor[i] for i in range(self.number_of_labels)]

    """returns 64-bit float"""

    def get_overall_accuracy(self):
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                all_values += self.confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += self.confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        return float(matrix_diagonal) / all_values

    def get_average_intersection_union(self):
        values = self.get_intersection_union_per_class()
        return sum(values) / len(values)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(8,8),
                          savefig=""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import itertools
    font = {
        'family': 'arial',
        'size': 14}

    matplotlib.rc('font', **font)

    if normalize:
        x = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), x, where=x!=0)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = 0.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    if len(savefig) > 0:
        plt.savefig(savefig, dpi=90)


def plot_bev_maps(X, nr, nc, idx=-1, title=None):
    if title is None:
        title = [
            'count',
            'mean_refl',
            'max_z',
            'min_z',
            'mean_z',
            'std_z','nx',
            'ny',
            'nz',
            'curvature',
            'eigenentropy',
            'linearity',
            'omnivariance',
            'planarity',
            'sphericity'
        ]

    if idx == -1:
        idx = np.random.randint(len(X))

    fig, axs = plt.subplots(nr, nc)

    for n, ax in enumerate(axs.ravel()):
        im = ax.imshow(X[idx, :, :, n])
        ax.set_title(title[n])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    plt.show()

def plot_Xy(X, y, alpha=0.5):

    plt.figure()
    plt.imshow(X[:, :,  6:], alpha=alpha)
    plt.imshow(y[:, :, 0], alpha=alpha)
    plt.show()


def color_cloud_by_confront(cloud, y_true, y_pred):
    """
    Function that color the point cloud according to the ground truth and prediction
    True positive are colored in green
    False positive are colored in red
    False negative are colored in blue

    Parameters
    ----------
    cloud: PyntCloud
        pyntcloud point cloud

    y_true: ndarray
        boolean ground truth array

    y_pred: ndarray
        boolean prediction array
    """
    cloud.points['y_true'] = y_true
    cloud.points['y_pred'] = y_pred
    # true positive
    tp = np.logical_and(cloud.points['y_true'], cloud.points['y_pred'])
    # false positive
    fp = np.logical_and(1 - cloud.points['road'], cloud.points['y_pred'])
    # false negative
    fn = np.logical_and(cloud.points['road'], 1 - cloud.points['y_pred'])
    cloud.points['red'] = 255*fp
    cloud.points['green'] = 255 * tp
    cloud.points['blue'] = 255 * fn

    return cloud


def plot_point_cloud(cloud,title):
    """
    Function that plot a point cloud

    Parameters
    ----------
    cloud: Pyntcloud
    """
    plotter = pv.BackgroundPlotter(title=title)
    mesh = pv.PolyData(cloud.xyz)
    columns = list(cloud.points)
    if 'red' in columns and 'green' in columns and 'blue' in columns:
        mesh['colors'] = np.array([cloud.points['red'], cloud.points['green'], cloud.points['blue']]).T

    pc = plotter.add_mesh(mesh, rgb=True, point_size=2)



def plot_all_recall_curve(prec_scores, recall_scores, figsize=(12,12), xlim=None, ylim=None, title='', savefig=""):
    if title == '':
        title = 'Precision-Recall curve'

    if xlim is None:
        xlim = [0.75, 1.0]

    if ylim is None:
        ylim = [0.75, 1.0]
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument

    plt.figure(figsize=figsize)
    plt.style.use('ggplot')
    legend_list = []
    n_16 = 0
    n_32 = 0
    n_64 = 0
    color_list = {'classical': 'b',
                  'classical_geometric': 'g',
                  'classical_eigen': 'r',
                  'classical_geometric_eigen': 'c',
                  'height_geometric': 'm'}

    for k in prec_scores:
        print(k)
        # if 'eigen' in k or 'height' in k:
        #     continue
        color_key = k.replace('_subsampled_32', '').replace('_subsampled_16', '')

        if '16' in k:
            legend_list.append(k)
            plt.step(prec_scores[k], recall_scores[k], linewidth=2, linestyle='--', color=color_list[color_key],
                     alpha=0.8, where='post')
            n_16 += 1

        if '32' in k:
            legend_list.append(k)
            plt.step(prec_scores[k], recall_scores[k], linewidth=2, linestyle='-.', color=color_list[color_key],
                     alpha=0.8, where='post')
            n_32 += 1

        if '16' not in k and '32' not in k:
            legend_list.append(k)
            plt.step(prec_scores[k], recall_scores[k], linewidth=2, linestyle='-', color=color_list[color_key],
                     alpha=0.8, where='post')
            n_64 += 1

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title(title)
    plt.legend(legend_list, loc=3)
    if len(savefig):
        plt.savefig(savefig, dpi=90)