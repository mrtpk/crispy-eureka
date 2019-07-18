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

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib qt # for process_stream
import cv2

def plot(imgs, labels=None, cmap='gray', figsize=(20,10), fontsize=30):
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