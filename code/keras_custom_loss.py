from __future__ import print_function

import tensorflow as tf
from keras import backend as K
import numpy as np

import pdb

smooth = 1.0
# Sources : 
# dice_coef and dice_coef_loss have been borrowed from:
# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
# Focal loss keras : https://github.com/aldi-dimara/keras-focal-loss/blob/master/focal_loss.py
# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

def binary_focal_loss(gamma=2.0, alpha=0.75):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)
        
        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise
        
        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise
        
        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):

        # # print(y_true, y_pred)
        # y_true = K.print_tensor(y_true, message='y_true = ')
        # y_pred = K.print_tensor(y_pred, message='y_pred = ')
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true)*alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1-p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss


def generalize_IoU(y_true, y_pred):
    """ 
    Generalized IoU : https://giou.stanford.edu/#method
    paper https://giou.stanford.edu/GIoU.pdf
    GIoU = IoU - |convexhull(A, B)| \setminus AUB| / |convexhull(A, B)|
    to be completed : requires an efficient way to calculate the convexhull(A, B)
    The GIoU can be seen as half of the Haussdorf distance since we measure the area difference
    between convexhull(A, B)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - intersection
    #TODO : add convex envolope of y_true, y_pred for each class
    #Tensorflow implementations
    return (intersection) / (union)

def dice1_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice1_loss(y_true, y_pred):
    return 1-dice1_coef(y_true, y_pred)

def dice2_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice2_loss(y_true, y_pred):
    return 1-dice2_coef(y_true, y_pred)

def jaccard2_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


# def jaccard2_loss(y_true, y_pred):
#     return 1-jaccard2_coef(y_true, y_pred)

def jaccard1_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f ) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def jaccard1_loss(y_true, y_pred):
    return 1-jaccard1_coef(y_true, y_pred)

def diag_dist_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(0.5 - 0.5 * np.cos(np.pi * np.abs(y_true_f - y_pred_f)))

def cross_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # return K.mean(y_true_f + y_pred_f -2*y_true_f*y_pred_f)
    return K.mean((y_true_f -y_pred_f)**4)

def rmse(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # return K.mean(y_true_f + y_pred_f -2*y_true_f*y_pred_f)
    return K.sqrt(K.mean(K.square(y_true_f -y_pred_f)))

def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index   
        classSelectors = [K.equal(i, tf.cast(classSelectors, tf.int32)) for i in range(len(weightsList))]

        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss

    return lossFunc


def weightedLoss2(originalLossFunc, weightsList):

    def lossFunc(true, pred):
        print(len(true.shape), true.shape)
        loss = 0
        for i in range(0, len(weightsList)):
            loss += weightsList[i] * originalLossFunc(true[:,:,:,i], pred[:,:,:,i])

        return loss

    return lossFunc

def ignoreLabelLoss(originalLossFunc, label_To_ingore=None):
    """
    Custom function that ignores some labels
    :param originalLossFunc:
    :param label_To_ingore:
    :return:
    """
    if label_To_ingore is None:
        label_To_ingore = [0]

    def lossFunc(true, pred):
        print(len(true.shape), print(true.shape))
        # _, _, _, nc = true.shape
        loss = 0
        for i in range(5):
            if i not in label_To_ingore:
                loss += originalLossFunc(true[:,:,:,i], pred[:,:,:,i])

        return loss

    return lossFunc

def custom_categorical_accuracy(y_true, y_pred):
    return K.mean(K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx()))


# distance_jaccard2_loss and no_distance_jaccard2_loss are used to calculate the loss function based on distance maps
# Implemented by Yang XIAO in Sep 2018

def distance_jaccard2_loss(y_true, y_pred, t=0.0, s=20.0, m=1.0):
    """Calculate the jaccard2 loss considering the distance map.

    This new loss function penalizes the False Positive according to the corresponding distance maps D.

    Arguments
        y_true: Ground truth containing the signed distance maps for each class.
        y_pred: Predicted probability for each class.
        t: Spatial tolerance for penalizing the False Positive.
        s: Step value for the function f(D).
        m: Maximum value for the function f(D) is (m+1).

    Returns
        loss function considering the distance maps
    """

    # recover the binary masks from the distance maps
    y_true_b = y_true < 0.0
    y_true_b = K.cast(y_true_b, K.dtype(y_true))

    # construct f(D)
    D = K.maximum(y_true - t, 0.0)
    f_D = ((2.0 * m) / (1.0 + K.exp(-D / s))) - (m - 1)

    intersection = K.sum(K.flatten(y_true_b * y_pred))
    union = K.sum(K.flatten(y_pred * y_pred * f_D + y_true_b * y_true_b - y_true_b * y_pred))
    jaccard2 = (intersection + 1.0) / (union + 1.0)
    return 1 - jaccard2


class DistanceJaccard2:
    """Functor for distance_jaccard2_loss()."""

    def __init__(self, t, s, m):
        """Init functor parameters.
        # Arguments
            t: Tolerance
            s: Step
            m: Maximum
        """
        self.__t__ = t
        self.__s__ = s
        self.__m__ = m
        self.__name__ = "DistanceJaccard2"

    def __call__(self, y_true, y_pred):
        return distance_jaccard2_loss(y_true, y_pred, self.__t__, self.__s__, self.__m__)

def weighted_cross_entropy(beta):
  def convert_to_logits(y_pred):
      # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
      y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

      return tf.log(y_pred / (1 - y_pred))

  def loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss)

  return loss


def jaccard_loss_mean(y_true, y_pred, smooth=1):
    """
    Mean Jaccard Loss per classes.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    """
    y_true=K.abs(y_true)
    y_pred=K.abs(y_pred)
    intersection = y_true * y_pred
    union= y_true * y_true + y_pred*y_pred - intersection
    union =K.sum(union, axis=[3,2])
    intersection = K.sum(intersection, axis=[3,2])
    jac = (intersection + smooth) / (union + smooth)
    jac = K.mean(jac,axis=1)
    return jac


def jaccard2_loss(y_true, y_pred, smooth=1):
    return 1 - jaccard_loss_mean(y_true, y_pred, smooth)

