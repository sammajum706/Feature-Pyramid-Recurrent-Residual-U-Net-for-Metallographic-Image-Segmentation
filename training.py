import segmentation_models_3D as sm
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime
import cv2
from PIL import Image
import keras
from keras import backend, optimizers
from tensorflow.keras.utils import normalize
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from keras.layers import *
from keras.models import *
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.metrics import MeanIoU
from keras.callbacks import ModelCheckpoint


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def negative_predictive_value(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())

def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def dice_coef(y_true, y_pred, smooth = 100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def precision(y_true, y_pred):
   
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def accuracy(y_true, y_pred):
    
    return K.mean(K.equal(y_true, K.round(y_pred)))


# Hybrid Loss Function
def hybrid_loss():
    focal_loss=sm.losses.CategoricalFocalLoss()
    dice_loss=sm.losses.DiceLoss(class_weights=np.array([0.3,0.3,0.5,0.8,0.8]))
    total_loss=dice_loss+(2*focal_loss)
    return total_loss


# Model Training
def train_model(model, model_directory,learning_rate,X_train,y_train,X_val,y_val,batch_size,epochs):

    checkpoint = ModelCheckpoint(filepath=model_directory,
                             monitor='val_dice_coef',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
    
    callbacks = [checkpoint]
    loss_function=hybrid_loss()
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss=loss_function,
    metrics=[dice_coef, precision, recall,specificity,f1_score,negative_predictive_value, accuracy , sm.metrics.IOUScore(threshold=0.5)])
    
    Unet_history = model.fit(X_train,y_train,
                    batch_size,
                    epochs,
                    validation_data=(X_val,y_val),
                          callbacks=callbacks)
    
    return Unet_history



