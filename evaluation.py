
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


def rest_metrics(model,X_test,y_test):
    print('\n-------------On Test Set--------------------------\n')
    res = model.evaluate(X_test, y_test, batch_size= 1)
    print('________________________')
    print('Dice Coef:      |   {:.2f}   |'.format(res[1]*100))
    print('Precision:      |   {:.2f}   |'.format(res[2]*100))
    print('Recall:         |   {:.2f}   |'.format(res[3]*100))
    print('Specificity:    |   {:.2f}   |'.format(res[4]*100))
    print('F1_Score:       |   {:.2f}   |'.format(res[5]*100))
    print('NPV :           |   {:.2f}   |'.format(res[6]*100))
    print('Accuracy:       |   {:.2f}   |'.format(res[7]*100))
    print("Loss:           |   {:.2f}   |".format(res[0]*100))



def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def pred(X,y,model,num,dsize):

  #returns the y preds and y accutual

    y_acc = []
    y_pred = []
    for i in range(0,num):
        y_acc.append(y[i].reshape(dsize[0],dsize[1],1))
        y_pred.append(create_mask(model.predict(X[i:i+1])) )

    y_acc = np.array(y_acc)
    y_pred = np.array(y_pred)

    return (y_acc,y_pred)



def mean_iou_score(model,X_test,y_test,n_classes=5,dsize=(224,224)):

    y_acc , y_pred = pred(X_test,y_test,model , len(y_test),dsize)
    IOU_ref = MeanIoU(num_classes=n_classes)
    IOU_ref.update_state(y_acc , y_pred)


    values = np.array(IOU_ref.get_weights()).reshape(n_classes,n_classes)


    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0]+ values[2,0]+ values[3,0] + values[4,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1]+ values[2,1]+ values[3,1] + values[4,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4]+ values[0,2]+ values[1,2]+ values[3,2]  +values[4,2] )
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3]+ values[1,3]+ values[2,3] + values[4,3])
    class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2]  +values[4,3] + values[0,4]+ values[1,4]+ values[2,4] + values[3,4])

    print()
    print("IoU for class 0 is: ", class1_IoU)
    print("IoU for class 1 is: ", class2_IoU)
    print("IoU for class 2 is: ", class3_IoU)
    print("IoU for class 3 is: ", class4_IoU)
    print("IoU for class 4 is: ", class5_IoU)
    print("Mean Iou: ", (class1_IoU + class2_IoU + class3_IoU + class4_IoU + class5_IoU)/4)
    print()