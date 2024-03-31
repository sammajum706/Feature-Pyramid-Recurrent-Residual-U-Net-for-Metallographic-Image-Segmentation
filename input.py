import cv2
import os
import numpy as np
from tqdm import tqdm
import keras
import tensorflow as tf
from tensorflow.keras.utils import normalize
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.metrics import MeanIoU


def for_img(X, dsize,image_directory):

    image_dataset = []

    for i in tqdm(X):

        image = cv2.imread(os.path.join(image_directory,i),0)
        image = cv2.resize(image, dsize= dsize, interpolation = cv2.INTER_NEAREST)
        image_dataset.append(image)

    image_dataset=np.array(image_dataset)

    image_dataset=np.expand_dims(image_dataset,axis=3)
    image_dataset=normalize(image_dataset,axis=1)
    return image_dataset


def for_mask(Y,dsize,cat=True,mask_directory="/.",n_classes=5 ):


    masks = []
    for i in tqdm(Y):
        img  = cv2.imread(os.path.join(mask_directory,i),0)
        res = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)
        masks.append(res)
    masks = np.array(masks)
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    train_masks_reshaped = masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    if(cat):
        train_masks_cat = to_categorical(train_masks_input, num_classes=n_classes)
        y_train_cat = train_masks_cat.reshape((train_masks_input.shape[0], train_masks_input.shape[1], train_masks_input.shape[2], n_classes))
        return y_train_cat

    else:
        return train_masks_input
    
def process(X_train,y_train,X_val,y_val,X_test,y_test,image_directory,mask_directory,dsize):

    train_X = for_img(X_train,dsize,image_directory)

    train_y = for_mask(y_train,dsize,mask_directory)

    val_X = for_img(X_val,dsize,image_directory)

    val_y = for_mask(y_val,dsize,mask_directory)

    test_X = for_img(X_test,dsize,image_directory)

    test_y = for_mask(y_test,dsize,mask_directory)

    test_y2 = for_mask(y_test, dsize,mask_directory,cat = False)

    return train_X ,train_y ,val_X ,val_y ,test_X ,test_y , test_y2

def kfold(X=[],y=[],fold=0,image_directory="/.",mask_directory="/.",dsize=(224,224), n_splits = 6):

    np.random.seed(42)  
    X = np.random.choice(X, size= len(X) , replace=False)
    np.random.seed(42) 
    y = np.random.choice(y, size= len(y)  , replace=False)

    X = np.array_split(X, n_splits)
    y = np.array_split(y, n_splits)

    l = []
    for i in range(0,n_splits):
        if(i != fold):
            l.append(i)



    X_train = np.concatenate([X[l[i]] for i in range(0,len(l))], axis=0)
    y_train = np.concatenate([y[l[i]] for i in range(0,len(l))], axis=0)

    np.random.seed(42)
    X_val = np.random.choice(X_train, size= int(len(X_train)*0.1) , replace=False)

    np.random.seed(42)
    y_val = np.random.choice(y_train, size= int(len(y_train)*0.1) , replace=False)


    X_test = X[fold]
    y_test = y[fold]


    return process(X_train,y_train,X_val,y_val,X_test,y_test,image_directory,mask_directory,dsize)