import random
import matplotlib.pyplot as plt
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


def model_prediction(model, X, Y,img_num):
    test_img = X[img_num]
    ground_truth=Y[img_num]
    test_img_norm=test_img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img, cmap='jet')
    plt.show()