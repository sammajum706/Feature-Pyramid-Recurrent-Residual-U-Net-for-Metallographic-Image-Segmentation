import os
import tensorflow as tf
from datetime import datetime
import keras
from keras import backend, optimizers
from tensorflow.keras.utils import normalize
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
from keras.layers import *
from keras.models import *


# spatial squeeze by mean and channel excitation


def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)  # H W
    # K.int_shape() Returns the shape of tensor or variable as a tuple of int or None entries
    lin1 = Dense(K.int_shape(prevlayer)[
                 3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[
                 3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])
    return x

# channel squeeze and spatial excitation


def sse_block(prevlayer, prefix):
    # Bug? Should be 1 here?
    conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal",
                  activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv

# concurrent spatial and channel squeeze and channel excitation


def csse_block(x, prefix):
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x

def Recurrent_block(x, filters, conv_layers=2):
    for i in range(2):
        if i == 0:
            x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        out = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same')(add([x, x]))
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
    return out

def RRCNN_block(x, filters, conv_layers=2):
    x1 = Conv2D(filters, kernel_size=(1, 1), strides=1, padding='same')(x)
    x2 = Recurrent_block(x1, filters, 2)
    x2 = Recurrent_block(x2, filters, 2)
    out = add([x1, x2])
    return out

# Feature Pyramid Model
def Feature_Pyr_R2_unet(input_shape = (224,224,1), nClasses=5, dropout_rate=0.0, batch_norm=True):
    # encoder

    #inputs = Input(shape=(width, height, input_channels))
    conv_layers=2
    filters=14
    UP_SAMP_SIZE =2
    inputs = layers.Input(input_shape, dtype=tf.float32)

    conv1 = RRCNN_block(inputs, filters, conv_layers=conv_layers) # (224,224,14)
    csse1 = csse_block(conv1, prefix="csse1")
    pool1 = MaxPooling2D((2, 2))(csse1) # (112,112,14)
    #pool1 = AveragePooling2D((2, 2))(conv1)



    conv2 = RRCNN_block(pool1, filters * 2, conv_layers=conv_layers) # (112,112,28)
    csse2 = csse_block(conv2, prefix="csse2")
    pool2 = MaxPooling2D((2, 2))(csse2) # (56,56,28)
    #pool2 = AveragePooling2D((2, 2))(conv2)

    conv3 = RRCNN_block(pool2, filters * 4, conv_layers=conv_layers) # (56,56,56)
    csse3 = csse_block(conv3, prefix="csse3")
    pool3 = MaxPooling2D((2, 2))(csse3) # (28,28,56)
    #pool3 = AveragePooling2D((2, 2))(conv3)

    conv4 = RRCNN_block(pool3, filters * 8, conv_layers=conv_layers)  # (28,28,112)
    csse4 = csse_block(conv4, prefix="csse4")
    pool4 = MaxPooling2D((2, 2))(csse4)  # (14,14,112)
    #pool4 = AveragePooling2D((2, 2))(conv4)

    conv5 = RRCNN_block(pool4, filters * 16, conv_layers=conv_layers) # (14,14,224)
    csse5 = csse_block(conv5, prefix="csse5")
    conv6=  RRCNN_block(csse5, filters * 16, conv_layers=conv_layers) # (14,14,224)
    csse6 = csse_block(conv6, prefix="csse6")

    conv7=  RRCNN_block(csse6, filters * 8, conv_layers=conv_layers)  # (14,14,112)
    csse7 = csse_block(conv7, prefix="csse7")
    net_up1 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last",interpolation='bilinear')(csse7)  # (28,28,112)
    new_concat1 = layers.concatenate([net_up1,csse4], axis=3)	 # (28,28,112)
    conv8=  RRCNN_block(new_concat1, filters * 8, conv_layers=conv_layers)  # (28,28,112)
    conv9=  RRCNN_block(conv8, filters * 8, conv_layers=conv_layers)   # (28,28,112)

    conv10 = RRCNN_block(conv9, filters * 4, conv_layers=conv_layers)   # (28,28,56)
    net_up2 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last",interpolation='bilinear')(conv10) # (56,56,56)
    new_concat2 = layers.concatenate([net_up2,csse3], axis=3)
    conv11 = RRCNN_block(new_concat2, filters * 4, conv_layers=conv_layers)  #(56,56,56)
    conv12 = RRCNN_block(conv11, filters * 4, conv_layers=conv_layers)   #(56,56,56)


    conv13 = RRCNN_block(conv12, filters * 2, conv_layers=conv_layers)   # (56,56,28)
    net_up3 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last",interpolation='bilinear')(conv13) # (112,112,28)
    new_concat3 = layers.concatenate([net_up3,csse2], axis=3)
    conv14 = RRCNN_block(new_concat3, filters * 2, conv_layers=conv_layers)  #(112,112,28)
    conv15 = RRCNN_block(conv14, filters * 2, conv_layers=conv_layers)   #(112,112,28)

    conv16 = RRCNN_block(conv15, filters , conv_layers=conv_layers)   # (112,112,14)

    # Feature Stacking
    pyr_layer1 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last",interpolation='bilinear')(conv16)
    pyr_layer2= layers.UpSampling2D(size=(UP_SAMP_SIZE*2, UP_SAMP_SIZE*2), data_format="channels_last",interpolation='bilinear')(conv13)
    pyr_layer3= layers.UpSampling2D(size=(UP_SAMP_SIZE*4, UP_SAMP_SIZE*4), data_format="channels_last",interpolation='bilinear')(conv10)
    pyr_layer4= layers.UpSampling2D(size=(UP_SAMP_SIZE*8, UP_SAMP_SIZE*8), data_format="channels_last",interpolation='bilinear')(csse7)



    concat1 = layers.concatenate([pyr_layer1,pyr_layer2,pyr_layer3,pyr_layer4], axis=3)
    conv17 = RRCNN_block(concat1, filters*4, conv_layers=conv_layers)  #(224,224,56)
    conv18 = RRCNN_block(conv17, filters, conv_layers=conv_layers)   #(224,224,14)

    conv_final = layers.Conv2D(nClasses, kernel_size=(1,1))(conv18)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)

    # Model integration
    model = models.Model(inputs, conv_final, name="Feature_Pyr_R2_unet")
    return model



def final_model(input_shape):
  final_unet_model = Feature_Pyr_R2_unet(input_shape=(input_shape[0],input_shape[1],1))
  return final_unet_model














