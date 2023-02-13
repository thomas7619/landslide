from __future__ import print_function, division
import scipy

import keras
import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,SpatialDropout2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU,ReLU,Add, PReLU,add
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose, SeparableConv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.layers import MaxPool2D,multiply,Lambda

from keras.optimizers import Adam
import datetime
import sys
import numpy as np
from keras import backend as K


from dataloader_landslide import *


import numpy as np
import os
import skimage.io


def dyn_weighted_bincrossentropy(true, pred):
    """
    Calculates weighted binary cross entropy. The weights are determined dynamically
    by the balance of each category. This weight is calculated for each batch.
    
    The weights are calculted by determining the number of 'pos' and 'neg' classes
    in the true labels, then dividing by the number of total predictions.
    
    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.
    
    This can be useful for unbalanced catagories.
    """
    # get the total number of inputs
    num_pred = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) + keras.backend.sum(true)
    
    # get weight of values in 'pos' category
    zero_weight =  keras.backend.sum(true)/ num_pred +  keras.backend.epsilon()
    
    # get weight of values in 'false' category
    one_weight = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) / num_pred +  keras.backend.epsilon()

    # calculate the weight vector
    weights =  (1.0 - true) * zero_weight +  true * one_weight
    
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy

    return keras.backend.mean(weighted_bin_crossentropy)


def weighted_bincrossentropy(true, pred, weight_zero = 0.25, weight_one = 1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.
    
    Adjust the weights here depending on what is required.
    
    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives
        will be penalize 10 times as much as false negatives.
    """
  
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return keras.backend.mean(weighted_bin_crossentropy)
    
    
def my_grad(x1,x2):
    gx1,gy1=tf.image.image_gradients(x1)
    gx2,gy2=tf.image.image_gradients(x2)
    norm1=tf.math.sqrt(gx1*gx1+gy1*gy1)
    norm2=tf.math.sqrt(gx2*gx2+gy2*gy2)
    loss=tf.math.reduce_mean(tf.multiply(norm1-norm2,norm1-norm2))
    return K.mean(loss)

def my_loss(coef_focal=1.,coef_grad=0.01):
    def loss_function(y_true,y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        loss=coef_focal*tf.keras.losses.BinaryFocalCrossentropy()(y_true,y_pred)+coef_grad*my_grad(y_true,y_pred)
        return K.mean(loss)
    return loss_function

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

DEF_BATCH_SIZE = 10
DEF_DROPOUT = 0.2
DEF_EPOCHS = 50
DEF_CUDA = "0"
DEF_COLS = 256
DEF_ROWS=256
# nombre de classes
DEF_CH_OUT = 1
DEF_CH_IN = 3
DEF_PATH_SAVE='./save'
DEF_PATH_PRETRAIN=''
DEF_README='readme_model.txt'
DEF_INITIAL_LR=0.001
DEF_FILTERS=32
DEF_DECAY_STEPS = 500
DEF_DECAY_RATE = 0.9
DEF_ALPHA = 0.2

alpha = [1.62916784,0.49345255,0.94459565,3.32227727]
sumalpha = np.sum(alpha)
alphanorm=alpha/sumalpha
invalphat = 1/alphanorm
suminvalpha = np.sum(invalphat)
alpha_inv= invalphat/suminvalpha


def recall_m(y_true2, y_pred2):
    y_true=tf.where(y_true2>0.5,tf.ones_like(y_true2),tf.zeros_like(y_true2))
    y_pred=tf.where(y_pred2>0.5,tf.ones_like(y_pred2),tf.zeros_like(y_pred2))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 
    K.epsilon())
    return recall

def precision_m(y_true2, y_pred2):
    y_true=tf.where(y_true2>0.5,tf.ones_like(y_true2),tf.zeros_like(y_true2))
    y_pred=tf.where(y_pred2>0.5,tf.ones_like(y_pred2),tf.zeros_like(y_pred2))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

DEF_CUSTOM_OBJECTS={'categorical_focal_loss':categorical_focal_loss,'binary_focal_loss':binary_focal_loss,'categorical_focal_loss_fixed':categorical_focal_loss(alpha=[[.125, 1.8, 1.,1.6,0.16,1.7,.11]], gamma=2.),'binary_focal_loss_fixed':binary_focal_loss(gamma=2., alpha=.25),'my_loss':my_loss,'my_grad':my_grad,'loss_function':my_loss(coef_focal=1.,coef_grad=0.1),'f1_m':f1_m,'dyn_weighted_bincrossentropy':dyn_weighted_bincrossentropy,'weighted_bincrossentropy':weighted_bincrossentropy}


class landslide():
    def __init__(self,
                 dataset_ima,
                 dataset_dem,
                 dataset_mask,
                 path_val_img,
                 path_val_dem,
                 path_val_mask,
                 img_rows=DEF_ROWS,
                 img_cols=DEF_COLS,
                 channels_out=DEF_CH_OUT,
                 channels_in=DEF_CH_IN,
                 filepath_save=DEF_PATH_SAVE,
                 initial_lr=DEF_INITIAL_LR,
                 decay_rate=DEF_DECAY_RATE,
                 decay_steps=DEF_DECAY_STEPS,
                 gf=DEF_FILTERS,
                 dropout_rate=DEF_DROPOUT,
                 concaten_dem_only=False,concaten_ima_only=False,efficientnet=False,nores=False,vgg=False,pretrain='pretrain',train_back=False,resnet=False,mixup=False,mixup3=False,alpha_mixup=DEF_ALPHA):

        self.dataset_img= dataset_img
        self.dataset_dem= dataset_dem
        self.dataset_mask= dataset_mask
        self.path_val_img=path_val_img
        self.path_val_dem=path_val_dem
        self.path_val_mask=path_val_mask
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.img_shape_out= (self.img_rows, self.img_cols, self.channels_out)
        self.img_shape_in = (self.img_rows, self.img_cols, self.channels_in)
        self.img_shape_dem = (self.img_rows, self.img_cols, 1)
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.filepath_save = filepath_save
        self.concaten_ima_only = concaten_ima_only
        self.concaten_dem_only = concaten_dem_only
        self.efficientnet = efficientnet
        self.vgg=vgg
        self.resnet=resnet
        self.mixup=mixup
        self.mixup3=mixup3
        self.alpha_mixup=alpha_mixup
        self.train_back=train_back
        self.pretrain=pretrain
        self.dropout_rate=dropout_rate
        # Use residual convolution blocks
        if nores is True:
            self.residual=False
        else:
            self.residual=True

        # Configure data loader

        # Number of filters in the first layer of G and D
        self.gf = gf
        self.name_readme='%s/%s'%(self.filepath_save,DEF_README)

        fid = open(self.name_readme, "a")
        fid.write('--------------------\n')
        fid.write('Network Optimization\n')
        fid.write('--------------------\n')
        fid.write('Initial learning rate : %.6f\n' % self.initial_lr)
        fid.write('Decay rate : %.6f\n' % self.decay_rate)
        fid.write('Decay steps : %d\n' % self.decay_steps)
        fid.close()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,staircase=True)
#        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,beta_1=0.9,beta_2=0.999)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        #optimizer = Adam(0.0002, 0.5)

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        if self.pretrain != '':
            self.generator =load_model(pretrain,custom_objects=DEF_CUSTOM_OBJECTS)
        elif self.efficientnet:
            self.generator = self.landslide_attention_efficientnet(self.img_shape_in,
                                                                   self.img_shape_dem,
                                                                   self.channels_out,
                                                                   self.gf,
                                                                   self.concaten_dem_only,
                                                                   self.concaten_ima_only)
        elif self.vgg:
            self.generator = self.landslide_attention_vgg(self.img_shape_in,
                                                          self.img_shape_dem,
                                                          self.channels_out,
                                                          self.gf,
                                                          self.concaten_dem_only,
                                                          self.concaten_ima_only)
        elif self.resnet:
            self.generator = self.landslide_attention_resnet(self.img_shape_in,
                                                          self.img_shape_dem,
                                                          self.channels_out,
                                                          self.gf,
                                                          self.concaten_dem_only,
                                                          self.concaten_ima_only)
        else:
            self.generator = self.landslide_attention()
        print('Generator build')
        #self.generator.compile(loss=binary_focal_loss(gamma=2., alpha=.25),
        #                       metrics=['accuracy',f1_m, tf.keras.metrics.BinaryIoU()],
        #                       optimizer=optimizer)


#        self.generator.compile(loss=tf.keras.losses.BinaryCrossentropy(),
        #self.generator.compile(loss=dyn_weighted_bincrossentropy,
#        self.generator.compile(loss=my_loss(),
        if ((self.mixup is False) and (self.mixup3 is False)):
            self.generator.compile(loss=dyn_weighted_bincrossentropy,
            metrics=['accuracy', f1_m, tf.keras.metrics.BinaryIoU()],
            optimizer=optimizer)
        else:
            self.generator.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(),
            metrics=['accuracy', f1_m, tf.keras.metrics.BinaryIoU()],
            optimizer=optimizer)

        print('Generator compiled')
        
    
    def landslide_attention_vgg(self,img_shape_in,img_shape_dem,channels_out,
                                gf,concaten_dem_only,concaten_ima_only):
        from tensorflow.keras.applications import VGG16

        def conv_block(inputs, filters, pool=True):
            x = Conv2D(filters, 3, padding="same")(inputs)

            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            if pool:
                p = MaxPool2D((2, 2))(x)
                p=SpatialDropout2D(self.dropout_rate)(p)
                return x, p
            else:
                return x
                
        def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
            x = Conv2D(filters, size, strides=strides, padding=padding)(x)
            x = BatchNormalization()(x)
            if activation == True:
                x = LeakyReLU(alpha=0.1)(x)
            return x

        def residual_block(blockInput, num_filters=16,pool=False):
            xi = Conv2D(num_filters, (3, 3), activation=None, padding="same")(blockInput)
            x = LeakyReLU(alpha=0.1)(xi)
            x = BatchNormalization()(x)
            xi = BatchNormalization()(xi)
            x = convolution_block(x, num_filters, (3,3) )
            x = convolution_block(x, num_filters, (3,3), activation=False)
            x = Add()([x, xi])
            if pool:
                p = MaxPool2D((2, 2))(x)
                p=SpatialDropout2D(self.dropout_rate)(p)
                return x, p
            else:
                return x
                
        def attention_block_2d(x, g, inter_channel):
            # theta_x(?,g_height,g_width,inter_channel)
            theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
            # phi_g(?,g_height,g_width,inter_channel)
            phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
            # f(?,g_height,g_width,inter_channel)
            f = Activation('relu')(add([theta_x, phi_g]))
            # psi_f(?,g_height,g_width,1)
            psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
            rate = Activation('sigmoid')(psi_f)
            att_x = multiply([x, rate])
            return att_x

        def attention_up_and_concate(down_layer, layer):
            in_channel = down_layer.get_shape().as_list()[3]
            up = UpSampling2D(size=(2, 2))(down_layer)
            layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
            my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
            concate = my_concat([up, layer])
            concate=SpatialDropout2D(self.dropout_rate)(concate)
            return concate
            
        def convolution(blockInput, num_filters,pool=False,residual=True):
            if residual is True:
                return residual_block(blockInput, num_filters,pool=pool)
            else:
                return conv_block(blockInput, num_filters,pool=pool)




            
        """ Encoder image """
        inputs_ima = Input(self.img_shape_in,name="input_image")
        """ Pre-trained VGG16 Model """
        vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs_ima)
        if self.train_back is False:
            for layer in vgg16.layers:
                layer.trainable=False
        
        """ Encoder """
        x1i = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
        x2i = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
        x3i = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
        x4i = vgg16.get_layer("block4_conv3").output         ## (64 x 64)
        """ Bridge """
        b1 = vgg16.get_layer("block5_conv3").output         ## (32 x 32)

        inputs_dem = Input(self.img_shape_dem,name="input_dem")
        """ Encoder dem """
        x1d, p1d = convolution(inputs_dem, self.gf/2,pool=True,residual=self.residual)
        x2d, p2d = convolution(p1d, self.gf ,pool=True,residual=self.residual)
        x3d, p3d = convolution(p2d, self.gf * 2,pool=True,residual=self.residual)
        x4d, p4d = convolution(p3d, self.gf * 3,pool=True,residual=self.residual)
        if self.concaten_dem_only is True:
            skip4=x4d
            skip3=x3d
            skip2=x2d
            skip1=x1d
        elif self.concaten_ima_only is True:
            skip4=x4i
            skip3=x3i
            skip2=x2i
            skip1=x1i
        else:
            x4d = Conv2D(512, 3, padding="same")(x4d)
            skip4 = Add()([x4i, x4d])
            x3d = Conv2D(256, 3, padding="same")(x3d)
            skip3 = Add()([x3i, x3d])
            x2d = Conv2D(128, 3, padding="same")(x2d)
            skip2 = Add()([x2i, x2d])
            x1d = Conv2D(64, 3, padding="same")(x1d)
            skip1 = Add()([x1i, x1d])

        bridge = Concatenate(axis=-1,name="fusion_encoders")([b1, p4d])
        b1 = convolution(bridge, gf * 8,pool=False,residual=self.residual)
        """ Decoder """
        """ Concaten DEM only """
        x = attention_up_and_concate(b1,skip4)
        x = convolution(x, 4 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x, skip3)
        x = convolution(x, 3 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x,skip2)
        x = convolution(x, 2 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x,skip1)
        x = convolution(x, 2 * gf, pool=False,residual=self.residual)




        """ Output layer """
        output = Conv2D(self.channels_out, 1, padding="same", activation="sigmoid")(x)
        return Model([inputs_ima,inputs_dem], output)

        

    def landslide_attention_resnet(self,img_shape_in,img_shape_dem,channels_out,
                                gf,concaten_dem_only,concaten_ima_only):
        from tensorflow.keras.applications import ResNet50

        def conv_block(inputs, filters, pool=True):
            x = Conv2D(filters, 3, padding="same")(inputs)

            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            if pool:
                p = MaxPool2D((2, 2))(x)
                p = SpatialDropout2D(self.dropout_rate)(p)

                return x, p
            else:
                return x
                
        def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
            x = Conv2D(filters, size, strides=strides, padding=padding)(x)
            x = BatchNormalization()(x)
            if activation == True:
                x = LeakyReLU(alpha=0.1)(x)
            return x

        def residual_block(blockInput, num_filters=16,pool=False):
            xi = Conv2D(num_filters, (3, 3), activation=None, padding="same")(blockInput)
            x = LeakyReLU(alpha=0.1)(xi)
            x = BatchNormalization()(x)
            xi = BatchNormalization()(xi)
            x = convolution_block(x, num_filters, (3,3) )
            x = convolution_block(x, num_filters, (3,3), activation=False)
            x = Add()([x, xi])
            if pool:
                p = MaxPool2D((2, 2))(x)
                p = SpatialDropout2D(self.dropout_rate)(p)
                return x, p
            else:
                return x
                
        def attention_block_2d(x, g, inter_channel):
            # theta_x(?,g_height,g_width,inter_channel)
            theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
            # phi_g(?,g_height,g_width,inter_channel)
            phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
            # f(?,g_height,g_width,inter_channel)
            f = Activation('relu')(add([theta_x, phi_g]))
            # psi_f(?,g_height,g_width,1)
            psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
            rate = Activation('sigmoid')(psi_f)
            att_x = multiply([x, rate])
            return att_x

        def attention_up_and_concate(down_layer, layer):
            in_channel = down_layer.get_shape().as_list()[3]
            up = UpSampling2D(size=(2, 2))(down_layer)
            layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
            my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
            concate = my_concat([up, layer])
            concate = SpatialDropout2D(self.dropout_rate)(concate)
            return concate
            
        def convolution(blockInput, num_filters,pool=False,residual=True):
            if residual is True:
                return residual_block(blockInput, num_filters,pool=pool)
            else:
                return conv_block(blockInput, num_filters,pool=pool)




            
        """ Encoder image """
        inputs_ima = Input(self.img_shape_in,name="input_image")
        """ Pre-trained VGG16 Model """
#        vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs_ima)
        resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs_ima)
        if self.train_back is False:
            for layer in resnet50.layers:
                layer.trainable=False
        
        """ Encoder """
        x1i = resnet50.get_layer("input_image").output         ## (512 x 512)
        x2i = resnet50.get_layer("conv1_relu").output         ## (256 x 256)
        x3i = resnet50.get_layer("conv2_block3_out").output         ## (128 x 128)
        x4i = resnet50.get_layer("conv3_block4_out").output         ## (64 x 64)
        """ Bridge """
        b1 = resnet50.get_layer("conv4_block6_out").output         ## (32 x 32)

        inputs_dem = Input(self.img_shape_dem,name="input_dem")
        """ Encoder dem """
        x1d, p1d = convolution(inputs_dem, self.gf/2,pool=True,residual=self.residual)
        x2d, p2d = convolution(p1d, self.gf ,pool=True,residual=self.residual)
        x3d, p3d = convolution(p2d, self.gf * 2,pool=True,residual=self.residual)
        x4d, p4d = convolution(p3d, self.gf * 3,pool=True,residual=self.residual)
        if self.concaten_dem_only is True:
            skip4=x4d
            skip3=x3d
            skip2=x2d
            skip1=x1d
        elif self.concaten_ima_only is True:
            skip4=x4i
            skip3=x3i
            skip2=x2i
            skip1=x1i
        else:
            x4d = Conv2D(512, 3, padding="same")(x4d)
            skip4 = Add()([x4i, x4d])
            x3d = Conv2D(256, 3, padding="same")(x3d)
            skip3 = Add()([x3i, x3d])
            x2d = Conv2D(64, 3, padding="same")(x2d)
            skip2 = Add()([x2i, x2d])
            x1d = Conv2D(3, 3, padding="same")(x1d)
            skip1 = Add()([x1i, x1d])

        bridge = Concatenate(axis=-1,name="fusion_encoders")([b1, p4d])
        b1 = convolution(bridge, gf * 8,pool=False,residual=self.residual)
        """ Decoder """
        """ Concaten DEM only """
        x = attention_up_and_concate(b1,skip4)
        x = convolution(x, 4 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x, skip3)
        x = convolution(x, 3 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x,skip2)
        x = convolution(x, 2 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x,skip1)
        x = convolution(x, 2 * gf, pool=False,residual=self.residual)




        """ Output layer """
        output = Conv2D(self.channels_out, 1, padding="same", activation="sigmoid")(x)
        return Model([inputs_ima,inputs_dem], output)

        




    def landslide_attention_efficientnet(self,img_shape_in,img_shape_dem,channels_out,gf,concaten_dem_only,concaten_ima_only):
        from tensorflow.keras.applications import EfficientNetB0
        def conv_block(inputs, filters, pool=True):
            x = Conv2D(filters, 3, padding="same")(inputs)

            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            if pool:
                p = MaxPool2D((2, 2))(x)
                p = SpatialDropout2D(self.dropout_rate)(p)
                return x, p
            else:
                return x
    

        def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
            x = Conv2D(filters, size, strides=strides, padding=padding)(x)
            x = BatchNormalization()(x)
            if activation == True:
                x = LeakyReLU(alpha=0.1)(x)
            return x

        def residual_block(blockInput, num_filters=16,pool=False):
            x = LeakyReLU(alpha=0.1)(blockInput)
            x = BatchNormalization()(x)
            blockInput = BatchNormalization()(blockInput)
            x = convolution_block(x, num_filters, (3,3) )
            x = convolution_block(x, num_filters, (3,3), activation=False)
            x = Add()([x, blockInput])
            if pool:
                p = MaxPool2D((2, 2))(x)
                p = SpatialDropout2D(self.dropout_rate)(p)
                return x, p
            else:
                return x



        def attention_block_2d(x, g, inter_channel):
            # theta_x(?,g_height,g_width,inter_channel)
            theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
            # phi_g(?,g_height,g_width,inter_channel)
            phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
            # f(?,g_height,g_width,inter_channel)
            f = Activation('relu')(add([theta_x, phi_g]))
            # psi_f(?,g_height,g_width,1)
            psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
            rate = Activation('sigmoid')(psi_f)
            att_x = multiply([x, rate])
            return att_x

        def attention_up_and_concate(down_layer, layer):
            in_channel = down_layer.get_shape().as_list()[3]
            up = UpSampling2D(size=(2, 2))(down_layer)
            layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
            my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
            concate = my_concat([up, layer])
            concate = SpatialDropout2D(self.dropout_rate)(concate)
            return concate
        def num_layer(model,name_layer):
            for i in range(len(model.layers)):
                if name_layer not in model.layers[i].name:
                    continue
                n=i
                print('number layer ',i,model.layers[i].name)
            return n

        """ Encoder image """
        #inputs = tf.keras.layers.Input(shape=img_shape_in)
        backbone = EfficientNetB0(weights='imagenet',include_top=False,input_shape=img_shape_in)
        #input = backbone(inputs)
        input = backbone.input


        nconv0= num_layer(backbone,'normalization')

        nconv1= num_layer(backbone,'block2a_expand_activation')
        nconv2= num_layer(backbone,'block3a_expand_activation')
        nconv3= num_layer(backbone,'block4a_expand_activation')
        nconv4= num_layer(backbone,'block6a_expand_activation')
    
        x1i = backbone.layers[nconv0].output # 86
        x2i = backbone.layers[nconv1].output # 86
        x3i = backbone.layers[nconv2].output # 144
        x4i = backbone.layers[nconv3].output # 240
        x5i = backbone.layers[nconv4].output # 480

    
        """ Encoder dem """
        input_dem = Input(img_shape_dem,name="input_2")
        xc = Conv2D(gf*1,(3, 3), activation=None, padding="same")(input_dem)
        x1d, p1d = residual_block(xc, gf,pool=True)
        p1d = Conv2D(gf*2,(3, 3), activation=None, padding="same")(p1d)
        x2d, p2d = residual_block(p1d, gf*2,pool=True )
        p2d = Conv2D(gf*4,(3, 3), activation=None, padding="same")(p2d)
        x3d, p3d = residual_block(p2d, gf * 4,pool=True)
        p3d = Conv2D(gf*8,(3, 3), activation=None, padding="same")(p3d)
        x4d, p4d = residual_block(p3d, gf * 8,pool=True)
    
        """ Bridge"""
        bridge = Concatenate(axis=-1,name="fusion_encoders")([x5i, p4d])
        bridge = Conv2D(gf*8,(3, 3), activation=None, padding="same")(bridge)
        bridge = residual_block(bridge, gf * 8)
        if self.concaten_dem_only is True:
            skip4=x4d
            skip3=x3d
            skip2=x2d
            skip1=x1d
        elif self.concaten_ima_only is True:
            skip4=x4i
            skip3=x3i
            skip2=x2i
            skip1=x1i
        else:
            x4d = Conv2D(240, 3, padding="same")(x4d)
            skip4 = Add()([x4i, x4d])
            x3d = Conv2D(144, 3, padding="same")(x3d)
            skip3 = Add()([x3i, x3d])
            x2d = Conv2D(96, 3, padding="same")(x2d)
            skip2 = Add()([x2i, x2d])
            x1d = Conv2D(3, 3, padding="same")(x1d)
            skip1 = Add()([x1i, x1d])


        x = attention_up_and_concate(bridge,skip4)
        x = Conv2D(gf * 8, (3, 3), activation=None, padding="same")(x)
        x = residual_block(x,gf * 8)
        x = residual_block(x,gf * 8)
        x = LeakyReLU(alpha=0.1)(x)

        x = attention_up_and_concate(x,skip3)
        x = Conv2D(gf * 4, (3, 3), activation=None, padding="same")(x)
        x = residual_block(x,gf * 4)
        x = residual_block(x,gf * 4)
        x = LeakyReLU(alpha=0.1)(x)

        x = attention_up_and_concate(x,skip2)
        x = Conv2D(gf * 2, (3, 3), activation=None, padding="same")(x)
        x = residual_block(x,gf * 2)
        x = residual_block(x,gf * 2)
        x = LeakyReLU(alpha=0.1)(x)

        x = attention_up_and_concate(x,skip1)
        x = Conv2D(gf , (3, 3), activation=None, padding="same")(x)
        x = residual_block(x,gf )
        x = residual_block(x,gf )
        x = LeakyReLU(alpha=0.1)(x)
        
        
        x = Conv2D(gf * 1, (3, 3), activation=None, padding="same")(x)
        x = residual_block(x,gf * 1)
        x = residual_block(x,gf * 1)
        x = LeakyReLU(alpha=0.1)(x)
    
        x = Dropout(self.dropout_rate/2)(x)
        output_layer = Conv2D(channels_out, (1,1), padding="same", activation="sigmoid")(x)
        return Model([input,input_dem], output_layer)

    
    def landslide_attention(self):
        def conv_block(inputs, filters, pool=True):
            x = Conv2D(filters, 3, padding="same")(inputs)

            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            if pool:
                p = MaxPool2D((2, 2))(x)
                p = SpatialDropout2D(self.dropout_rate)(p)
                return x, p
            else:
                return x
                
        def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
            x = Conv2D(filters, size, strides=strides, padding=padding)(x)
            x = BatchNormalization()(x)
            if activation == True:
                x = LeakyReLU(alpha=0.1)(x)
            return x

        def residual_block(blockInput, num_filters=16,pool=False):
            xi = Conv2D(num_filters, (3, 3), activation=None, padding="same")(blockInput)
            x = LeakyReLU(alpha=0.1)(xi)
            x = BatchNormalization()(x)
            xi = BatchNormalization()(xi)
            x = convolution_block(x, num_filters, (3,3) )
            x = convolution_block(x, num_filters, (3,3), activation=False)
            x = Add()([x, xi])
            if pool:
                p = MaxPool2D((2, 2))(x)
                p = SpatialDropout2D(self.dropout_rate)(p)
                return x, p
            else:
                return x

        def attention_block_2d(x, g, inter_channel):
            # theta_x(?,g_height,g_width,inter_channel)
            theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
            # phi_g(?,g_height,g_width,inter_channel)
            phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
            # f(?,g_height,g_width,inter_channel)
            f = Activation('relu')(add([theta_x, phi_g]))
            # psi_f(?,g_height,g_width,1)
            psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
            rate = Activation('sigmoid')(psi_f)
            att_x = multiply([x, rate])
            return att_x

        def attention_up_and_concate(down_layer, layer):
            in_channel = down_layer.get_shape().as_list()[3]
            up = UpSampling2D(size=(2, 2))(down_layer)
            layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4)
            my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
            concate = my_concat([up, layer])
            concate = SpatialDropout2D(self.dropout_rate)(concate)
            return concate
            
        def convolution(blockInput, num_filters,pool=False,residual=True):
            if residual is True:
                return residual_block(blockInput, num_filters,pool=pool)
            else:
                return conv_block(blockInput, num_filters,pool=pool)

                

        inputs_ima = Input(self.img_shape_in,name="input_image")
        """ Encoder image """
        x1i, p1i = convolution(inputs_ima, self.gf,pool=True,residual=self.residual)
        x2i, p2i = convolution(p1i, self.gf * 2,pool=True,residual=self.residual)
        x3i, p3i = convolution(p2i, self.gf * 3,pool=True,residual=self.residual)
        x4i, p4i = convolution(p3i, self.gf * 4,pool=True,residual=self.residual)
        
        inputs_dem = Input(self.img_shape_dem,name="input_dem")
        """ Encoder dem """
        x1d, p1d = convolution(inputs_dem, self.gf,pool=True,residual=self.residual)
        x2d, p2d = convolution(p1d, self.gf * 2 ,pool=True,residual=self.residual)
        x3d, p3d = convolution(p2d, self.gf * 3,pool=True,residual=self.residual)
        x4d, p4d = convolution(p3d, self.gf * 4,pool=True,residual=self.residual)

        bridge = Concatenate(axis=-1,name="fusion_encoders")([p4i, p4d])
    #    bridge = p4d
        b1 = convolution(bridge, gf * 8,pool=False,residual=self.residual)

        if self.concaten_dem_only is True:
            skip4=x4d
            skip3=x3d
            skip2=x2d
            skip1=x1d
        elif self.concaten_ima_only is True:
            skip4=x4i
            skip3=x3i
            skip2=x2i
            skip1=x1i
        else:
            skip4 = Add()([x4i, x4d])
            skip3 = Add()([x3i, x3d])
            skip2 = Add()([x2i, x2d])
            skip1 = Add()([x1i, x1d])

        """ Decoder """
        x = attention_up_and_concate(b1,skip4)
        x = convolution(x, 4 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x, skip3)
        x = convolution(x, 3 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x,skip2)
        x = convolution(x, 2 * gf, pool=False,residual=self.residual)
        x = attention_up_and_concate(x,skip1)
        x = convolution(x, 2 * gf, pool=False,residual=self.residual)




        """ Output layer """
        output = Conv2D(self.channels_out, 1, padding="same", activation="sigmoid")(x)
        return Model([inputs_ima,inputs_dem], output)


    def train(self, epochs, batch_size):
        # Training
        # -- Callbacks
        # Parameters
        start_time = datetime.datetime.now()
        #class_weight={0: 0, 1: 0}

        callbacks = [
#            tf.keras.callbacks.EarlyStopping(patience=435,monitor = 'val_loss'),
            tf.keras.callbacks.EarlyStopping(patience=75,monitor = 'val_loss',start_from_epoch=100),
#            tf.keras.callbacks.ModelCheckpoint(filepath='%s/models/model-{epoch:05d}-{loss:.5f}-{val_loss:.5f}.h5'%self.filepath_save,
            tf.keras.callbacks.ModelCheckpoint(filepath='%s/models/best_current_model.h5'%self.filepath_save,
#            tf.keras.callbacks.ModelCheckpoint(filepath='%s/models/model-{epoch:05d}-{loss:.5f}-{accuracy:.5f}.h5'%self.filepath_save,
                                               save_best_only=True,
                                               save_weights_only=False,
                                               #monitor = ['accuracy','val_meanIoU'],
                                               monitor = 'val_loss',
                                               verbose=2,
                                               save_freq='epoch'),#,class_weight=class_weight),
        ]       # update_freq log a batch-level summary every N
        if self.efficientnet is False:
            if self.mixup is True:
                training_generator = MixUpGenerator(self.dataset_img,
                                                   self.dataset_dem,
                                                   self.dataset_mask,
                                                   self.img_shape_in,
                                                   self.img_shape_out,
                                                   batch_size=batch_size,
                                                   type='train',
                                                   shuffle=True,alpha=self.alpha_mixup)
            elif self.mixup3 is True:
                training_generator = MixUp3_Generator(self.dataset_img,
                                                   self.dataset_dem,
                                                   self.dataset_mask,
                                                   self.img_shape_in,
                                                   self.img_shape_out,
                                                   batch_size=batch_size,
                                                   type='train',
                                                   shuffle=True,alpha=self.alpha_mixup)
            else:
                training_generator = DataGenerator(self.dataset_img,
                                                   self.dataset_dem,
                                                   self.dataset_mask,
                                                   self.img_shape_in,
                                                   self.img_shape_out,
                                                   batch_size=batch_size,
                                                   type='train',
                                                   shuffle=True)

            valid_generator = DataGenerator(self.path_val_img,
                                            self.path_val_dem,
                                            self.path_val_mask,
                                            self.img_shape_in,
                                            self.img_shape_out,
                                            batch_size=batch_size,
                                            type='val',
                                            shuffle=False)
        else:
            if self.mixup is True:
                training_generator = MixUpGenerator_oldnorm(self.dataset_img,
                                                   self.dataset_dem,
                                                   self.dataset_mask,
                                                   self.img_shape_in,
                                                   self.img_shape_out,
                                                   batch_size=batch_size,
                                                   type='train',
                                                   shuffle=True,alpha=self.alpha_mixup)
            elif self.mixup3 is True:
                training_generator = MixUp3_Generator_oldnorm(self.dataset_img,
                                                   self.dataset_dem,
                                                   self.dataset_mask,
                                                   self.img_shape_in,
                                                   self.img_shape_out,
                                                   batch_size=batch_size,
                                                   type='train',
                                                   shuffle=True,alpha=self.alpha_mixup)
            else:
                training_generator = DataGenerator_oldnorm(self.dataset_img,
                                                   self.dataset_dem,
                                                   self.dataset_mask,
                                                   self.img_shape_in,
                                                   self.img_shape_out,
                                                   batch_size=batch_size,
                                                   type='train',
                                                   shuffle=True)

            valid_generator = DataGenerator_oldnorm(self.path_val_img,
                                            self.path_val_dem,
                                            self.path_val_mask,
                                            self.img_shape_in,
                                            self.img_shape_out,
                                            batch_size=batch_size,
                                            type='val',
                                            shuffle=False)

        print('------------------------------------')
        print('Start training (batch size %d)' % batch_size)
        print('------------------------------------')
        #hist = self.generator.fit(training_generator,validation_data=valid_generator,validation_steps=2,epochs=epochs,callbacks=callbacks)
        hist = self.generator.fit(training_generator,validation_data=valid_generator,validation_steps=1,epochs=epochs,callbacks=callbacks)
        return hist


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Landslide detection with attention layers and fusion image/dem")
    parser.add_argument("--path_img", required=True,
                        help="path to images")
    parser.add_argument("--path_dem", required=True,
                        help="path to dem")
    parser.add_argument("--path_mask", required=True,
                        help="path to classes")
    parser.add_argument("--val_img", default='',
                        help="path to validation images  ")
    parser.add_argument("--val_dem", default='',
                        help="path to validation dem  ")
    parser.add_argument("--val_mask", default='',
                        help="path to validation classes  ")
    parser.add_argument("--path_model",  type=str, default=DEF_PATH_SAVE,
                        help="path to save models ")
    parser.add_argument("--batch_size", type=int, default=DEF_BATCH_SIZE,
                        help="Size of the batch (default : %d)"%DEF_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEF_EPOCHS,
                        help="Number of epochs (default : %d)"%DEF_EPOCHS)
    parser.add_argument("--ncols", type=int, default=DEF_COLS,
                        help="numbers of columns of low res (default : %d)"%DEF_COLS)
    parser.add_argument("--nrows", type=int, default=DEF_ROWS,
                        help="numbers of rows of low res (default : %d)"%DEF_ROWS)
    parser.add_argument("--ch_out", type=int, default=DEF_CH_OUT,
                        help="channels in output (default : %d)"%DEF_CH_OUT)
    parser.add_argument("--ch_in", type=int, default=DEF_CH_IN,
                        help="channels in input (default : %d)"%DEF_CH_IN)
    parser.add_argument("--dropout", type=float, default=DEF_DROPOUT,
                        help="dropout rate (default : %f)"%DEF_DROPOUT)
    parser.add_argument("--gpu_ids", type=str, default=DEF_CUDA,
                        help="priority for GPU (default : %s)"%DEF_CUDA)
    parser.add_argument("--pretrain", type=str, default=DEF_PATH_PRETRAIN,
                        help="pretrained model (default : %s)"%DEF_PATH_PRETRAIN)
    parser.add_argument("--gf", type=int, default=DEF_FILTERS,
                        help="number of filter for the residual part (default : %d)"%DEF_FILTERS)
    parser.add_argument("--initial_lr", type=float, default=DEF_INITIAL_LR,
                        help="Initial learning rate (default : %f)"%DEF_INITIAL_LR)
    parser.add_argument("--decay_steps", type=int, default=DEF_DECAY_STEPS,
                        help="Decay steps (default : %d)"%DEF_DECAY_STEPS)
    parser.add_argument("--decay_rate", type=float, default=DEF_DECAY_RATE,
                        help="Decay rate (default : %f)"%DEF_DECAY_RATE)
    parser.add_argument("--mixup", help="Use mixup (with pairs)",action="store_true")
    parser.add_argument("--mixup3", help="Use mixup (with tuples)",action="store_true")
    parser.add_argument("--alpha", type=float, default=DEF_ALPHA,
                        help="Alpha coef in beta function for mixup (if mixup is True. default : %f)"%DEF_ALPHA)
    parser.add_argument("--efficientnet", help="Use efficientnet backcone",action="store_true")
    parser.add_argument("--vgg", help="Use VGG16 backcone",action="store_true")
    parser.add_argument("--resnet", help="Use ResNet backcone",action="store_true")
    parser.add_argument("--train_backbone", help="Train also the backcone (VGG16 or ResNet50) parameters (if vgg or resnet)",action="store_true")
    parser.add_argument("--concat_ima", help="Concatene image only in decoder (instead of images and DEM by default)",action="store_true")
    parser.add_argument("--concat_dem", help="Concatene DEM only in decoder (instead of images and DEM by default)",action="store_true")
    parser.add_argument("--nores", help="Use classic convolutions instead of residual ones",action="store_true")
    parser.add_argument("--note", type=str, default='',
                        help="Personal note for the readme file ")

    args = parser.parse_args()
    dataset_img=args.path_img
    dataset_dem=args.path_dem
    dataset_mask=args.path_mask
    path_val_img=args.val_img
    path_val_dem=args.val_dem
    path_val_mask=args.val_mask
    batch_size = args.batch_size
    epochs = int(args.epochs)
    ncols=int(args.ncols)
    nrows=int(args.nrows)
    ch_out = int(args.ch_out)
    ch_in = int(args.ch_in)
    path_model = args.path_model
    cuda_id=args.gpu_ids
    gf= args.gf
    note=args.note
    decay_steps=args.decay_steps
    decay_rate=args.decay_rate
    initial_lr=args.initial_lr
    concaten_ima_only = args.concat_ima
    concaten_dem_only = args.concat_dem
    efficientnet = args.efficientnet
    nores = args.nores
    pretrain = args.pretrain
    vgg = args.vgg
    resnet = args.resnet
    mixup = args.mixup
    mixup3 = args.mixup3
    alpha_mixup = args.alpha
    train_back = args.train_backbone
    dropout_rate = args.dropout

    pretrain=args.pretrain
    config = tf.ConfigProto()
    #config = tf.compat.v1.ConfigProto
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id[0])  # Or 2, 3, etc. other than 01

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    session = tf.Session(config=config)

    #session = tf.compat.v1.Session(config=config)
    if ((mixup3 is True) and (mixup is True)):
        sys.exit('impossible to have mixup and mixup3 options')
#        return 0
#    gan = Cloudy2free(dataset_name=dataset_name)
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    if not os.path.exists('%s/models'%path_model):
        os.makedirs('%s/models'%path_model)
    if not os.path.exists('%s/python'%path_model):
        os.makedirs('%s/python'%path_model)
    comm = 'cp *py %s/python/'%path_model
    os.system(comm)
    name_readme='%s/%s'%(path_model,DEF_README)
    fid = open(name_readme, "w")
    fid.write('--------------------\n')
    fid.write('Input parameters\n')
    fid.write('Dataset train img : %s\n'%dataset_img)
    fid.write('Dataset train dem : %s\n'%dataset_dem)
    fid.write('Dataset train mask : %s\n'%dataset_mask)
    fid.write('Dataset val img : %s\n'%path_val_img)
    fid.write('Dataset val dem : %s\n'%path_val_dem)
    fid.write('Dataset val mask : %s\n'%path_val_mask)
    fid.write('Batch size : %d\n'%batch_size)
    fid.write('Epochs : %d\n'%epochs)
    if mixup is True:
        fid.write('Use MixUp (pairs) data augmentation with alpha = %f\n'%alpha_mixup)
    if mixup3 is True:
        fid.write('Use MixUp (tuples) data augmentation with alpha = %f\n'%alpha_mixup)
    fid.write('Dropout Rate : %f\n'%dropout_rate)
    fid.write('Size im (col,lig) : %d x %d \n'%(ncols,nrows))
    fid.write('Channels (in, out)  : %d x %d \n'%(ch_in,ch_out))
    fid.write('Number of filters  : %d \n'%(gf))
    if concaten_dem_only is True:
        fid.write('Use image only in decoder\n')
    elif concaten_ima_only is True:
        fid.write('Use DEM only in decoder\n')
    else:
        fid.write('Use Images and DEM in decoder\n')
    if efficientnet is True:
        fid.write('Use Efficientnet backbone encoder\n')
    if resnet is True:
        fid.write('Use ResNet backbone encoder\n')
        if train_back is True:
            fid.write('Retrain ResNet encoder\n')
        else:
            fid.write('Freeze ResNet encoder\n')
    if vgg is True:
        fid.write('Use VGG backbone encoder\n')
        if train_back is True:
            fid.write('Retrain VGG encoder\n')
        else:
            fid.write('Freeze VGG encoder\n')


    elif nores is True:
        fid.write('Use classic 2D convolutions\n')
    else:
        fid.write('Use residual 2D convolutions\n')


    if pretrain != '':
        fid.write('Pretrain\n')
        fid.write('--------------------\n')
        fid.write(pretrain)
        fid.write('--------------------\n')
    if note != '':
        fid.write('Note : %s\n'%note)
        fid.write('--------------------\n')
    fid.write('--------------------\n')
    fid.write('Command\n')
    fid.write(' '.join(sys.argv))
    fid.write('\n')
    fid.write('--------------------\n')
    fid.close()
    network = landslide(dataset_img,
                 dataset_dem,
                 dataset_mask,
                 path_val_img,
                 path_val_dem,
                 path_val_mask,
                 img_rows=nrows,
                 img_cols=ncols,
                 channels_out=ch_out,
                 channels_in=ch_in,
                 filepath_save=path_model,
                 initial_lr=initial_lr,
                 decay_rate=decay_rate,
                 decay_steps=decay_steps,
                 dropout_rate=dropout_rate,
                 gf=gf,concaten_ima_only=concaten_ima_only,concaten_dem_only=concaten_dem_only,efficientnet=efficientnet,nores=nores,vgg=vgg,pretrain=pretrain,train_back=train_back,resnet=resnet,mixup=mixup,mixup3=mixup3,alpha_mixup=alpha_mixup)
    if os.path.exists(pretrain):
        print('---------------------------------')
        print('load pretrain model %s'%pretrain)
        print('---------------------------------')
    else:
        print('---------------------------------')
        print('Train from scratch')
        print('---------------------------------')


    #s=str(gan.generator.summary())
    stringlist = []
    network.generator.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)

    fid = open(name_readme, "a")
    fid.write('--------------------\n')
    fid.write('model\n')
    fid.write(short_model_summary)
    fid.close()

    history = network.train(epochs=epochs, batch_size=batch_size)
    np.save("%s/history.npy"%path_model,history.history)
    #hist=np.load("%s/history.npy"%path_model,allow_pickle=True).item()

 #   else:
 #       gan.train_save(epochs=epochs, batch_size=batch_size, sample_interval=save_interval)


# Combine generators
# https://stackoverflow.com/questions/46313525/how-do-i-combine-two-keras-generator-functions
