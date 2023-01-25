from __future__ import print_function, division
import scipy

import keras
import tensorflow as tf
#import tensorflow-addons as tfa
#import tensorflow_addons as tfa
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
#from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.layers import LeakyReLU,ReLU,Add, PReLU,add
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose, SeparableConv2D
from tensorflow.keras.models import Sequential, Model, load_model,save_model
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.layers import MaxPool2D,multiply,Lambda

#from tensorflow.python.keras.optimizers import Adam
#import matplotlib.pyplot as plt
import sys
import numpy as np
from keras import backend as K

#from data_loader_before_generator import DataLoader

from dataloader_landslide import DataGenerator_oldnorm,DataGenerator
#from losses import *
#from losses import VGG_LOSS

import numpy as np
import os
from  skimage import io

DEF_TMPO='.tmpomodel.h5'
DEF_SEPARATOR_classes='_land'
DEF_SEPARATOR_land='_land'
DEF_SEPARATOR_gt='_gt'
DEF_BATCH_SIZE = 15
DEF_CUDA = "0"
DEF_COLS = 256
DEF_ROWS=256
DEF_CH_OUT = 1
DEF_CH_IN = 3
coef=[2.8,0.2,1]
DEF_MODE_NORM='01'
list_out = [0,1,2,3,4,5,255]


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

def my_loss(coef_focal=1.,coef_grad=0.1):
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
DEF_DECAY_STEPS = 100000000
DEF_DECAY_RATE = 0.9

alpha = [1.62916784,0.49345255,0.94459565,3.32227727]
sumalpha = np.sum(alpha)
alphanorm=alpha/sumalpha
invalphat = 1/alphanorm
suminvalpha = np.sum(invalphat)
alpha_inv= invalphat/suminvalpha


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives +
    K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

DEF_CUSTOM_OBJECTS={'categorical_focal_loss':categorical_focal_loss,'binary_focal_loss':binary_focal_loss,'categorical_focal_loss_fixed':categorical_focal_loss(alpha=[[.125, 1.8, 1.,1.6,0.16,1.7,.11]], gamma=2.),'binary_focal_loss_fixed':binary_focal_loss(gamma=2., alpha=.25),'my_loss':my_loss,'my_grad':my_grad,'loss_function':my_loss(coef_focal=1.,coef_grad=0.1),'f1_m':f1_m,'dyn_weighted_bincrossentropy':dyn_weighted_bincrossentropy,'weighted_bincrossentropy':weighted_bincrossentropy}


def change_model(model_name,newname):
    model=tf.keras.models.load_model(model_name,custom_objects=DEF_CUSTOM_OBJECTS)
    tf.keras.models.save_model(model,newname)

class test():
    def __init__(self,model_name,dataset_name_ima,dataset_name_dem,dataset_name_mask='',img_rows=DEF_ROWS,img_cols=DEF_COLS,channels_out=DEF_CH_OUT,channels_in=DEF_CH_IN,batch_size=DEF_BATCH_SIZE,oldnorm=False,gt=False):
        # Input shape
        self.model_name=model_name
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.img_shape_out = (self.img_rows, self.img_cols, self.channels_out)
        self.img_shape_in = (self.img_rows, self.img_cols, self.channels_in)
        # Configure data loader
        self.dataset_name_ima = dataset_name_ima
        self.dataset_name_dem = dataset_name_dem
        self.dataset_name_mask = dataset_name_mask
        self.batch_size=batch_size
        self.oldnorm=oldnorm
        self.gt=gt
        print('charging model')
        self.model=tf.keras.models.load_model(model_name, custom_objects=DEF_CUSTOM_OBJECTS)
        print('Done')
        if self.gt is True:
            type='eval'
        else:
            type = 'test'
        if self.oldnorm is True:
            self.data_generator = DataGenerator_oldnorm(self.dataset_name_ima,
                                                self.dataset_name_dem,
                                                self.dataset_name_mask,
                                                self.img_shape_in,
                                                self.img_shape_out,
                                                batch_size=batch_size,
                                                type=type,
                                                shuffle=False)

        else:
            self.data_generator = DataGenerator(self.dataset_name_ima,
                                                self.dataset_name_dem,
                                                self.dataset_name_mask,
                                                self.img_shape_in,
                                                self.img_shape_out,
                                                batch_size=batch_size,
                                                type=type,
                                                shuffle=False)



    def test_data(self):
        print('------------------------------------')
        print('Start predict (batch size %d)' % self.batch_size)
        print('------------------------------------')
        pred = self.model.predict(self.data_generator)
        return pred
    def evaluate_data(self):
        print('------------------------------------')
        print('Start predict (batch size %d)' % self.batch_size)
        print('------------------------------------')

        #eva = self.model.evaluate(self.data_generator,callbacks=callbacks)
        names=[]
        eva = self.model.evaluate(self.data_generator)
        for i in range(len(self.model.metrics)):
            names.append(self.model.metrics[i].name)
        return eva,names

    def save_data(self,pred,output_folder=None):
        if output_folder is None:
            for i in range(pred.shape[0]):
                name=os.path.splitext(self.data_generator.path_in[i])
                ima=io.imread('%s/%s.png'%(self.dataset_name_ima,name)).astype(np.uint8)
                #name_save_clas='%s%s.png'%(name[0],DEF_SEPARATOR_classes)
                #self.save_img_labels(name_save_clas,np.where(pred[i,:,:,:]<0.5,0,255).astype(np.uint8))
                name_save_prob='%s%s.png'%(name[0],DEF_SEPARATOR_land)
                
            
                impred =np.where(pred[i,:,:,:]<0.5,0,255).astype(np.uint8)
                improb =(pred[i,:,:,:]*255).astype(np.uint8)
                impred=np.repeat(impred.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)
                improb=np.repeat(improb.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)

                imc1=np.concatenate((ima,impred,improb),axis=1)

                #self.save_img_labels(name_save_prob,(pred[i,:,:,:]*255).astype(np.uint8))
                self.save_img(name_save_prob,imc1)
                #print('image %s saved'%name_save_prob)
        else:
            for i in range(pred.shape[0]):
                #name=os.path.splitext(self.data_generator.path_in[i])
                name_image=os.path.splitext(os.path.split(self.data_generator.path_in[i])[1])[0]
                ima=io.imread('%s/%s.png'%(self.dataset_name_ima,name_image)).astype(np.uint8)
#                name_save_clas='%s/%s%s.png'%(output_folder,name_image,DEF_SEPARATOR_classes)
#                name_save_prob='%s/%s%s.png'%(output_folder,name_image,DEF_SEPARATOR_land)
                name_save='%s/%s%s.png'%(output_folder,name_image,DEF_SEPARATOR_land)
#                self.save_img_labels(name_save_clas,pred[i,:,:,:].argmax(axis=-1))
                predict=np.where(pred[i,:,:,:]<0.5,0,255).astype(np.uint8)
                impred =np.where(pred[i,:,:,:]<0.5,0,255).astype(np.uint8)
                improb =(pred[i,:,:,:]*255).astype(np.uint8)
                impred=np.repeat(impred.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)
                improb=np.repeat(improb.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)

                imc1=np.concatenate((ima,impred,improb),axis=1)
#                self.save_img_labels(name_save_clas,predict)
#                self.save_img_labels(name_save_clas,np.where(pred[i,:,:,:]<0.5,0,255).astype(np.uint8))
                #print('image %s saved'%name_save_clas)
#                self.save_img_labels(name_save_prob,(pred[i,:,:,:]*255).astype(np.uint8))
                #print('image %s saved'%name_save_prob)
                self.save_img(name_save,imc1)

    def save_data_and_gt(self,pred,folder_gt,output_folder,output_compar):

        if output_compar is None:
            for i in range(pred.shape[0]):
                # prob and classes
                name=os.path.splitext(os.path.split(self.data_generator.path_in[i])[1])[0]
                name_save_clas='%s/%s%s.png'%(output_folder,name,DEF_SEPARATOR_classes)
                impred =np.where(pred[i,:,:,:]<0.5,0,255).astype(np.uint8)
                #self.save_img_labels(name_save_clas,impred)
                name_save_prob='%s/%s%s.png'%(output_folder,name,DEF_SEPARATOR_land)
                improb =(pred[i,:,:,:]*255).astype(np.uint8)
                #self.save_img_labels(name_save_prob,improb)
                # gt
                impred=np.repeat(impred.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)
                improb=np.repeat(improb.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)

                im_gt=io.imread('%s/%s.png'%(folder_gt,name)).astype(np.uint8)
                im_gt=np.repeat(im_gt.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)
                ima=io.imread('%s/%s.png'%(self.dataset_name_ima,name)).astype(np.uint8)

                imc1=np.concatenate((ima,im_gt),axis=1)
                imc2=np.concatenate((improb,impred),axis=1)
                imc=np.concatenate((imc1,imc2),axis=0)

                name_save='%s/%s%s.png'%(output_folder,name,DEF_SEPARATOR_gt)
                self.save_img(name_save,imc.astype(np.uint8))
        else:
             for i in range(pred.shape[0]):

                # prob and classes
                name=os.path.splitext(os.path.split(self.data_generator.path_in[i])[1])[0]
                name_save_clas='%s/%s%s.png'%(output_folder,name,DEF_SEPARATOR_classes)
                name_save_prob='%s/%s%s.png'%(output_folder,name,DEF_SEPARATOR_land)
                name_save='%s/%s%s.png'%(output_compar,name,DEF_SEPARATOR_gt)

                impred =np.where(pred[i,:,:,:]<0.5,0,255).astype(np.uint8)
                self.save_img_labels(name_save_clas,impred)
                improb =(pred[i,:,:,:]*255).astype(np.uint8)
                self.save_img_labels(name_save_prob,improb)
                # gt
                impred=np.repeat(impred.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)
                improb=np.repeat(improb.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)

                im_gt=io.imread('%s/%s.png'%(folder_gt,name)).astype(np.uint8)
                im_gt=np.repeat(im_gt.reshape(self.img_shape_in[0], self.img_shape_in[1], self.img_shape_out[2]), 3, axis=2)
                ima=io.imread('%s/%s.png'%(self.dataset_name_ima,name)).astype(np.uint8)

                imc1=np.concatenate((ima,im_gt),axis=1)
                imc2=np.concatenate((improb,impred),axis=1)
                imc=np.concatenate((imc1,imc2),axis=0)

                self.save_img(name_save,imc.astype(np.uint8))

    def save_img(self,name_save,data):
        io.imsave(name_save,data, check_contrast=False)
    def save_img_labels(self,name_save,data):
        #yclasses=data.argmax(axis=-1)
        io.imsave(name_save,data, check_contrast=False)

    def save_img_labels_rio(self,name_save,data):
        #yclasses=data.argmax(axis=-1)
        meta=DEF_META.copy()
        #data = np.where(data==6,255,data)
        with rio.open(name_save, 'w', **meta) as dst:
            dst.write(data.astype(rio.uint8), 1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Test a landslide detection model of some data")
    parser.add_argument("--data_ima", required=True,
                        help="path to image folder ")
    parser.add_argument("--data_dem", required=True,
                        help="path to dem folder ")
    parser.add_argument("--model", required=True,
                        help="path to model")
    parser.add_argument("--gpu_ids", nargs='+', type=str, default=DEF_CUDA,
                        help="priority for GPU (default : %s)"%DEF_CUDA)
    parser.add_argument("--batch_size", type=int, default=DEF_BATCH_SIZE,
                        help="size of batch (larer = speeder but need ram, default : %d)"%DEF_BATCH_SIZE)
    parser.add_argument("--ncols", type=int, default=DEF_COLS,
                        help="numbers of columns (default : %d)"%DEF_COLS)
    parser.add_argument("--nrows", type=int, default=DEF_ROWS,
                        help="numbers of rows (default : %d)"%DEF_ROWS)
    parser.add_argument("--ch_out", type=int, default=DEF_CH_OUT,
                        help="channels in output (default : %d)"%DEF_CH_OUT)
    parser.add_argument("--ch_in", type=int, default=DEF_CH_IN,
                        help="channels in input (channels in output and structure. Default : %d)"%DEF_CH_IN)
    parser.add_argument("--output", default=None,
                        help="path to output (if None : same folder as data")
    parser.add_argument("--ground_truth", default=None,
                        help="path to ground truth (if exists)")
    parser.add_argument("--oldnorm", help="(for very first models compatibility, older normalization)",action="store_true")

    #parser.add_argument("--compar_path", default=None,
    #                    help="path to save comparaison results (if ground truth exists)")

    args = parser.parse_args()

    dataset_name_ima = args.data_ima
    dataset_name_dem = args.data_dem
    ground_truth = args.ground_truth
#    compar_path = args.compar_path
    compar_path = None
    model_name = args.model
    output_path = args.output
    oldnorm=args.oldnorm
    cuda_id=args.gpu_ids
    ncols=int(args.ncols)
    nrows=int(args.nrows)
    ch_out = int(args.ch_out)
    batch = args.batch_size
    ch_in = int(args.ch_in)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id[0])  # Or 2, 3, etc. other than 01

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    session = tf.Session(config=config)
    if ground_truth is not None:
        gt=True
        dataset_name_mask=ground_truth
    else:
        gt=False
        dataset_name_mask = ''
    if output_path is not None:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    newname=DEF_TMPO
    print('loading and convert model')
  #  change_model(model_name,newname)

 #   model_test = test(newname,dataset_name_ima,dataset_name_dem,img_rows=nrows,img_cols=ncols,channels_out=ch_out,channels_in=ch_in,batch_size=batch)
    model_test = test(model_name,dataset_name_ima,dataset_name_dem,dataset_name_mask=dataset_name_mask,img_rows=nrows,img_cols=ncols,channels_out=ch_out,channels_in=ch_in,batch_size=batch,oldnorm=oldnorm,gt=gt)
    pred=model_test.test_data()
    #commande='\rm %s'%newname
    #print('remove tempo files')
    #os.system.commande(commande)
    print('save_data')

    if ground_truth is not None:
        if compar_path is not None:
            if not os.path.isdir(compar_path):
                os.makedirs(compar_path)
        model_test.save_data_and_gt(pred,ground_truth,output_path,compar_path)
    else:
        model_test.save_data(pred,output_folder=output_path)

    print('Evaluation : ')
    if ground_truth is not None:
        eval,names_eva=model_test.evaluate_data()
        name_readme = '%s/metrics.txt'%output_path
        fid = open(name_readme, "w")
        fid.write('------------ Evaluation metrics --------\n')
        for i in range(len(names_eva)):
            print(names_eva[i],' = ',eval[i])
            fid.write('%s = %f\n' %(names_eva[i],eval[i]))
        fid.close()



        
        

    



# Combine generators
# https://stackoverflow.com/questions/46313525/how-do-i-combine-two-keras-generator-functions
