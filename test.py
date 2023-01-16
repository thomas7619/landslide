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
#DEF_META= {'driver': 'PNG', 'dtype': 'uint8', 'nodata': None, 'width': 256, 'height': 256, 'count': 1, 'crs': None, 'transform': rio.transform.Affine(1.0, 0.0, 0.0,
#       0.0, 1.0, 0.0)}
def vide_loss(x1,x2):
    return x1+x2
DEF_CUSTOM_OBJECTS={'binary_focal_loss':vide_loss,
                    'binary_focal_loss_fixed':vide_loss,
                    'categorical_focal_loss':vide_loss,
                    'categorical_focal_loss_fixed':vide_loss,
                    'myloss':vide_loss,'vide_loss':vide_loss}


def change_model(model_name,newname):
    model=tf.keras.models.load_model(model_name,custom_objects=DEF_CUSTOM_OBJECTS)
    tf.keras.models.save_model(model,newname)

class test():
    def __init__(self,model_name,dataset_name_ima,dataset_name_dem,img_rows=DEF_ROWS,img_cols=DEF_COLS,channels_out=DEF_CH_OUT,channels_in=DEF_CH_IN,batch_size=DEF_BATCH_SIZE,oldnorm=False):
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
        self.batch_size=batch_size
        self.oldnorm=oldnorm
        print('charging model')
        self.model=tf.keras.models.load_model(model_name, custom_objects=DEF_CUSTOM_OBJECTS)
        print('Done')
        if self.oldnorm is True:
            self.data_generator = DataGenerator_oldnorm(self.dataset_name_ima,
                                                self.dataset_name_dem,
                                                '',
                                                self.img_shape_in,
                                                self.img_shape_out,
                                                batch_size=batch_size,
                                                type='test',
                                                shuffle=False)

        else:
            self.data_generator = DataGenerator(self.dataset_name_ima,
                                                self.dataset_name_dem,
                                                '',
                                                self.img_shape_in,
                                                self.img_shape_out,
                                                batch_size=batch_size,
                                                type='test',
                                                shuffle=False)



    def test_data(self):
        print('------------------------------------')
        print('Start predict (batch size %d)' % self.batch_size)
        print('------------------------------------')
        pred = self.model.predict(self.data_generator)
        return pred

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
    if output_path is not None:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
    newname=DEF_TMPO
    print('loading and convert model')
  #  change_model(model_name,newname)

 #   model_test = test(newname,dataset_name_ima,dataset_name_dem,img_rows=nrows,img_cols=ncols,channels_out=ch_out,channels_in=ch_in,batch_size=batch)
    model_test = test(model_name,dataset_name_ima,dataset_name_dem,img_rows=nrows,img_cols=ncols,channels_out=ch_out,channels_in=ch_in,batch_size=batch,oldnorm=oldnorm)
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


        
        

    



# Combine generators
# https://stackoverflow.com/questions/46313525/how-do-i-combine-two-keras-generator-functions
