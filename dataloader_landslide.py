import scipy
import glob
import numpy as np
#import matplotlib.pyplot as plt
from skimage.transform import resize
import keras
import tensorflow
import os
from skimage import io

#class DataGenerator(keras.utils.Sequence):
DEF_MODE_NORM='01'
#DEF_NORM = 40000
DEF_NORM = 255.
list_out = [1,2,3,4]

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 dataset_img,
                 dataset_dem='',
                 dataset_mask='',
                 img_res_in=(256,256,3),
                 img_res_out=(256,256,1),
                 batch_size=16,
                 type='train',
                 shuffle=True):
        'Initialization'
        self.dataset_img= dataset_img
        self.dataset_dem= dataset_dem
        self.dataset_mask= dataset_mask
        self.img_res = (img_res_in[0],img_res_in[1])
        self.channel_in = img_res_in[2]
        self.channel_out = img_res_out[2]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.type=type
        self.lig=img_res_in[0]
        self.col=img_res_in[1]
        if self.type=='test':
            self.path_in = glob.glob('%s/*png' % (self.dataset_img))
            self.path_in.sort()
            self.path_in_dem = glob.glob('%s/*png' % (self.dataset_dem))
            self.path_in_dem.sort()
            self.path_out_classes = None
        else:
            self.path_in = glob.glob('%s/*png' % (self.dataset_img))
            self.path_in.sort()
            self.path_in_dem = glob.glob('%s/*png' % (self.dataset_dem))
            self.path_in_dem.sort()
            self.path_out_classes = glob.glob('%s/*png' % (self.dataset_mask))
            self.path_out_classes.sort()
        self.data_type=type
        self.on_epoch_end()
        #config = tf.ConfigProto()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.path_in) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        # Find list of IDs
        if self.type=='test':
            path_in_temp_ima =  [self.path_in[k] for k in indexes]
            path_in_temp_dem =  [self.path_in_dem[k] for k in indexes]
            # to get image name
            #print(path_in_temp)
            X = self.__data_generation_test(path_in_temp_ima,path_in_temp_dem)
            return X
        else:
            path_in_temp_ima = [self.path_in[k] for k in indexes]
            path_in_temp_dem = [self.path_in_dem[k] for k in indexes]
            #print(path_in_temp)
            path_out_temp_classes = [self.path_out_classes[k] for k in indexes]
            X,y = self.__data_generation(path_in_temp_ima,path_in_temp_dem,path_out_temp_classes)
            return X,y
        # Generate data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.path_in))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self,path_in_temp_ima,path_in_temp_dem,path_out_temp_classes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ima_t = []
        dem_t = []
        y_t = []
        for nimg_path in np.arange(len(path_out_temp_classes)):
#            yt_classes = self.imread_labels(path_out_temp_classes[nimg_path])
            y = self.imread_mask(path_out_temp_classes[nimg_path])
            ima = self.imread(path_in_temp_ima[nimg_path])
            dem = self.imread_dem(path_in_temp_dem[nimg_path])

            y_t.append(y)
            ima_t.append(ima)
            dem_t.append(dem)
        y = np.array(y_t)/DEF_NORM
        ima = np.array(ima_t)/DEF_NORM
        dem = np.array(dem_t)/DEF_NORM
        return {'input_image': ima, 'input_dem': dem},y

    def __data_generation_test(self,path_in_temp,path_in_temp_dem):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ima_t = []
        dem_t = []
        for nimg_path in np.arange(len(path_in_temp)):
            ima = self.imread(path_in_temp[nimg_path])
            dem = self.imread_dem(path_in_temp_dem[nimg_path])
            ima_t.append(ima)
            dem_t.append(dem)
        ima = np.array(ima_t)/DEF_NORM
        dem = np.array(dem_t)/DEF_NORM
        return {'input_image': ima, 'input_dem': dem}
#        return {'input_1': ima, 'input_2': dem}


    def imread(self, path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
#        return np.rollaxis(rio.open(path).read().astype(np.float),0,3)
        return io.imread(path)
    def imread_dem(self, path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
#        return io.imread(path)
        return io.imread(path)[:,:,0].reshape(self.img_res[0],self.img_res[1],1)
    def imread_mask(self, path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
#        return io.imread(path)
        return io.imread(path).reshape(self.img_res[0],self.img_res[1],1)

    def imread_data_16b(self,path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
        return np.rollaxis(rio.open(path).read().astype(np.float),0,3)
    def imread_labels_int(self,path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
        im=np.zeros((self.img_res[0],self.img_res[1]))
        imt=np.zeros((self.img_res[0],self.img_res[1]))
        tmp=np.rollaxis(rio.open(path).read().astype(np.int),0,3)
        for i in range(len(list_out)):
            imt=np.where(tmp[:,:,0]==list_out[i],i,0)
        im=im+imt
        return im

    def imread_labels(self,path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
        im=np.zeros((self.img_res[0],self.img_res[1],self.channel_out))
        tmp=np.rollaxis(rio.open(path).read().astype(np.int),0,3)
        for i in range(len(list_out)):
            imt=np.where(tmp[:,:,0]==list_out[i],1,0)
            im[:,:,i]=imt
        return im



class DataGenerator_oldnorm(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,
                 dataset_img,
                 dataset_dem='',
                 dataset_mask='',
                 img_res_in=(256,256,3),
                 img_res_out=(256,256,1),
                 batch_size=16,
                 type='train',
                 shuffle=True):
        'Initialization'
        self.dataset_img= dataset_img
        self.dataset_dem= dataset_dem
        self.dataset_mask= dataset_mask
        self.img_res = (img_res_in[0],img_res_in[1])
        self.channel_in = img_res_in[2]
        self.channel_out = img_res_out[2]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.type=type
        self.lig=img_res_in[0]
        self.col=img_res_in[1]
        if self.type=='test':
            self.path_in = glob.glob('%s/*png' % (self.dataset_img))
            self.path_in.sort()
            self.path_in_dem = glob.glob('%s/*png' % (self.dataset_dem))
            self.path_in_dem.sort()
            self.path_out_classes = None
        else:
            self.path_in = glob.glob('%s/*png' % (self.dataset_img))
            self.path_in.sort()
            self.path_in_dem = glob.glob('%s/*png' % (self.dataset_dem))
            self.path_in_dem.sort()
            self.path_out_classes = glob.glob('%s/*png' % (self.dataset_mask))
            self.path_out_classes.sort()
        self.data_type=type
        self.on_epoch_end()
        #config = tf.ConfigProto()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.path_in) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(indexes)
        # Find list of IDs
        if self.type=='test':
            path_in_temp_ima =  [self.path_in[k] for k in indexes]
            path_in_temp_dem =  [self.path_in_dem[k] for k in indexes]
            # to get image name
            #print(path_in_temp)
            X = self.__data_generation_test(path_in_temp_ima,path_in_temp_dem)
            return X
        else:
            path_in_temp_ima = [self.path_in[k] for k in indexes]
            path_in_temp_dem = [self.path_in_dem[k] for k in indexes]
            #print(path_in_temp)
            path_out_temp_classes = [self.path_out_classes[k] for k in indexes]
            X,y = self.__data_generation(path_in_temp_ima,path_in_temp_dem,path_out_temp_classes)
            return X,y
        # Generate data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.path_in))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self,path_in_temp_ima,path_in_temp_dem,path_out_temp_classes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ima_t = []
        dem_t = []
        y_t = []
        for nimg_path in np.arange(len(path_out_temp_classes)):
#            yt_classes = self.imread_labels(path_out_temp_classes[nimg_path])
            y = self.imread_mask(path_out_temp_classes[nimg_path])
            ima = self.imread(path_in_temp_ima[nimg_path])
            dem = self.imread_dem(path_in_temp_dem[nimg_path])

            y_t.append(y)
            ima_t.append(ima)
            dem_t.append(dem)
        y = np.array(y_t)/DEF_NORM
        ima = np.array(ima_t)/DEF_NORM
        dem = np.array(dem_t)/DEF_NORM
        return {'input_1': ima, 'input_2': dem},y

    def __data_generation_test(self,path_in_temp,path_in_temp_dem):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        ima_t = []
        dem_t = []
        for nimg_path in np.arange(len(path_in_temp)):
            ima = self.imread(path_in_temp[nimg_path])
            dem = self.imread_dem(path_in_temp_dem[nimg_path])
            ima_t.append(ima)
            dem_t.append(dem)
        ima = np.array(ima_t)/DEF_NORM
        dem = np.array(dem_t)/DEF_NORM
        return {'input_1': ima, 'input_2': dem}


    def imread(self, path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
#        return np.rollaxis(rio.open(path).read().astype(np.float),0,3)
        return io.imread(path)
    def imread_dem(self, path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
#        return io.imread(path)
        return io.imread(path)[:,:,0].reshape(self.img_res[0],self.img_res[1],1)
    def imread_mask(self, path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
#        return io.imread(path)
        return io.imread(path).reshape(self.img_res[0],self.img_res[1],1)

    def imread_data_16b(self,path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
        return np.rollaxis(rio.open(path).read().astype(np.float),0,3)
    def imread_labels_int(self,path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
        im=np.zeros((self.img_res[0],self.img_res[1]))
        imt=np.zeros((self.img_res[0],self.img_res[1]))
        tmp=np.rollaxis(rio.open(path).read().astype(np.int),0,3)
        for i in range(len(list_out)):
            imt=np.where(tmp[:,:,0]==list_out[i],i,0)
        im=im+imt
        return im

    def imread_labels(self,path):
        #return scipy.misc.imread(path, mode='RGB').astype(np.float)
        im=np.zeros((self.img_res[0],self.img_res[1],self.channel_out))
        tmp=np.rollaxis(rio.open(path).read().astype(np.int),0,3)
        for i in range(len(list_out)):
            imt=np.where(tmp[:,:,0]==list_out[i],1,0)
            im[:,:,i]=imt
        return im

