from __future__ import print_function, division
import scipy

import keras
import tensorflow as tf


from sklearn.utils import class_weight
import numpy as np
import os
# from keras_unet.models import custom_unet
import skimage.io

#0 : clearland
#1 : clearwater
#2 : cloud/shadow
#3 : snow
#4 : cloud
#5 : urbain
#6 : field_outside
from keras import backend as K


def compute_weights(Y,list_out):
    return class_weight.compute_class_weight('balanced', list_out, Y)
    
def read_labels(file):
    data = np.load(file)
    data = data.reshape((data.shape[0], data.shape[2]))
    im=np.zeros((data.shape[0],data.shape[1],len(list_out)))
    # tmp=np.rollaxis(rio.open(path).read().astype(np.int),0,3)
    for i in range(len(list_out)):
        imt=np.where(data[:,:]==list_out[i],1,0)
        im[:,:,i]=imt
    return im

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

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


class temporal_network():
    def __init__(self,
                 dataset,
                 dataset_val,
                 list_out,
                 img_rows=DEF_ROWS,
                 img_cols=DEF_COLS,
                 channels_out=DEF_CH_OUT,
                 channels_in=DEF_CH_IN,
                 filepath_save=DEF_PATH_SAVE,
                 mode_norm=DEF_MODE_NORM,
                 initial_lr=DEF_INITIAL_LR,
                 decay_rate=DEF_DECAY_RATE,
                 decay_steps=DEF_DECAY_STEPS,
                 gf=DEF_FILTERS):
        # Input shape
        self.dataset=dataset
        self.dataset_val=dataset_val
        self.list_out=list_out
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.mode_norm=mode_norm
        self.img_shape_out= (self.img_rows, self.img_cols, self.channels_out)
        self.img_shape_in = (self.img_rows, self.img_cols, self.channels_in)
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.mode_norm = mode_norm
        self.filepath_save = filepath_save
        self.train_loader=DataLoader(self.dataset)
        self.val_loader=DataLoader(self.dataset_val)
        # Configure data loader

        # Number of filters in the first layer of G and D
        self.gf = gf
        self.name_readme='%s/%s'%(self.filepath_save,DEF_README)
        self.cost=[]
        self.valid=[]
        self.epo=[]
        self.alpha = compute_weights(self.dataset,self.list_out)

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
            decay_rate=self.decay_rate)

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the network
        self.tempcnn = self.multiclass_unet1D()
        # Input images and their conditioning images
        self.tempcnn.compile(loss=weighted_categorical_crossentropy(self.alpha),
        #                      ['mse','binary_crossentropy']
                             #loss=tf.keras.losses.CategoricalCrossentropy()
                               optimizer=tf.keras.optimizers.SGD(lr=self.initial_lr, momentum=0.9),
                               metrics=["categorical_accuracy"])

    def multiclass_unet(self):
        def conv_block(inputs, filters, pool=True):
            x = Conv2D(filters, (1,3), padding="same")(inputs)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(filters, (1,3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            if pool == True:
                p = MaxPool2D((1, 2))(x)
                return x, p
            else:
                return x
        inputs = Input(self.img_shape_in)

        """ Encoder """
        x1, p1 = conv_block(inputs, self.gf, pool=True)
        x2, p2 = conv_block(p1, self.gf*2, pool=True)
        x3, p3 = conv_block(p2, self.gf*3, pool=True)
        x4, p4 = conv_block(p3, self.gf*4, pool=True)

        """ Bridge """
        b1 = conv_block(p4, self.gf*8, pool=False)

        """ Decoder """
        u1 = UpSampling2D((1, 2), interpolation="bilinear")(b1)
        c1 = Concatenate()([u1, x4])
        x5 = conv_block(c1, 4*self.gf, pool=False)

        u2 = UpSampling2D((1, 2), interpolation="bilinear")(x5)
        c2 = Concatenate()([u2, x3])
        x6 = conv_block(c2, 3*self.gf, pool=False)

        u3 = UpSampling2D((1, 2), interpolation="bilinear")(x6)
        c3 = Concatenate()([u3, x2])
        x7 = conv_block(c3, self.gf*2, pool=False)

        u4 = UpSampling2D((1, 2), interpolation="bilinear")(x7)
        c4 = Concatenate()([u4, x1])
        x8 = conv_block(c4, self.gf, pool=False)

        """ Output layer """
        output = Conv2D(self.channels_out, 1, padding="same", activation="softmax")(x8)

        return Model(inputs, output)

    def multiclass_unet1D(self):
        def conv_block(inputs, filters, pool=True):
            x = Conv1D(filters, 3, padding="same")(inputs)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv1D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            if pool == True:
                p = MaxPool1D(2)(x)
                return x, p
            else:
                return x
        inputs = Input((self.img_cols, self.channels_in))

        """ Encoder """
        x1, p1 = conv_block(inputs, self.gf, pool=True)
        x2, p2 = conv_block(p1, self.gf*2, pool=True)
        x3, p3 = conv_block(p2, self.gf*3, pool=True)
        x4, p4 = conv_block(p3, self.gf*4, pool=True)

        """ Bridge """
        b1 = conv_block(p4, self.gf*8, pool=False)

        """ Decoder """
        u1 = UpSampling1D(2)(b1)
        c1 = Concatenate()([u1, x4])
        x5 = conv_block(c1, 4*self.gf, pool=False)

        u2 = UpSampling1D(2)(x5)
        c2 = Concatenate()([u2, x3])
        x6 = conv_block(c2, 3*self.gf, pool=False)

        u3 = UpSampling1D(2)(x6)
        c3 = Concatenate()([u3, x2])
        x7 = conv_block(c3, self.gf*2, pool=False)

        u4 = UpSampling1D(2)(x7)
        c4 = Concatenate()([u4, x1])
        x8 = conv_block(c4, self.gf, pool=False)

        """ Output layer """
        output = Conv1D(self.channels_out, 1, padding="same", activation="softmax")(x8)

        return Model(inputs, output)



    def train_old(self, X,y,epochs, batch_size):
#   def train(self, X,y,epochs, batch_size):
        # Training
        # -- Callbacks
        # Parameters
        start_time = datetime.datetime.now()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20,monitor = 'val_loss'),
            tf.keras.callbacks.ModelCheckpoint(filepath='%s/model-{epoch:05d}-{loss:.5f}-{val_loss:.5f}.h5'%self.filepath_save,
                                               save_best_only=False,
                                               save_weights_only=False,
                                               monitor = 'val_loss',
                                               mode='min',
                                               verbose=1,
                                               save_freq='epoch'),
            tf.keras.callbacks.TensorBoard(log_dir='./logs')
        ]

        # self.generator.compile(optimizer=optimizer)
        print('------------------------------------')
        print('Start training (batch size %d)' % batch_size)
        print('------------------------------------')
        # Lire les données X et y
        self.tempcnn.fit(X,y,batch_size=batch_size,validation_split=0.2,validation_steps=2,epochs=epochs,callbacks=callbacks)

    def train(self,epochs, batch_size):
        start_time = datetime.datetime.now()
        moy_loss = 0
        Niter_epoch=int(self.train_loader.nb_ech/batch_size)+1


        for iter in range(epochs*Niter_epoch):
            # Sample images and their conditioning counterparts
            X, y = self.train_loader.load_batch(batch_size)

            # ------------------
            #  Train Generator
            # ------------------
            g_loss = self.tempcnn.train_on_batch(X, y)
            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            eval_g_loss = K.eval(g_loss)
            #            print("[%.7d] [time: %s] --" % (epoch, elapsed_time), 'gen losses', eval_g_loss)
            toprint = "[%.7d] [time: %s] --gen losses : %f " % (iter, elapsed_time, eval_g_loss)
            sys.stdout.write(toprint + chr(13))

            moy_loss = moy_loss + eval_g_loss / sample_interval
            # If at save interval => save generated image samples
            if iter % Niter_epoch == 0: #and iter > 3:
                print('\n')
                self.chek_training(iter, moy_loss)
                moy_loss = 0

        self.dataset.close()
        self.dataset_val.close()

    def chek_training(self,iter,gloss=0):
        self.epo.append(iter)
        lossf=0.
        Xval, Yval = self.val_loader.load_data()
        Ypredict = self.tempcnn.predict(Xval)
        valid_loss=weighted_categorical_crossentropy(self.alpha)(Yval,Ypredict)
        eval_valid_loss=K.eval(valid_loss)
        self.cost.append(gloss)
        self.valid.append(eval_valid_loss)

        fig=plt.plot(self.epo,self.cost)
        fig=plt.plot(self.epo,self.valid)
        fig=plt.legend(['loss','valid'])
        name_fig='%s/graph.png'%(self.filepath_save)
        fig.figure.savefig(name_fig)
        fig=plt.close()

        print('Iter : %.8d -- Av generator loss : %f ; Valid  LOSS : %f '%(iter,gloss, eval_valid_loss))
        self.tempcnn.save('%s/models/model_%.8d_loss_%.5f_val_%.5f.h5'%(self.filepath_save, iter,gloss,eval_valid_loss))
        self.plot_res(Ypredict,Yval,Xval)


    def plot_res(self,Ypredict,Yval,Xval):
            Ytemp = Ypredict.argmax(axis=-1).astype(np.uint8)
            labels_pred = np.zeros(Ytemp.shape)
            for i in range(len(list_out)):
                labels_pred = np.where(Ytemp == i, list_out[i], labels_pred)

            Ytemp = Yval.argmax(axis=-1).astype(np.uint8)
            labels_true = np.zeros(Ytemp.shape)
            for i in range(len(list_out)):
                labels_true = np.where(Ytemp == i, list_out[i], labels_true)

            for i in range(labels_true.shape[0]):




        fig=plt.plot(self.epo,self.cost)
        fig=plt.plot(self.epo,self.valid)
        if self.loss_gan!=0:
            fig = plt.plot(self.epo, self.loss_d)
            fig=plt.legend(['generator','valid','discriminator'])
            fid = open(self.name_readme, "a")
            fid.write('Epoch [%.8d] -- Gen loss : %f -- Disc loss : %f --  Val loss : %f\n' % (epoch, gloss, self.loss_d[-1:][0],lossf))
            fid.close()

        else:
            fid = open(self.name_readme, "a")
            fid.write('Epoch [%.8d] -- Gen loss : %f -- Val loss : %f\n' % (epoch, gloss, lossf))
            fid.close()
            fig=plt.legend(['generator','valid'])
        name_fig='%s/graph.png'%(self.filepath_save)
        fig.figure.savefig(name_fig)
        fig=plt.close()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Cloud mask estimation")
    parser.add_argument("--path_train", required=True,
                        help="path to h5 train file")
    parser.add_argument("--path_val", required=True,
                        help="path to h5 val file")
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
    parser.add_argument("--gpu_ids", type=str, default=DEF_CUDA,
                        help="priority for GPU (default : %s)"%DEF_CUDA)
    parser.add_argument("--path_model",  type=str, default=DEF_PATH_SAVE,
                        help="path to save models ")
    parser.add_argument("--pretrain", type=str, default=DEF_PATH_PRETRAIN,
                        help="pretrained model (default : %s)"%DEF_PATH_PRETRAIN)
    parser.add_argument("--norm", default=DEF_MODE_NORM,
                        help="normalisation method (def : %s)"%DEF_MODE_NORM)
    parser.add_argument("--gf", type=int, default=DEF_FILTERS,
                        help="number of filter for the residual part (default : %d)"%DEF_FILTERS)
    parser.add_argument("--initial_lr", type=float, default=DEF_INITIAL_LR,
                        help="Initial learning rate (default : %f)"%DEF_INITIAL_LR)
    parser.add_argument("--decay_steps", type=int, default=DEF_DECAY_STEPS,
                        help="Decay steps (default : %d)"%DEF_DECAY_STEPS)
    parser.add_argument("--decay_rate", type=float, default=DEF_DECAY_RATE,
                        help="Decay rate (default : %f)"%DEF_DECAY_RATE)
    parser.add_argument("--note", type=str, default='',
                        help="Personal note for the readme file ")

    args = parser.parse_args()
    dataset=args.path_train
    dataset_val=args.path_val
    batch_size = args.batch_size
    mode_norm=args.norm
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

    # Plusieurs GPU
    pretrain=args.pretrain
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id[0])  # Or 2, 3, etc. other than 01
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    session = tf.Session(config=config)

    if not os.path.exists(path_model):
        os.makedirs(path_model)
    name_readme='%s/%s'%(path_model,DEF_README)
    fid = open(name_readme, "w")
    fid.write('--------------------\n')
    fid.write('Input parameters\n')
    fid.write('Dataset train (h5)) : %s\n'%dataset)
    fid.write('Dataset val (h5)) : %s\n'%dataset_val)
    fid.write('Batch size : %d\n'%batch_size)
    fid.write('Mode norm : %s\n'%mode_norm)
    fid.write('Epochs : %d\n'%epochs)
    fid.write('Size LR (col,lig) : %d x %d \n'%(ncols,nrows))
    fid.write('Channels (in, out)  : %d x %d \n'%(ch_in,ch_out))
    fid.write('Number of filters  : %d \n'%(gf))
    fid.write('--------------------\n')
    fid.write('Losses\n')
    fid.write('--------------------\n')
    fid.write('--------------------\n')
    if pretrain is not '':
        fid.write('Pretrain\n')
        fid.write('--------------------\n')
        fid.write(pretrain)
        fid.write('--------------------\n')
    if note is not '':
        fid.write('Note : %s\n'%note)
        fid.write('--------------------\n')
    fid.write('--------------------\n')
    fid.write('Command\n')
    fid.write(' '.join(sys.argv))
    fid.write('\n')
    fid.write('--------------------\n')
    fid.close()
    net = temporal_network(dataset=dataset,dataset_val=dataset_val,
                 img_rows=nrows,
                 img_cols=ncols,
                 channels_out=ch_out,
                 channels_in=ch_in,
                 filepath_save=path_model,
                 mode_norm=mode_norm,
                 initial_lr=initial_lr,
                 decay_rate=decay_rate,
                 decay_steps=decay_steps,
                 gf=gf)
    if os.path.exists(pretrain):
        print('---------------------------------')
        print('load pretrain model %s'%pretrain)
        net.tempcnn=load_model(pretrain,custom_objects=DEF_CUSTOM_OBJECTS)
        print('Done')
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,beta_1=0.9,beta_2=0.999)
        net.tempcnn.compile(loss=weighted_categorical_crossentropy(alpha),
                              optimizer=tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9),
                              metrics=["categorical_accuracy"])
        print('---------------------------------')
    else:
        print('---------------------------------')
        print('Train from scratch')
        print('---------------------------------')


    stringlist = []
    net.tempcnn.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary)

    fid = open(name_readme, "a")
    fid.write('--------------------\n')
    fid.write('model\n')
    fid.write(short_model_summary)
    fid.close()
    # Lire X et y
    # X = np. ...


    net.train(dataset,dataset_val,epochs=epochs, batch_size=batch_size)
    #gan.train(epochs=epochs, batch_size=batch_size)


# Combine generators
# https://stackoverflow.com/questions/46313525/how-do-i-combine-two-keras-generator-functions

