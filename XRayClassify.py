# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:20:31 2018

@author: omegi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 16:14:51 2018

@author: omegi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 18:07:06 2018

@author: omegi
"""
### loss: 0.1699 acc: 0.9340
import os
import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
from keras.layers import Input, Conv2D, Reshape, Permute, Activation, MaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint # from tutorial 12 in class
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model
import itertools
from keras import backend as K
import time
from sklearn.metrics import roc_curve, auc
K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code
K.image_data_format()

## http://forums.fast.ai/t/tip-clear-tensorflow-gpu-memory/1979 ## GPU memory clear
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

limit_mem()


 # Exponential decay
def schedule(epoch):
  return 0.01 * (0.95**epoch)
  
    
def preprocess_class(datapath, subs):
    source ='img128'
    dest = 'img128S%d'%subs
    target_size=64
    subsPL1 = subs+1
    imagelist = [fn for fn in os.listdir(os.path.join(datapath, source)) if fn[-3:]=='jpg']
    N = len(imagelist)
    L_masks = {}
    Classes_orig = ['bg', 'ht', 'll', 'rl', 'lc', 'rc']
    for ii in range(N):
        fn = imagelist[ii]
        print('fn=', fn)
        fn_head, fn_tail = os.path.splitext(fn)
        img = plt.imread(os.path.join(datapath, source, fn))
        
        for this_class in Classes_orig:
            if this_class == 'bg': continue
            msk = plt.imread(os.path.join(datapath, source, this_class+fn_head[3:]+'.png'))
            L_masks[this_class] = msk
 
        for sub_s_x in range(subsPL1):
            delta_x=sub_s_x*target_size//subs
            for sub_s_y in range (subsPL1):
                delta_y = sub_s_y * target_size // subs
                This_idx = subsPL1*subsPL1*ii+subsPL1*sub_s_x+sub_s_y
                image = img[delta_x:delta_x+target_size, delta_y:delta_y+target_size]

                lc_center = L_masks['lc'][delta_x+3*target_size//8:delta_x+5*target_size//8, delta_y+3*target_size//8:delta_y+5*target_size//8]
                rc_center = L_masks['rc'][delta_x+3*target_size//8:delta_x+5*target_size//8, delta_y+3*target_size//8:delta_y+5*target_size//8]
                if np.sum(lc_center)+np.sum(rc_center)>0:
                    # Tag this sub image as a clavicle
                    plt.imsave(os.path.join(datapath, dest, '%05d-c.png'%This_idx), image)
                    #plt.imsave(os.path.join(datapath, dest, '1%05d-c.png'%This_idx), np.fliplr(image))
                else:
                    h_center = L_masks['ht'][delta_x+target_size//2:delta_x + target_size//2+2,delta_y+target_size//2:delta_y + target_size//2+2]
                    if np.sum(h_center)>0:
                        # Tag this sub image as a heart
                        plt.imsave(os.path.join(datapath, dest, '%05d-h.png'%This_idx), image)
                    else:
                        ll_center = L_masks['ll'][delta_x + target_size//2,delta_y+target_size//2]
                        rl_center = L_masks['rl'][delta_x + target_size//2,delta_y+target_size//2]
                        if ll_center+rl_center > 0:
                            # Tag this sub image as a lung
                            plt.imsave(os.path.join(datapath, dest, '%05d-l.png'%This_idx), image)
                           # plt.imsave(os.path.join(datapath, dest, '1%05d-l.png'%This_idx), np.fliplr(image))
                        else:
                            # Tag this sub image as a background
                            plt.imsave(os.path.join(datapath, dest, '%05d-b.png'%This_idx), image)
    return os.path.join(datapath, dest)
 
def preprocess(datapath):
    #
    #
    # Replace this with your preprocess method
    #
    classes = ['b','c','l','h']

    path, folder = os.path.split(datapath)
    if folder=='test':
        imagelist = [fn for fn in os.listdir(datapath) if fn[-3:]=='png']
        N = len(imagelist)
        num_classes = len(classes)
        images = np.zeros((N, 64, 64, 1))
        Y = np.zeros((N,num_classes))
        ii=0
        for fn in imagelist:
            images[ii,:,:,0] = imread(os.path.join(datapath, fn), as_gray=True)
            cc = -1
            for cl in range(len(classes)):
                if fn[-5] == classes[cl]:
                    cc = cl
            Y[ii,cc]=1
            ii += 1
    else:
        imagelistdir = [fn for fn in os.listdir(os.path.join(datapath,'augmented_data')) if fn[-3:]=='png']
        listlen = len(imagelistdir)
        iter_per_class = 2500
    
        if listlen==0:
            # This part reads the images
            imagelist = [fn for fn in os.listdir(datapath) if fn[-3:]=='png']
            #imagelist = imagelist[:-1]
            N = len(imagelist)
            num_classes = len(classes)
            images = np.zeros((N, 64, 64, 1))
            Y = np.zeros((N,num_classes))
            ii=0
            for fn in imagelist:
                images[ii,:,:,0] = imread(os.path.join(datapath, fn), as_gray=True)
                cc = -1
                for cl in range(len(classes)):
                    if fn[-5] == classes[cl]:
                        cc = cl
                Y[ii,cc]=1
                ii += 1
            # This part is a simplified preprocess
    
        
            ImageGen = ImageDataGenerator(rotation_range=15., 
                               width_shift_range=0.2, 
                               height_shift_range=0.2, 
                               shear_range=0.15, 
                               fill_mode='constant', 
                               horizontal_flip=True, 
                               data_format = 'channels_last')
            
            b_images = images[Y[:,0]==1]
            c_images = images[Y[:,1]==1]
            l_images = images[Y[:,2]==1]
            h_images = images[Y[:,3]==1]
            iter_per_class = 2500
            B_aug = ImageGen.flow(b_images, Y[np.where(Y[:,0])], batch_size=1,
                                          save_to_dir=os.path.join(datapath, 'augmented_data'), save_prefix = 'b', save_format='png')
            C_aug = ImageGen.flow(c_images, Y[np.where(Y[:,1])], batch_size=1,
                                          save_to_dir=os.path.join(datapath, 'augmented_data'), save_prefix = 'c', save_format='png')
            L_aug = ImageGen.flow(l_images, Y[np.where(Y[:,2])], batch_size=1,
                                          save_to_dir=os.path.join(datapath, 'augmented_data'), save_prefix = 'l', save_format='png')
            H_aug = ImageGen.flow(h_images, Y[np.where(Y[:,3])], batch_size=1,
                                          save_to_dir=os.path.join(datapath, 'augmented_data'), save_prefix = 'h', save_format='png')    
            for i in range(iter_per_class):
                next(B_aug)
                next(C_aug)
                next(L_aug)
                next(H_aug)
            
    
        N = len(os.listdir(datapath +'\\augmented_data'))
        num_classes = len(classes)
        imagelist = [fn for fn in os.listdir(datapath +'\\augmented_data') if fn[-3:]=='png']
        images = np.zeros((N, 64, 64, 1))
        Y = np.zeros((N,num_classes))
        ii=0
        for fn in imagelist:
            images[ii,:,:,0] = imread(os.path.join(os.path.join(datapath, 'augmented_data'), fn), as_gray=True)
            cc = -1
            for cl in range(len(classes)):
                if fn[0] == classes[cl]:
                    cc = cl
            Y[ii,cc]=1
            ii += 1        
    target_size=32
    sample=64//target_size
    BaseImages = images[:,::sample,::sample,:]
    BaseY = Y
    #TrainSet = N//8
    return BaseImages, BaseY


######################---VGG-Modified-to-fit-fully-convolutional---######################
def get_miniVGGnet(num_classes, final_avg=128):
    #
    #
    # This is just a placeholder
    # Replace the folowing with your network
    #

    inputs = Input((None,None,1))

    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_1')(inputs)
    conv1 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_2')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2,2))(conv1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_1')(pool1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_2')(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_1')(pool2)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_2')(conv3)
    conv3 = Conv2D(256, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_3')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2,2))(conv3)

#    pool5 = MaxPooling2D((2, 2), padding='same', name='pool5')(drop5)
    
#    conv0 = Conv2D(256, (3, 3), data_format = 'channels_last', 
#                   kernel_initializer="he_normal", padding='same', activation="relu", name='conv6')(pool5)
#    norm = BatchNormalization()(conv0)

    classif = Conv2D(num_classes, (final_avg,final_avg), activation='relu', padding='valid',
                             data_format = 'channels_last', name='conv_last')(pool3)

    reshape = Reshape((-1, num_classes))(classif)
    activation = Activation('softmax', name='activation')(reshape)
    model = Model(inputs=inputs, outputs=activation)
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def get_net(num_classes, final_avg=128):
    #
    #
    # This is just a placeholder
    # Replace the folowing with your network
    #

    inputs = Input((None,None,1))

    conv1 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_1')(inputs)
    conv1 = Conv2D(512, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv1_2')(conv1)
    drop1 = Dropout(0.1)(conv1)
    pool1 = MaxPooling2D((2, 2))(drop1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_1')(pool1)
    conv2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv2_2')(conv2)
    drop2 = Dropout(0.1)(conv2)
    pool2 = MaxPooling2D((2,2))(drop2)
    conv3 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_1')(pool2)
    conv3 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv3_2')(conv3)
    drop3 = Dropout(0.4)(conv3)
    pool3 = MaxPooling2D((2, 2))(drop3)
    conv4 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_1')(pool3)
    conv4 = Conv2D(64, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv4_2')(conv4)
    drop4 = Dropout(0.4)(conv4)
    conv5 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_1')(drop4)
    conv5 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv5_2')(conv5)
    drop5 = Dropout(0.4)(conv5)

    classif = Conv2D(num_classes, (final_avg,final_avg), activation='relu', padding='valid',
                             data_format = 'channels_last', name='conv_last')(drop5)

    reshape = Reshape((-1, num_classes))(classif)
    activation = Activation('softmax', name='activation')(reshape)
    model = Model(inputs=inputs, outputs=activation)
    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()

def plot_ROC(fpr,tpr,roc_auc,labels,n_classes,title):
    plt.figure(dpi=500)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f} {2:s})'
                 ''.format(i, roc_auc[i], labels[i]))
    plt.title('ROC curve' + title)
    plt.tight_layout()
    plt.legend()
    plt.savefig('images/ROC curve'+ title +'.png', bbox_inches='tight')


def train(X, Y, model_fn, epochs, X_val, Y_val):
    class_names = ['Background', 'Clavicles', 'Lungs', 'Heart']
    target_size=4
    start_InvNet = time.clock()

    NNet = get_net(Y.shape[-1], final_avg=target_size)
#    lr_decay_exp_cb = LearningRateScheduler(schedule)
    cp_cb = ModelCheckpoint('weights/weights.Invnet_modefied{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5', 
                            monitor='val_acc', 
                            save_best_only=False, 
                            save_weights_only=True)
    history_I = NNet.fit(X, Y, epochs=epochs, batch_size=64, verbose=1, validation_data=(X_val, Y_val), callbacks=[cp_cb])
    plot_history(history_I, title='InvNet modified by me',label='InvNet',epochs=epochs)
    
    plot_model(NNet, 'model.png',show_shapes=True)
    
    print(time.clock() - start_InvNet)
    Y_val_classes = Y_val.argmax(axis=-1)
    Y_pred = NNet.predict(X_val)
    Y_classes = Y_pred.argmax(axis=-1)
    
    ################################
    ##-------modified-VGG16-------##
    ################################
    start_VGG7 = time.clock()

    miniVGGNet = get_miniVGGnet(Y.shape[-1], final_avg=target_size)
#    lr_decay_exp_cb = LearningRateScheduler(schedule)
    cp_cb = ModelCheckpoint('weights/weights.miniVGGnet_modefied{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5', 
                            monitor='val_acc', 
                            save_best_only=False, 
                            save_weights_only=True)
    history_V = miniVGGNet.fit(X, Y, epochs=epochs, batch_size=64, verbose=1, validation_data=(X_val, Y_val), callbacks=[cp_cb])
    plot_history(history_V,baseline=history_I,title='miniVGGnet (VGG7)',label='miniVGG',label_b='InvNet',epochs=epochs)
   
    
    print(time.clock() - start_VGG7)

    Y_pred_V = miniVGGNet.predict(X_val)
    Y_classes_V = Y_pred_V.argmax(axis=-1)    
    ######################
    ##------Graphs------##
    ######################
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_val_classes, Y_classes)
    np.set_printoptions(precision=2)

    cnf_matrix_V = confusion_matrix(Y_val_classes, Y_classes_V)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix NNet
    plt.figure(dpi=500)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix modified InvertedNet')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.savefig('images/Normalized Confusion Matrix - InvertedNet modified by Or.png', bbox_inches='tight')

    # Plot normalized confusion matrix NNet
    plt.figure(dpi=500)
    plot_confusion_matrix(cnf_matrix_V, classes=class_names, normalize=True, title='Normalized confusion matrix miniVGGnet (VGG7)')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.savefig('images/Normalized Confusion Matrix - miniVGG (VGG7).png', bbox_inches='tight')
    
    #### plot ROC curve ####
    #### https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a ####
    fpr = dict()
    tpr = dict()
    classes = dict()
    roc_auc = dict()
#    Y_classes_ROC = label_binarize(Y_pred.argmax(axis=-1).ravel(),classes=[0,1,2,3])
    Y_val_roc = Y_val[:,0]
    Y_pred_roc = Y_pred[:,0]
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(Y_val_roc[:,i], Y_pred_roc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        classes[i]= class_names[i]
    plot_ROC(fpr,tpr,roc_auc,classes,4,' InvertedNet modified by Or')

    fpr = dict()
    tpr = dict()
    classes = dict()
    roc_auc = dict()
#    Y_classes_ROC = label_binarize(Y_pred_V.argmax(axis=-1).ravel(),classes=[0,1,2,3])
    Y_pred_V = Y_pred_V[:,0]
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(Y_val_roc[:, i], Y_pred_V[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        classes[i]= class_names[i]
    plot_ROC(fpr,tpr,roc_auc,classes,4,' miniVGGnet')

    
    NNet.save(model_fn)
    return NNet

def test(X, Y, model_fn):
    NNet = load_model(model_fn)
    Y_hat = NNet.predict(X)
    class_names = ['Background', 'Clavicles', 'Lungs', 'Heart']

    Y_classes = Y_hat.argmax(axis=-1)  
    Y_gt = Y.argmax(axis=-1)
    ######################
    ##------Graphs------##
    ######################
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_gt, Y_classes)
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix NNet
    plt.figure(dpi=500)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized Test confusion matrix modified InvertedNet')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.savefig('images/Normalized Confusion Matrix Test - InvertedNet modified by Or.png', bbox_inches='tight')

    fpr = dict()
    tpr = dict()
    classes = dict()
    roc_auc = dict()
    Y_pred = Y_hat[:,0]
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        classes[i]= class_names[i]
    plot_ROC(fpr,tpr,roc_auc,classes,4,' Test InvertedNet')

    return Y_hat

def plot_history(history, baseline = None, title=None,label=None,label_b=None,epochs=100):
    plt.figure(dpi=500)
    his = history.history
    val_acc = his['val_acc']    
    train_acc = his['acc']
    plt.plot(np.arange(1,len(val_acc)+1),val_acc, label=label+'val_acc')
    plt.plot(np.arange(1,len(train_acc)+1),train_acc,label = label+'acc')

    if baseline is not None:
        his = baseline.history
        val_acc = his['val_acc']
        train_acc = his['acc']
        plt.plot(np.arange(1,len(val_acc)+1),val_acc,label=label_b+'baseline val_acc')
        plt.plot(np.arange(1,len(train_acc)+1),train_acc, label=label_b+'baseline acc') 
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.xlabel('epochs')
    plt.xticks(np.arange(1,epochs, step=5),np.arange(1,epochs, step=5))
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig('images/' + title +'accuracy.png', bbox_inches='tight')


if __name__ == "__main__":
    #
    #
    #
    # Replace this with your base path
    #
    if os.path.isdir('images')!=True:
        os.mkdir('images')
        os.mkdir('weights')
    basepath = 'C:\\Users\\omegi\\Documents\\HW03\\ML4BME-ex3' 
    datapath = os.path.join(basepath, 'data')
    task='test'
#    task = 'train'
#    task='preproc_class' 
    Sub_samp = 4
    num_classes = 4
    epochs = 80
    batch_size = 32
    if task == 'preproc_class':
        # This prepares the dataset for the classification task
        class_datapath = preprocess_class(datapath, Sub_samp)
    if task == 'train':
        
        class_path = os.path.join(datapath,'img128S4')
        X, Y = preprocess(class_path)
        X = X.astype('float32')
        Y = Y.astype('float32')
        # data normalization       
        X/=255
        # spliting the data
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=42)
        
        Y_train = Y_train.reshape((-1,1,num_classes))
        Y_val = Y_val.reshape((-1,1,num_classes))
        model = train(X_train, Y_train, '200827293_ex3_1', epochs, X_val, Y_val)

    if task == 'test':
        testpath = os.path.join(datapath, 'test')
        X, Y = preprocess(testpath)
        Y_hat = test(X, Y, '200827293_ex3_1')
        print(np.sum(np.abs(Y_hat.reshape((-1,1,1,4))-Y.reshape((-1,1,1,4)))))


        