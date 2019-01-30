import os
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint # from tutorial 12 in class
#from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from skimage.transform import resize
import itertools
import time

K.set_image_dim_ordering('tf')  # Tensorflow dimension ordering in this code
K.image_data_format()

### http://forums.fast.ai/t/tip-clear-tensorflow-gpu-memory/1979 ## GPU memory clear
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

limit_mem()

    
def preprocess_seg(datapath):
    # Read all the *.jpg images (x-ray images)
    imagelist = [fn for fn in os.listdir(os.path.join(datapath)) if fn[-3:]=='jpg']
    N = len(imagelist)
    
    img = np.zeros((N,128,128,1))
    Y = np.zeros((N,128,128,4)).astype(bool)
    # 80x80 is the largest resolution the computer could manage 
    X = np.zeros((N,80,80,1))
    # creating an array that fits the size of the last layer of the network 
    Y_shape = np.zeros((N,7,7,4))
   
    Classes_orig = ['bg', 'ht', 'll', 'rl', 'lc', 'rc']
    classes = ['b','c','l','h']
    for ii in range(N):
        fn = imagelist[ii]
        print('fn=', fn)
        fn_head, fn_tail = os.path.splitext(fn)
        img[ii,:,:,0] = plt.imread(os.path.join(datapath, fn))
        X[ii,:,:,0] = resize(img[ii,:,:,0],(80,80))
        for this_class in Classes_orig:
            if this_class == 'bg': continue
            mask = plt.imread(os.path.join(datapath, this_class+fn_head[3:]+'.png')).astype(bool)
            ## create background mask
            Y[ii,:,:,classes.index('b')]|= mask
            if this_class == 'lc' or this_class == 'rc':
                # create clavicles mask (ground truth)
                Y[ii,:,:,classes.index('c')]|= mask
            if this_class == 'll' or this_class == 'rl':
                # create lungs mask (ground truth)
                Y[ii,:,:,classes.index('l')]|= mask
            if this_class == 'ht':
                # create heart mask (ground truth)
                Y[ii,:,:,classes.index('h')]|= mask
        Y[ii,:,:,classes.index('b')] = ~Y[ii,:,:,classes.index('b')] 
        Y_shape[ii,:,:,classes.index('b')] = resize(Y[ii,:,:,classes.index('b')] , (7,7))
        Y_shape[ii,:,:,classes.index('c')] = resize(Y[ii,:,:,classes.index('c')] , (7,7))
        Y_shape[ii,:,:,classes.index('l')] = resize(Y[ii,:,:,classes.index('l')] , (7,7))
        Y_shape[ii,:,:,classes.index('h')] = resize(Y[ii,:,:,classes.index('h')] , (7,7))

    Y.astype('float32')
    Y_shape.astype('float32')
    
    return img, X, Y, Y_shape



def get_net(num_classes, final_avg=128):

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
#######################################################################################
##--------Took out the reshape so my output will be an image and not a vector--------##
#######################################################################################

#    reshape = Reshape((-1, num_classes))(classif)

    activation = Activation('softmax', name='activation')(classif)
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
    plt.savefig('images_seg/ROC curve'+ title +'.png', bbox_inches='tight')


def train(X, Y, model_fn, epochs, X_val, Y_val, img):
    class_names = ['Background', 'Clavicles', 'Lungs', 'Heart']
    target_size = 4
    start_InvNet = time.clock()

    NNet = get_net(Y.shape[-1], final_avg=target_size)
    NNet = load_model('200827293_ex3_2')
    cp_cb = ModelCheckpoint('weights_seg/weights.Invnet_modefied_section_D{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5', 
                            monitor='val_acc', 
                            save_best_only=True, 
                            save_weights_only=True)
    
    history_I = NNet.fit(X, Y, epochs=epochs, batch_size=None, verbose=1, validation_data=(X_val, Y_val), callbacks=[cp_cb])
    plot_history(history_I, title='InvNet modified by me section D',label='InvNet',epochs=epochs)
    

    print(time.clock() - start_InvNet)
    Y_pred = NNet.predict(X_val)

    N = X_val.shape[0]
    Y_pred_big = np.zeros((N,128,128,4))
    Y_val_big = np.zeros((N,128,128,4))
    #######################
    ##----Resize back----##
    #######################
    for ii in range(0,N):
        Y_pred_big[ii,:,:,class_names.index('Background')] = resize(Y_pred[ii,:,:,class_names.index('Background')], (128,128)) 
        Y_pred_big[ii,:,:,class_names.index('Clavicles')]  = resize(Y_pred[ii,:,:,class_names.index('Clavicles')],  (128,128)) 
        Y_pred_big[ii,:,:,class_names.index('Lungs')]      = resize(Y_pred[ii,:,:,class_names.index('Lungs')],      (128,128)) 
        Y_pred_big[ii,:,:,class_names.index('Heart')]      = resize(Y_pred[ii,:,:,class_names.index('Heart')],      (128,128)) 
        Y_val_big[ii,:,:,class_names.index('Background')]  = resize(Y_val[ii,:,:,class_names.index('Background')],  (128,128)) 
        Y_val_big[ii,:,:,class_names.index('Clavicles')]   = resize(Y_val[ii,:,:,class_names.index('Clavicles')],   (128,128)) 
        Y_val_big[ii,:,:,class_names.index('Lungs')]       = resize(Y_val[ii,:,:,class_names.index('Lungs')],       (128,128)) 
        Y_val_big[ii,:,:,class_names.index('Heart')]       = resize(Y_val[ii,:,:,class_names.index('Heart')],       (128,128)) 
     ######################
     ##------Graphs------##
     ######################
        

    Y_pred_mat = Y_pred[61,:].argmax(axis=-1).ravel()
    Y_val_mat = Y_val[61,:].argmax(axis=-1).ravel()
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y_val_mat, Y_pred_mat)
    np.set_printoptions(precision=2)
#
    # Plot normalized confusion matrix NNet
    plt.figure(dpi=500)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix modified InvertedNet section D')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.savefig('images_seg/Normalized Confusion Matrix - InvertedNet modified by Or section D image #80.png', bbox_inches='tight')
    #####################################
    ##----Small images presentation----##
    #####################################
    
    plt.figure(dpi=300)
    plt.title('7x7 images example')
    plt.subplot(241)
    plt.gca().set_title('Background')
    plt.imshow(Y_pred[61,:,:,class_names.index('Background')])
    plt.subplot(242)
    plt.gca().set_title('Clavicles')
    plt.imshow(Y_pred[61,:,:,class_names.index('Clavicles')])
    plt.subplot(243)
    plt.gca().set_title('Lungs')
    plt.imshow(Y_pred[61,:,:,class_names.index('Lungs')])
    plt.subplot(244)
    plt.gca().set_title('Heart')
    plt.imshow(Y_pred[61,:,:,class_names.index('Heart')])
    plt.subplot(245)
    plt.imshow(Y_val[61,:,:,class_names.index('Background')])
    plt.subplot(246)
    plt.imshow(Y_val[61,:,:,class_names.index('Clavicles')])
    plt.subplot(247)
    plt.imshow(Y_val[61,:,:,class_names.index('Lungs')])
    plt.subplot(248)
    plt.imshow(Y_val[61,:,:,class_names.index('Heart')])
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig('images_seg/7by7images.png')
    
    ##############################################
    ##----Scaled up back images presentation----##
    ##############################################
    
    plt.figure(dpi=300)
    plt.title('128x128 images example')
    plt.subplot(241)
    plt.gca().set_title('Background')
    plt.imshow(Y_pred_big[61,:,:,class_names.index('Background')])
    plt.subplot(242)
    plt.gca().set_title('Clavicles')
    plt.imshow(Y_pred_big[61,:,:,class_names.index('Clavicles')])
    plt.subplot(243)
    plt.gca().set_title('Lungs')
    plt.imshow(Y_pred_big[61,:,:,class_names.index('Lungs')])
    plt.subplot(244)
    plt.gca().set_title('Heart')
    plt.imshow(Y_pred_big[61,:,:,class_names.index('Heart')])
    plt.subplot(245)
    plt.imshow(Y_val_big[61,:,:,class_names.index('Background')])
    plt.subplot(246)
    plt.imshow(Y_val_big[61,:,:,class_names.index('Clavicles')])
    plt.subplot(247)
    plt.imshow(Y_val_big[61,:,:,class_names.index('Lungs')])
    plt.subplot(248)
    plt.imshow(Y_val_big[61,:,:,class_names.index('Heart')])
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig('images_seg/128by128images.png')
    ##############################
    ####----plot ROC curve----####
    ##############################

    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    fpr = dict()
    tpr = dict()
    classes = dict()
    roc_auc = dict()
    Y_classes_ROC = label_binarize(Y_pred_mat,classes=[0,1,2,3])
    Y_val_ROC = label_binarize(Y_val_mat,classes=[0,1,2,3])

    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(Y_val_ROC[:,i], Y_classes_ROC[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        classes[i]= class_names[i]
    plot_ROC(fpr,tpr,roc_auc,classes,4,' InvertedNet modified by Or section D')
 

    NNet.save(model_fn)

    return NNet


def test(X, Y, model_fn):
    NNet = load_model(model_fn)
    Y_hat = NNet.predict(X)
    class_names = ['Background', 'Clavicles', 'Lungs', 'Heart']

    ######################
    ##------Graphs------##
    ######################
    # Compute confusion matrix    
    Y_hat_mat = Y_hat[0,:].argmax(axis=-1).ravel()
    Y_mat = Y[0,:].argmax(axis=-1).ravel()

    cnf_matrix = confusion_matrix(Y_mat, Y_hat_mat,labels=[0,1,2,3])
    np.set_printoptions(precision=2)
    
    # Plot normalized confusion matrix NNet
    plt.figure(dpi=500)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized Test confusion matrix modified InvertedNet section D')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    plt.savefig('images_seg/Normalized Confusion Matrix Test - InvertedNet modified by Or  section D.png', bbox_inches='tight')

    #####################################
    ##----Small images presentation----##
    #####################################
    
    plt.figure(dpi=300)
    plt.title('128x128 images example')
    plt.subplot(241)
    plt.gca().set_title('Background')
    plt.imshow(Y_hat[0,:,:,class_names.index('Background')])
    plt.subplot(242)
    plt.gca().set_title('Clavicles')
    plt.imshow(Y_hat[0,:,:,class_names.index('Clavicles')])
    plt.subplot(243)
    plt.gca().set_title('Lungs')
    plt.imshow(Y_hat[0,:,:,class_names.index('Lungs')])
    plt.subplot(244)
    plt.gca().set_title('Heart')
    plt.imshow(Y_hat[0,:,:,class_names.index('Heart')])
    plt.subplot(245)
    plt.imshow(Y[0,:,:,class_names.index('Background')])
    plt.subplot(246)
    plt.imshow(Y[0,:,:,class_names.index('Clavicles')])
    plt.subplot(247)
    plt.imshow(Y[0,:,:,class_names.index('Lungs')])
    plt.subplot(248)
    plt.imshow(Y[0,:,:,class_names.index('Heart')])
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig('images_seg/128by128images.png')


    fpr = dict()
    tpr = dict()
    classes = dict()
    roc_auc = dict()
    Y_classes_ROC = label_binarize(Y_hat_mat,classes=[0,1,2,3])
    Y_ROC = label_binarize(Y_mat,classes=[0,1,2,3])

    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(Y_ROC[:,i], Y_classes_ROC[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        classes[i]= class_names[i]
    plot_ROC(fpr,tpr,roc_auc,classes,4,' Test InvertedNet section D')

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
    plt.savefig('images_seg/' + title +'accuracy.png', bbox_inches='tight')


if __name__ == "__main__":
 
    if os.path.isdir('images_seg')!=True:
        os.mkdir('images_seg')
        os.mkdir('weights_seg')
    #######################################################################
    ##----Path for the base folder, thw one who's 'data' folder is in----##
    #######################################################################
    basepath = 'C:\\Users\\omegi\\Documents\\HW03\\ML4BME-ex3\\ex3_2' 
    datapath = os.path.join(basepath, 'data')
#    task='test'
    task = 'train'
#    task='preproc_class' 
#    Sub_samp = 4
    num_classes = 4
    epochs = 80
    batch_size = 32
#    if task == 'preproc_class':
#        # This prepares the dataset for the classification task
#        class_datapath = preprocess_class(datapath, Sub_samp)
    if task == 'train':
        
        class_path = os.path.join(datapath,'img128')
        img, X, Y, Y_shape = preprocess_seg(class_path)
        X = X.astype('float32')
        Y = Y.astype('float32')
        Y_shape = Y_shape.astype('float32')
        # data normalization     
        X/=255
        # spliting the data
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y_shape, test_size=0.4, random_state=42)
        
        model = train(X_train, Y_train, '200827293_ex3_2', epochs, X_val, Y_val, img)

    if task == 'test':
        testpath = os.path.join(datapath, 'test')
        X,_, Y,_ = preprocess_seg(testpath)
        X.astype('float32')
        Y.astype('float32')
        Y_hat = test(X, Y, '200827293_ex3_2')
        print(np.sum(np.abs(Y_hat.reshape((-1,1,1,4))-Y.reshape((-1,1,1,4)))))


        