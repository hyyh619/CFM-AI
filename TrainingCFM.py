import platform
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.utils import np_utils

import KerasTFRecord as ktfr
import Helper
import ILModel
import History
import TrainingDefines

tf.control_flow_ops = tf

def ShowTrainingResult(history):
    x = range(len(history.losses))
    y1 = history.losses
    y2 = history.acc
    y3 = history.mae
    plt.plot(x, y1, marker='o', mec='r', mfc='w', label=u'loss')
    plt.plot(x, y2, marker='*', ms=10, label=u'acc')
    plt.plot(x, y3, marker='*', ms=10, label=u'mae')
    plt.xlabel(u"batches")      # X-axis label
    plt.ylabel("loss")          # Y-axis label
    plt.title("A simple plot")  # title
    plt.savefig("./models/training_result.jpg")
    plt.show()
    return

def Training(imgW, imgH, inputW, inputH, modelName, classesNum,
             trainingList, validFolder, trainingNum, validNum, 
             epochs = 20, optimizer='adam', lossFunc='categorical_crossentropy'):
    platformType = platform.system()

    classes         = classesNum
    origImgW        = imgW
    origImgH        = imgH
    w               = inputW
    h               = inputH
    bShuffle        = True
    bLayerTrainable = True
    bResize         = True

    loss            = lossFunc
    num_threads     = 8

    config = tf.ConfigProto(device_count={'gpu':0})
    # config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    train_model_define_file = './models/train_model.json'
    train_model_weight_file = './models/train_model.h5'
    train_model_pic_file    = './models/train_model.png'
    test_model_define_file  = './models/test_model.json'
    test_model_weight_file  = './models/test_model.h5'
    test_model_pic_file     = './models/test_model.png'

    # history
    history = History.TrainHistory()

    if modelName == 'InceptionV3':
        # For inception-v3
        modelFunc = ILModel.InceptionV3Model
        bUseLSTM = bLayerTrainable
    elif modelName == 'MobileNet':
        # For MobileNet
        modelFunc = ILModel.MobileNetModel
        bUseLSTM = bLayerTrainable
    elif modelName == 'ResNet50':
        # For ResNet50Model
        # ResNet requires w is equal to h.
        modelFunc = ILModel.ResNet50Model
        bUseLSTM = bLayerTrainable
    elif modelName == 'BehavioralClone':
        # For CarCloneModelWithLSTM False 
        modelFunc = ILModel.CarCloneModelWithLSTM
        bUseLSTM = False
    else:
        print("There is no right model name: %s" %(modelName))
        return

    min_after_dequeue = 8
    capacity = min_after_dequeue + 3 * TrainingDefines.BATCH_SIZE
    bTrain = True

    if bTrain:
        number_of_epochs = epochs
        number_of_samples_per_epoch = trainingNum
        number_of_validation_samples = validNum
        fileList = trainingList

        if len(fileList) <= 0:
            print ("There is no such file")
            return

        for file in fileList:
            bExist = os.path.exists(file)
            if bExist == False:
                print ("There is no such file: %s" %(file))
                return
    else :
        bExist = os.path.exists("../CFM-Dataset/40800-Action7/tfdata_orig/train0.tfrecords")
        if bExist == False:
            print ("There is no such file")
            return

        number_of_samples_per_epoch = 1024
        number_of_validation_samples = 256
        number_of_epochs = epochs
        fileList = ["../CFM-Dataset/40800-Action7/tfdata_orig/train0.tfrecords"]

    steps_of_validation = number_of_validation_samples // TrainingDefines.BATCH_SIZE
    steps_of_train = number_of_samples_per_epoch // TrainingDefines.BATCH_SIZE

    x_train_, y_train_ = ktfr.read_and_decode(fileList, w, h, one_hot=True, n_class=classes,
                                              is_train=True, bResize=bResize, origImgW=origImgW, origImgH=origImgH)

    if bShuffle == False:
        train_images_batch, train_labels_batch = K.tf.train.batch([x_train_, y_train_],
                                                                  batch_size=TrainingDefines.BATCH_SIZE,
                                                                  capacity=capacity,
                                                                  num_threads=num_threads) # set the number of threads here
    else:
        train_images_batch, train_labels_batch = K.tf.train.shuffle_batch([x_train_, y_train_],
                                                                          batch_size=TrainingDefines.BATCH_SIZE,
                                                                          capacity=capacity,
                                                                          num_threads=num_threads,
                                                                          min_after_dequeue = min_after_dequeue) # set the number of threads here

    model_input = Input(tensor=train_images_batch)
    model_output = modelFunc(model_input, "train_out", classes, "softmax", bUseLSTM, bTrain)
    train_model  = Model(inputs=model_input, outputs=model_output)

    ktfr.compile_tfrecord(train_model,
                          optimizer=optimizer,
                          loss=loss,
                          out_tensor_lst=[train_labels_batch],
                          metrics=['accuracy', 'mae'])

    # Output train model information
    train_model.summary()
    if platformType == 'Windows':
        plot_model(train_model, to_file=train_model_pic_file)

    ktfr.fit_tfrecord(train_model, number_of_samples_per_epoch, TrainingDefines.BATCH_SIZE, number_of_epochs, callbacks=[history])

    Helper.save_model(train_model, train_model_define_file, train_model_weight_file)
    K.clear_session()

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator()

    validation_generator2 = test_datagen.flow_from_directory(
            validFolder,
            target_size=(w, h),
            batch_size=TrainingDefines.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
            )

    test_model_input = Input(batch_shape=(None,)+(w, h, 3))
    test_model_output = modelFunc(test_model_input, "test_out", classes, "softmax", bUseLSTM, False) 
    test_model = Model(inputs=test_model_input, outputs=test_model_output)

    test_model.load_weights(train_model_weight_file)
    test_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # test_model.summary()
    if platformType == 'Windows':
        plot_model(test_model, to_file=test_model_pic_file)

    print (steps_of_validation)
    print (validation_generator2)
    loss, acc = test_model.evaluate_generator(validation_generator2, steps_of_validation)
    print('\nTest accuracy: {0}'.format(acc))
    print('Test loss: {0}'.format(loss))

    Helper.save_model(test_model, test_model_define_file, test_model_weight_file)

    # plot loss
    ShowTrainingResult(history)

    sess.close()

    return

if __name__ == '__main__':
    fileList = ["../CFM-Dataset/40800-Action7/tfdata_orig/train0.tfrecords",
                "../CFM-Dataset/40800-Action7/tfdata_orig/train1.tfrecords",
                "../CFM-Dataset/40800-Action7/tfdata_orig/train2.tfrecords"]

    Training(imgW = 320,
             imgH = 180,
             inputW = 256,
             inputH = 256,
             modelName = 'BehavioralClone',
             classesNum = TrainingDefines.CLASSES_NUM,
             trainingList = fileList,
             validFolder = "../CFM-Dataset/40800-Action7/one_hot_validate_orig",
             trainingNum = 26057,
             validNum = 6847,
             epochs = 20)

    # Training(imgW = 320,
    #          imgH = 180,
    #          inputW = 256,
    #          inputH = 256,
    #          modelName = 'BehavioralClone',
    #          classesNum = 6,
    #          trainingList = fileList,
    #          validFolder = "../CFM-Dataset/40800-Action7/one_hot_validate_orig",
    #          trainingNum = 26057,
    #          validNum = 6847,
    #          epochs = 20)