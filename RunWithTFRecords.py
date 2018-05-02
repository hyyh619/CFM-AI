import platform
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Dense, Reshape, GlobalAveragePooling2D
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.backend.tensorflow_backend import set_session
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv3D
from keras.layers import LSTM
from keras import backend as K
from keras.utils import np_utils

import KerasTFRecord as ktfr
import Helper
import ILModel
import History
import TrainingDefines


# config = tf.ConfigProto()
# config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

tf.control_flow_ops = tf

def ShowTrainingResult(history):
    x = range(len(history.losses))
    y1 = history.losses
    y2 = history.acc
    plt.plot(x, y1, marker='o', mec='r', mfc='w', label=u'loss')
    plt.plot(x, y2, marker='*', ms=10, label=u'acc')
    plt.xlabel(u"batches")      # X-axis label
    plt.ylabel("loss")          # Y-axis label
    plt.title("A simple plot")  # title
    plt.savefig("./models/training_result.jpg")
    plt.show()
    return

if __name__ == '__main__':
    platformType = platform.system()

    classes = 7
    sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))
    K.set_session(sess)

    train_model_define_file = './models/train_model.json'
    train_model_weight_file = './models/train_model.h5'
    train_model_pic_file    = './models/train_model.png'
    test_model_define_file = './models/test_model.json'
    test_model_weight_file = './models/test_model.h5'
    test_model_pic_file    = './models/test_model.png'    

    optimizer='adam'
    # loss='mse'
    # optimizer='rmsprop'
    loss='categorical_crossentropy'
    num_threads = 8
    VALIDATE_LOG_FILE = '../ViZDoom-Dataset/validate_data50Skip2/test.csv'

    # history
    history = History.TrainHistory()

    # For inception-v3
    # modelFunc = ILModel.InceptionV3Model
    # w = 108*2
    # h = 60*2
    # bShuffle = True
    # bLayerTrainable = False
    # bUseLSTM = bLayerTrainable
    # bResize = True
    # resizeW = 224
    # resizeH = 224

    # For ResNet50Model
    # modelFunc = ILModel.ResNet50Model
    # w = 108*2
    # h = 60*2
    # bShuffle = True
    # bLayerTrainable = True
    # bUseLSTM = bLayerTrainable
    # bResize = True
    # resizeW = 224
    # resizeH = 224

    # For CarCloneModelWithLSTM False 
    modelFunc = ILModel.CarCloneModelWithLSTM
    w = 108*2
    h = 60*2
    bShuffle = True
    bUseLSTM = False
    bResize = False
    resizeW = w
    resizeH = h

    # For CarCloneModelWithLSTM True 
    # modelFunc = ILModel.CarCloneModelWithLSTM
    # w = 108
    # h = 60
    # bShuffle = True
    # bUseLSTM = True
    # bResize = False
    # resizeW = w
    # resizeH = h

    min_after_dequeue = 8
    capacity = min_after_dequeue + 3 * TrainingDefines.BATCH_SIZE
    bTrain = True
    bUseSkip2 = True
    if bTrain:
        number_of_epochs = 20
        if bUseSkip2:
            number_of_samples_per_epoch = 369856
            number_of_validation_samples = 15680
            fileList = ["../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train0.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train1.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train2.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train3.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train4.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train5.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train6.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train7.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train8.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train9.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train10.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train11.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train12.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train13.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train14.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train15.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train16.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train17.tfrecords",
                        "../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train18.tfrecords"]
        else:
            # data1-50, validate51-53
            number_of_samples_per_epoch = 385856
            number_of_validation_samples = 15744
            fileList = ["../ViZDoom-Dataset/tfdata/train0.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train1.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train2.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train3.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train4.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train5.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train6.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train7.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train8.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train9.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train10.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train11.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train12.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train13.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train14.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train15.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train16.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train17.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train18.tfrecords",
                        "../ViZDoom-Dataset/tfdata/train19.tfrecords"]
        fileTestList = ["../ViZDoom-Dataset/tfdata/test0.tfrecords"]
    else :
        number_of_samples_per_epoch = 3000
        number_of_validation_samples = 500
        number_of_epochs = 5
        fileList = ["../ViZDoom-Dataset/tfdata100Skip2Merge4_newAction/train0.tfrecords"]

    steps_of_validation = number_of_validation_samples // TrainingDefines.BATCH_SIZE
    steps_of_train = number_of_samples_per_epoch // TrainingDefines.BATCH_SIZE

    x_train_, y_train_ = ktfr.read_and_decode(fileList, w, h, one_hot=True, n_class=classes, 
                                              is_train=True, bResize=bResize, resizeW=resizeW, resizeH=resizeH)

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

    # model_input = Input(tensor=train_images_batch)
    model_input = Input(tensor=train_images_batch)
    model_output = modelFunc(model_input, "train_out", classes, "softmax", bUseLSTM, bTrain)
    train_model = Model(inputs=model_input, outputs=model_output)
    ktfr.compile_tfrecord(train_model,
                          optimizer=optimizer,
                          loss=loss,
                          out_tensor_lst=[train_labels_batch],
                          metrics=['accuracy'])
    train_model.summary()
    if platformType == 'Windows':
        plot_model(train_model, to_file=train_model_pic_file)

    ktfr.fit_tfrecord(train_model, number_of_samples_per_epoch, TrainingDefines.BATCH_SIZE, number_of_epochs, callbacks=[history])
    Helper.save_model(train_model, train_model_define_file, train_model_weight_file)
    K.clear_session()

    # validate
    # validation_generator1 = Helper.generate_next_validate_batch(VALIDATE_LOG_FILE, TrainingDefines.BATCH_SIZE, True, classes)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator()

    validation_generator2 = test_datagen.flow_from_directory(
            "../ViZDoom-Dataset/one_hot_validate_data100Skip2Merge4_newAction",
            target_size=(resizeW, resizeH),
            batch_size=TrainingDefines.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False)

    test_model_input = Input(batch_shape=(None,)+(w, h, 3))
    test_model_output = modelFunc(test_model_input, "test_out", classes, "softmax", bUseLSTM, bTrain)
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