import platform
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Reshape, GlobalAveragePooling2D, Dropout
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

platformType = platform.system()
if platformType != 'Linux':
    from keras.applications.mobilenet import MobileNet

activation_relu = 'relu'
lstm_output_size = 128
learning_rate = 1e-4

def ConvLSTMModel():
    model = Sequential()

    #model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(60, 108, 3)))

    model.add(ConvLSTM2D(filters=24, kernel_size=(5, 5),
                         input_shape=(None, 60, 108, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=36, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())    

    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3),
                     activation='relu',
                     padding='same', data_format='channels_last'))
    model.add(Reshape((60, 108, 96), input_shape=(32, 60, 108, 3)))
    model.add(Flatten())

    # Next, five fully connected layers
    # model.add(Dense(1164))
    # model.add(Activation(activation_relu))

    model.add(Dense(100))
    model.add(Activation(activation_relu))

    model.add(Dense(50))
    model.add(Activation(activation_relu))

    model.add(Dense(10))
    model.add(Activation(activation_relu))

    model.add(Dense(1))

    model.summary()

    model.compile(optimizer=Adam(learning_rate), loss="mse", )
    # plot_model(model, to_file='model.png')
    
    return model, 60, 108

def CarCloneModel():
    # Our model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
    # Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(60, 108, 3)))

    # starts with five convolutional and maxpooling layers
    model.add(Conv2D(24, (5, 5), padding='same', strides=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(36, (5, 5), padding='same', strides=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(48, (5, 5), padding='same', strides=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

    model.add(Flatten())

    # Next, five fully connected layers
    model.add(Dense(1164))
    model.add(Activation(activation_relu))

    model.add(Dense(100))
    model.add(Activation(activation_relu))

    model.add(Dense(50))
    model.add(Activation(activation_relu))

    model.add(Dense(10))
    model.add(Activation(activation_relu))

    model.add(Dense(1))

    model.summary()

    model.compile(optimizer=Adam(learning_rate), loss="mse", )
    # plot_model(model, to_file='model.png')
    
    return model, 60, 108

def CarCloneModelWithLSTM(model_input, out_name, classes, last_activate, bUseLSTM=False, bTrain=True):
    # starts with five convolutional and maxpooling layers
    x = Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(model_input)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

    x = Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

    x = Conv2D(128, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    if bTrain:
        x = Dropout(rate=0.2)(x)

    if bUseLSTM:
        x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

        x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

        x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

        x = Reshape((400, 64), input_shape=(20, 20, 64))(x)
        x = LSTM(lstm_output_size)(x)
        x = Dense(512, activation='relu')(x)
        if bTrain:
            x = Dropout(rate=0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
    else:
        x = Conv2D(128, (3, 3), padding='same', strides=(2, 2), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

        x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
        if bTrain:
            x = Dropout(rate=0.2)(x)

        x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)

        x = Flatten()(x)
        if bTrain:
            x = Dropout(rate=0.2)(x)
        x = Dense(1024, activation='relu')(x)
        if bTrain:
            x = Dropout(rate=0.2)(x)
        x = Dense(256, activation='relu')(x)
        if bTrain:
            x = Dropout(rate=0.2)(x)
        x = Dense(128, activation='relu')(x)

    # x = Dense(10, activation='relu')(x)
    model_output = Dense(classes, activation=last_activate, name=out_name)(x)
    return model_output

# 391ms/step, GPU utility: 90%
def InceptionV3Model(model_input, out_name, classes, last_activate, bLayerTrainable=False, bTrain=True):
    # create the base pre-trained model
    base_model = InceptionV3(input_tensor=model_input, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    #x = Dense(10, activation='relu')(x)
    predictions = Dense(classes, activation=last_activate, name=out_name)(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = bLayerTrainable

    return predictions

# There is bug on vgg16 when running on windows7+gtx1070. It run into reboot.
def VGG16Model(model_input, out_name, classes, last_activate, bLayerTrainable=False, bTrain=True):
    # create the base pre-trained model
    base_model = VGG16(input_tensor=model_input, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dense(256, activation='relu')(x)
   
    x = Dense(64, activation='relu')(x)
    #x = Dense(10, activation='relu')(x)
    predictions = Dense(classes, activation=last_activate, name=out_name)(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = bLayerTrainable

    return predictions


# Loss increases always.
def VGG19Model(model_input, out_name, classes, last_activate, bLayerTrainable=False, bTrain=True):
    # create the base pre-trained model
    base_model = VGG19(input_tensor=model_input, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    #x = Dense(10, activation='relu')(x)
    predictions = Dense(classes, activation=last_activate, name=out_name)(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = bLayerTrainable

    return predictions


def ResNet50Model(model_input, out_name, classes, last_activate, bLayerTrainable=False, bTrain=True):
    # create the base pre-trained model
    base_model = ResNet50(input_tensor=model_input, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    #x = Dense(10, activation='relu')(x)
    predictions = Dense(classes, activation=last_activate, name=out_name)(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = bLayerTrainable

    return predictions


def MobileNetModel(model_input, out_name, classes, last_activate, bLayerTrainable=False, bTrain=True):
    if platformType == 'Linux':
        return

    # create the base pre-trained model
    base_model = MobileNet(input_tensor=model_input, weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    #x = Dense(10, activation='relu')(x)
    predictions = Dense(classes, activation=last_activate, name=out_name)(x)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = bLayerTrainable

    return predictions


class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.
    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.
    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.
    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)