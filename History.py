import numpy as np
import json
from keras.callbacks import Callback
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

import Helper
import TrainingDefines

train_tmp_model_define_file = './tmp/train_model.json'
train_tmp_model_weight_file = './tmp/train_model.h5' 

class TrainHistory(Callback):
    def __init__(self):
        self.losses          = []
        self.acc             = []
        self.mae             = []
        self.precision       = []
        self.fmeasure        = []
        self.batch_counter   = 0
        self.total_loss      = 0.
        self.total_acc       = 0.
        self.total_mae       = 0.
        self.total_precision = 0.
        self.total_fmeasure  = 0.   

    def on_train_begin(self, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.batch_counter += 1
        
        loss = logs.get('loss')
        if loss != None:
            self.total_loss += loss

        acc = logs.get('acc')
        if acc != None:
            self.total_acc += acc

        mae = logs.get('mean_absolute_error')
        if mae != None:
            self.total_mae += mae

        fmeasure = logs.get('fmeasure')
        if fmeasure != None:
            self.total_fmeasure += fmeasure

        precision = logs.get('precision')
        if precision != None:
            self.total_precision += precision         

        loss = self.total_loss / self.batch_counter
        acc = self.total_acc / self.batch_counter
        mae = self.total_mae / self.batch_counter
        precision = self.total_precision / self.batch_counter
        fmeasure = self.total_fmeasure / self.batch_counter

        self.losses.append(loss)
        self.acc.append(acc)
        self.mae.append(mae)
        self.fmeasure.append(fmeasure)
        self.precision.append(precision)
        return

    def on_epoch_end(self, epoch, logs=None):
        self.batch_counter = 0
        self.total_loss = 0.
        self.total_acc = 0.  

        # Helper.save_model(self.model, train_tmp_model_define_file, train_tmp_model_weight_file)
        # print("Save model define and weight files")

        # with open(train_tmp_model_define_file, 'r') as jfile:
        #     val_model = model_from_json(json.load(jfile))

        # val_model.compile("adam", "categorical_crossentropy")
        # val_model.load_weights(train_tmp_model_weight_file)
        # input_shape = val_model.input_shape

        # test_datagen = ImageDataGenerator()

        # validation_generator = test_datagen.flow_from_directory(
        #     TrainingDefines.VALIDATE_DATA_PATH,
        #     target_size=(int(input_shape[1]), int(input_shape[2])),
        #     batch_size=TrainingDefines.BATCH_SIZE,
        #     class_mode='categorical',
        #     shuffle=True)

        # loss, acc = val_model.evaluate_generator(validation_generator, TrainingDefines.STEPS_OF_VALIDATE)
        # print('\nTmp Test accuracy: %f, loss: %f' %(acc, loss))        
        return

    def on_train_end(self, logs):
        data_losses = np.array(self.losses)
        np.save("./models/losses.npy", data_losses)
        data_acc = np.array(self.acc)
        np.save("./models/acc.npy", data_acc)
        data_mae = np.array(self.mae)
        np.save("./models/mae.npy", data_mae)
        return