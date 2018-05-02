from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

import Helper
import ILModel

tf.control_flow_ops = tf

# data1-24
# number_of_samples_per_epoch = 157472
# number_of_validation_samples = 11488

# data1-data6
# number_of_samples_per_epoch = 39360
# number_of_validation_samples = 7872

# test
# number_of_samples_per_epoch = 11
# number_of_validation_samples = 7

# LSTM


if __name__ == '__main__':
    learning_rate = 1e-4
    batch_size = 64

    model_input = Input(tensor=train_images_batch)

    # model = CarCloneModel()
    # model, w, h = InceptionV3Model()
    model, w, h = ILModel.CarCloneModelWithLSTM()
    # create two generators for training and validation
    train_generator = Helper.generate_next_batch()
    validation_generator = Helper.generate_next_validate_batch()

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator()

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
            "./one_hot_data",
            target_size=(w, h),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
            "./one_hot_validate_data",
            target_size=(w, h),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True)

    filepath = "./models/weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1
    )
    callbacks_list = [checkpoint]

    history = model.fit_generator(train_generator,
                                  steps_per_epoch = number_of_samples_per_epoch / global_batch_size,
                                  nb_epoch = number_of_epochs,
                                  validation_data = validation_generator,
                                  validation_steps = number_of_validation_samples / global_batch_size,
                                  verbose = 1,
                                  workers = 32,
                                  callbacks = callbacks_list)

    # finally save our model and weights
    helper.save_model(model)