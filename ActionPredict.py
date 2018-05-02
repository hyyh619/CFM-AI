import argparse
import base64
import json
import os
from io import BytesIO

import eventlet.wsgi
import cv2
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json

import Logger

ACTION_NAME = ['ATTACK', # 0
               'TURN_LEFT',
               'TURN_LEFT+ATTACK',
               'TURN_RIGHT',
               'TURN_RIGHT+ATTACK',
               'MOVE_FORWARD', # 5
               'MOVE_FORWARD+ATTACK',
               'MOVE_FORWARD+TURN_LEFT',
               'MOVE_FORWARD+TURN_LEFT+ATTACK',
               'MOVE_FORWARD+TURN_RIGHT',
               'MOVE_FORWARD+TURN_RIGHT+ATTACK', # 10
               'MOVE_BACKWARD',
               'MOVE_BACKWARD+ATTACK',
               'MOVE_BACKWARD+TURN_LEFT',
               'MOVE_BACKWARD+TURN_LEFT+ATTACK',
               'MOVE_BACKWARD+TURN_RIGHT', # 15
               'MOVE_BACKWARD+TURN_RIGHT+ATTACK',
               'MOVE_LEFT',
               'MOVE_LEFT+ATTACK',
               'MOVE_LEFT+TURN_LEFT',
               'MOVE_LEFT+TURN_LEFT+ATTACK', # 20
               'MOVE_LEFT+TURN_RIGHT',
               'MOVE_LEFT+TURN_RIGHT+ATTACK',
               'MOVE_RIGHT',
               'MOVE_RIGHT+ATTACK',
               'MOVE_RIGHT+TURN_LEFT', # 25
               'MOVE_RIGHT+TURN_LEFT+ATTACK',
               'MOVE_RIGHT+TURN_RIGHT',
               'MOVE_RIGHT+TURN_RIGHT+ATTACK']

tf.control_flow_ops = tf


if __name__ == '__main__':
    dump_path = "./dump"
    logger = Logger.get_logger(filepath=os.path.join(dump_path, 'test.log'))

    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    test_model_define_file = "./models/CarCloneModelWithLSTM-with4Merge-data53/test_model.json"
    test_model_weight_file = "./models/CarCloneModelWithLSTM-with4Merge-data53/test_model.h5"
    test_data = "../ViZDoom-Dataset/data4Merge/test.csv"

    with open(test_model_define_file, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile(optimizer, loss)
    model.load_weights(test_model_weight_file)
    model.summary()

    data = pd.read_csv(test_data)
    num_of_img = len(data)
    for index in range(num_of_img):
        img = data.iloc[index]['name'].strip()
        imgData = cv2.imread(img).transpose(1, 0, 2)
        imgData = imgData.astype('float32')
        imgData = (imgData * (1. / 255)) - 0.5
        transformed_image_array = imgData[None, :, :, :]
        actions = model.predict(transformed_image_array, batch_size=1)
        action_list = np.argsort(-actions, axis=1)

        format_str = "%d:%s:%f, %d:%s:%f, %d:%s:%f" \
                      %(action_list[0][0], ACTION_NAME[action_list[0][0]], actions[0][action_list[0][0]], 
                        action_list[0][1], ACTION_NAME[action_list[0][1]], actions[0][action_list[0][1]], 
                        action_list[0][2], ACTION_NAME[action_list[0][2]], actions[0][action_list[0][2]])
        logger.info(format_str)
        m = re.findall(r'\d+',img)
        actual = int(m[2])
        if actual == action_list[0][0]:
            format_str = "%s\n" %(img)
        else:
            format_str = "err: %s: %d:%s\n" %(img, action_list[0][0], ACTION_NAME[int(m[2])])    
        logger.info(format_str)